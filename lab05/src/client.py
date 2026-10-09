import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODELS = {"model_1": os.getenv("MODEL_1"), "model_2": os.getenv("MODEL_2")}
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
THINK = os.getenv("THINK", "").strip().lower()    

ROOT = Path(__file__).resolve().parent.parent  
prompts = json.load(open(ROOT / "prompts.json", encoding="utf-8"))


def ask(model, prompt, temperature=TEMPERATURE):
    """Запрос к локальной модели через API Ollama. Возвращает ответ, время, токены и скорость (токенов/с)."""
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
               "options": {"temperature": temperature, "num_predict": MAX_TOKENS}}
    if THINK in ("true", "false"):
        payload["think"] = THINK == "true"
    for attempt in range(2):                      # один повтор: модель может ещё загружаться
        started = time.perf_counter()
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=900)
            r.raise_for_status()
            data = r.json()
            elapsed = time.perf_counter() - started
            tokens = data.get("eval_count")                     # число сгенерированных токенов
            gen_s = data.get("eval_duration", 0) / 1e9          # время генерации, с
            text = re.sub(r"<think>.*?</think>", "", data["message"]["content"], flags=re.S).strip()
            return {"answer": text, "time_s": elapsed, "load_s": data.get("load_duration", 0) / 1e9,
                    "output_tokens": tokens, "tokens_per_s": tokens / gen_s if tokens and gen_s else None, "error": ""}
        except Exception as exc:
            error = str(exc)
            time.sleep(5)
    return {"answer": "", "time_s": time.perf_counter() - started, "load_s": None, "output_tokens": None,
            "tokens_per_s": None, "error": error}


def run_queries(model):
    """Задание 2: пять запросов разных типов."""
    return [{"type": q["type"], "prompt": q["prompt"], **ask(model, q["prompt"])} for q in prompts["queries"]]


def run_benchmark(model):
    """Задание 4: один и тот же запрос несколько раз, первый запуск отмечается отдельно."""
    runs = []
    for i in range(prompts["benchmark_runs"]):
        runs.append({"run": i + 1, "kind": "первый" if i == 0 else "повторный", **ask(model, prompts["benchmark_prompt"])})
    repeated = [r["time_s"] for r in runs[1:] if not r["error"]]
    return {"runs": runs, "mean_repeated_time_s": sum(repeated) / len(repeated) if repeated else None}


def run_temperatures(model):
    """Задание 6: творческий запрос при трёх значениях температуры (по два запуска)."""
    out = []
    for t in prompts["temperatures"]:
        for n in range(prompts["creative_repeats"]):
            out.append({"temperature": t, "attempt": n + 1, **ask(model, prompts["creative_prompt"], temperature=t)})
    return out


def save(results):
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(ROOT / "results" / "benchmark.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["модель", "№", "запуск", "время ответа, с", "загрузка модели, с", "выходных токенов", "токенов/с"])
        for key, m in results["models"].items():
            for r in m["benchmark"]["runs"]:
                w.writerow([m["model"], r["run"], r["kind"], round(r["time_s"], 2),
                            round(r["load_s"], 2) if r["load_s"] is not None else "", r["output_tokens"],
                            round(r["tokens_per_s"], 2) if r["tokens_per_s"] else ""])
    with open(ROOT / "results" / "examples.md", "w", encoding="utf-8") as f:
        for key, m in results["models"].items():
            f.write(f"# {m['model']}\n\n")
            for i, q in enumerate(m["queries"], 1):
                f.write(f"## Запрос {i} ({q['type']})\n\n{q['prompt']}\n\n**Ответ** ({q['time_s']:.1f} с):\n\n{q['answer'] or q['error']}\n\n")
            f.write("## Температура\n\n")
            for t in m["temperatures"]:
                f.write(f"**{t['temperature']}, запуск {t['attempt']}:** {t['answer'] or t['error']}\n\n")


def main():
    for key, model in MODELS.items():
        if not model:
            raise SystemExit(f"В .env не задано имя модели для {key} (MODEL_1 / MODEL_2)")
    try:
        loaded = [m["name"] for m in requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json()["models"]]
    except Exception as exc:
        raise SystemExit(f"Ollama недоступен: {exc}")
    results = {"date": datetime.now().isoformat(timespec="seconds"), "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
               "think": THINK or "по умолчанию", "models": {}}
    for key, model in MODELS.items():
        if model not in loaded and f"{model}:latest" not in loaded:   # "gemma" в Ollama то же, что "gemma:latest"
            print(f"Внимание: модель {model} ({key}) не найдена среди установленных (ollama list): {loaded}")
        print(f"\n=== {model} ===")
        print("Задание 2: пять запросов...")
        queries = run_queries(model)
        print("Задание 4: измерение производительности...")
        benchmark = run_benchmark(model)
        print("Задание 6: влияние температуры...")
        temperatures = run_temperatures(model)
        results["models"][key] = {"model": model, "queries": queries, "benchmark": benchmark, "temperatures": temperatures}
        for r in benchmark["runs"]:
            speed = f"{r['tokens_per_s']:.1f}" if r["tokens_per_s"] else "-"
            print(f"  запуск {r['run']} ({r['kind']}): {r['time_s']:.2f} с, токенов: {r['output_tokens']}, токенов/с: {speed}")
        print(f"  среднее время повторных запусков: {benchmark['mean_repeated_time_s']}")
        save(results)
    print("\nГотово: results/results.json, results/benchmark.csv, results/examples.md")


if __name__ == "__main__":
    main()
