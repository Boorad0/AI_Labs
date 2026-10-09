import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

import charts
import metrics
import providers

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
data = json.load(open(ROOT / "data" / "test_cases.json", encoding="utf-8"))
SYSTEM = data["system"]
TESTS = data["tests"]

# модель -> функция вызова
MODELS = {"local": (providers.LOCAL_MODEL, providers.call_local),
          "cloud": (providers.CLOUD_MODEL, providers.call_cloud)}

CLOUD_PAUSE_S = float(os.getenv("CLOUD_PAUSE_S", "4"))

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "5"))
RETRY_PAUSE_S = float(os.getenv("RETRY_PAUSE_S", "5"))
PRICE_IN = float(os.getenv("CLOUD_PRICE_IN") or 0)       # USD за 1 млн входных токенов
PRICE_OUT = float(os.getenv("CLOUD_PRICE_OUT") or 0)     # USD за 1 млн выходных токенов


def run_one(backend, test, run_name):
    """Один запрос к одной модели. Возвращает строку для таблицы результатов."""
    model, call = MODELS[backend]
    peaks = {"cpu_percent": 0.0, "ram_used_gb": 0.0, "vram_mib": None, "gpu_percent": None}

    if backend == "local":                    
        stop = threading.Event()
        watcher = threading.Thread(target=metrics.watch_resources, args=(stop, peaks))
        watcher.start()
    else:
        peaks = dict.fromkeys(peaks)         
        time.sleep(CLOUD_PAUSE_S)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = call(SYSTEM, test["prompt"])
        if not result["error"]:
            break
        print(f"    попытка {attempt} не удалась: {result['error'][:70]}")
        time.sleep(RETRY_PAUSE_S * attempt)   

    if backend == "local":
        stop.set()
        watcher.join()

    speed = None
    if result["output_tokens"] and result["ttft_s"] is not None:
        generation_time = result["latency_s"] - result["ttft_s"]
        speed = result["output_tokens"] / generation_time if generation_time > 0 else None

    return {"test_id": test["id"], "type": test["type"], "backend": backend, "model": model, "run": run_name,
            "answer": result["text"], "ttft_s": result["ttft_s"], "latency_s": result["latency_s"],
            "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"], "tokens_per_s": speed,
            "score": 0 if result["error"] else metrics.score(test["check"], result["text"]),
            "error": result["error"], "attempts": attempt, **peaks}


def run_all():
    """Основной прогон всех тестов и повторные запуски выбранных тестов для обеих моделей."""
    rows = []
    for backend, (model, _) in MODELS.items():
        print(f"\n=== {backend}: {model} ===")
        for test in TESTS:
            row = run_one(backend, test, "main")
            rows.append(row)
            print(f"  тест {test['id']:>2}: {row['latency_s']:.2f} с, оценка {row['score']} {row['error'][:70]}")
        for test in TESTS:
            if test["id"] in data["repeat_ids"]:
                for n in range(1, data["repeat_runs"] + 1):
                    rows.append(run_one(backend, test, f"repeat{n}"))
        print("  повторные запуски выполнены")
    return pd.DataFrame(rows)


def make_summary(raw):
    """Итоговая таблица по каждой модели (только основной прогон)."""
    main = raw[raw["run"] == "main"]
    summary = []
    for backend, group in main.groupby("backend"):
        good = group[group["error"] == ""]                    # запросы без ошибок
        row = {"backend": backend, "model": group["model"].iloc[0], "tests": len(group),
               "errors": int((group["error"] != "").sum()), "requests_with_retries": int((group["attempts"] > 1).sum()),
               "mean_score": group["score"].mean(), "fully_correct_share": (group["score"] == 2).mean(),
               "latency_mean_s": good["latency_s"].mean(), "latency_median_s": good["latency_s"].median(),
               "ttft_mean_s": good["ttft_s"].mean(), "tokens_per_s_mean": good["tokens_per_s"].mean(),
               "input_tokens": good["input_tokens"].sum(), "output_tokens": good["output_tokens"].sum()}
        if backend == "local":
            row.update(cpu_percent_peak=group["cpu_percent"].max(), ram_used_gb_peak=group["ram_used_gb"].max(),
                       vram_mib_peak=group["vram_mib"].max(), gpu_percent_peak=group["gpu_percent"].max())
        else:
            total = metrics.cost_usd(row["input_tokens"], row["output_tokens"], PRICE_IN, PRICE_OUT)
            row.update(cost_usd_total=total, cost_usd_per_1000=total / len(good) * 1000 if len(good) else None,
                       cost_usd_per_100000=total / len(good) * 100000 if len(good) else None)
        summary.append(row)
    return pd.DataFrame(summary)


def save(df, name):
    df.to_csv(RESULTS / name, sep=";", index=False, encoding="utf-8-sig")


def main():
    if not providers.LOCAL_MODEL or not providers.CLOUD_MODEL:
        raise SystemExit("В .env нужно задать LOCAL_MODEL и CLOUD_MODEL")
    if not os.getenv("CLOUD_API_KEY"):
        raise SystemExit("В .env не задан CLOUD_API_KEY (ключ OpenRouter)")
    try:
        requests.get(f"{providers.LOCAL_URL}/api/tags", timeout=10).raise_for_status()
    except Exception as error:
        raise SystemExit(f"Ollama недоступен: {error}")

    RESULTS.mkdir(exist_ok=True)
    raw = run_all()
    save(raw, "raw_results.csv")

    summary = make_summary(raw)
    save(summary, "summary.csv")

    main_runs = raw[raw["run"] == "main"]
    by_type = main_runs.groupby(["backend", "type"])["score"].mean().reset_index()
    save(by_type, "summary_by_type.csv")

    repeats = raw[raw["test_id"].isin(data["repeat_ids"]) & (raw["error"] == "")]
    repeats = repeats.groupby(["backend", "test_id"])["latency_s"].agg(["count", "mean", "std", "min", "max"]).reset_index()
    save(repeats, "repeats.csv")

    config = {"date": datetime.now().isoformat(timespec="seconds"), "temperature": providers.TEMPERATURE,
              "max_tokens": providers.MAX_TOKENS, "local_think": providers.LOCAL_THINK or "по умолчанию",
              "local_model": providers.LOCAL_MODEL, "cloud_model": providers.CLOUD_MODEL,
              "price_in_usd_per_1m": PRICE_IN, "price_out_usd_per_1m": PRICE_OUT}
    json.dump(config, open(RESULTS / "config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n=== Итоги ===")
    print(summary.T.to_string())
    print("\nКачество по типам заданий:\n", by_type.pivot(index="type", columns="backend", values="score").round(2).to_string())
    print("\nПовторные запуски (задержка, с):\n", repeats.round(2).to_string(index=False))
    charts.build()
    print("\nГотово: таблицы и графики сохранены в results/")


if __name__ == "__main__":
    main()
