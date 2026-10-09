import json
import os
import re
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

load_dotenv()
client = OpenAI(base_url=os.getenv("LMM_BASE_URL", "http://localhost:1234/v1"), api_key=os.getenv("API_TOKEN", "lm-studio"),
                max_retries=0, timeout=900)
MODELS = {"model_1": os.getenv("LMM_MODEL_1"), "model_2": os.getenv("LMM_MODEL_2")}
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))

LABELS = ["normal", "suspicious"]
POSITIVE = "suspicious"
SAMPLE_IDS = [2, 3, 4, 10, 15, 19, 22, 23, 27, 29]   # ответы для ручной оценки и judge
HARD_IDS = [4, 23, 25, 27, 29]                        # примеры для повторных прогонов
REPEATS = 3

prompts = json.load(open("prompts.json", encoding="utf-8"))
tests = json.load(open("tests.json", encoding="utf-8"))


def call_model(model, system, user):
    """Запрос к модели. Возвращает текст ответа, время, токены и ошибку (если была)."""
    started = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        usage = resp.usage
        return {"raw": resp.choices[0].message.content or "", "latency_s": time.perf_counter() - started,
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None, "error": ""}
    except Exception as exc:
        return {"raw": "", "latency_s": time.perf_counter() - started, "input_tokens": None,
                "output_tokens": None, "error": str(exc)}


def parse(raw):
    """Ответ должен быть JSON с полем label. Иначе parse_error (такие ответы из выборки не исключаются)."""
    try:
        data = json.loads(raw)
        if data["label"] in LABELS:
            return data["label"], True, str(data.get("reason", ""))
    except Exception:
        pass
    return "parse_error", False, ""


def run_tests(model):
    results = []
    for t in tests:
        r = call_model(model, prompts["system"], f"Событие: {t['input']}")
        predicted, format_ok, reason = parse(r["raw"])
        results.append({"id": t["id"], "difficulty": t["difficulty"], "expected": t["expected_label"],
                        "predicted": predicted, "format_ok": format_ok, "reason": reason, **r})
    return results


def metrics(df):
    y_true, y_pred = df["expected"], df["predicted"]
    yt, yp = y_true == POSITIVE, y_pred == POSITIVE
    labels = LABELS + ["parse_error"]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(yt, yp, zero_division=0),
        "recall": recall_score(yt, yp, zero_division=0),
        "f1": f1_score(yt, yp, zero_division=0),
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
        "format_compliance": float(df["format_ok"].mean()),
        "latency_mean_s": float(df["latency_s"].mean()), "latency_median_s": float(df["latency_s"].median()),
        "latency_p95_s": float(df["latency_s"].quantile(0.95)),
        "input_tokens": int(df["input_tokens"].sum()), "output_tokens": int(df["output_tokens"].sum()),
        "api_errors": int((df["error"] != "").sum()),
        "accuracy_by_difficulty": {d: float((g["expected"] == g["predicted"]).mean()) for d, g in df.groupby("difficulty")},
    }


def repeat_hard(model):
    """Самые сложные примеры запускаются ещё REPEATS раз: меняется ли класс."""
    out = {}
    for t in tests:
        if t["id"] in HARD_IDS:
            runs = []
            for _ in range(REPEATS):
                predicted, format_ok, reason = parse(call_model(model, prompts["system"], f"Событие: {t['input']}")["raw"])
                runs.append({"predicted": predicted, "format_ok": format_ok, "reason": reason})
            out[t["id"]] = runs
    return out


def judge(judge_model, answers):
    """Judge оценивает ответ по рубрике 0-2 (4 критерия). Возвращает список оценок."""
    scores = []
    for a in answers:
        t = next(x for x in tests if x["id"] == a["id"])
        user = json.dumps({"запрос": f"Событие: {t['input']}. Определи normal или suspicious и кратко объясни, "
                                     "ответ строго в JSON {label, confidence, reason}.",
                           "ответ_модели": a["raw"], "эталонная_метка": t["expected_label"]}, ensure_ascii=False)
        raw = call_model(judge_model, prompts["judge"], user)["raw"]
        try:
            d = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            scores.append({"id": a["id"], "factual_correctness": d["factual_correctness"], "completeness": d["completeness"],
                           "instruction_following": d["instruction_following"], "relevance": d["relevance"],
                           "rationale": d.get("rationale", "")})
        except Exception:
            scores.append({"id": a["id"], "error": "judge вернул неразбираемый ответ", "raw": raw})
    return scores


def main():
    for key, model in MODELS.items():
        if not model:
            raise SystemExit(f"В .env не задано имя модели для {key} (LMM_MODEL_1 / LMM_MODEL_2)")
    try:
        loaded = [m.id for m in client.models.list().data]
    except Exception as exc:
        raise SystemExit(f"LM Studio недоступен: {exc}")
    for key, model in MODELS.items():
        if model not in loaded:
            print(f"Внимание: модель {model} ({key}) не найдена среди доступных на сервере: {loaded}")
    results = {}
    for key, model in MODELS.items():
        print(f"{key} ({model}): прогон {len(tests)} примеров...")
        rows = run_tests(model)
        print(f"{key}: повторные прогоны сложных примеров...")
        results[key] = {"model": model, "results": rows, "metrics": metrics(pd.DataFrame(rows)), "repeats": repeat_hard(model)}
    print("judge: оценка ответов (перекрёстная)...")
    for key, other in (("model_1", "model_2"), ("model_2", "model_1")):
        sample = [r for r in results[key]["results"] if r["id"] in SAMPLE_IDS]
        results[key]["judge"] = {"judge_model": MODELS[other], "scores": judge(MODELS[other], sample)}

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({"date": datetime.now().isoformat(timespec="seconds"), "prompt_version": prompts["version"], "system_prompt": prompts["system"],
                   "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "models": results}, f, ensure_ascii=False, indent=2)

    print("\n| Показатель | model_1 | model_2 |\n|---|---|---|")
    for title, key in [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"),
                       ("Format compliance", "format_compliance"), ("Средняя latency, с", "latency_mean_s"),
                       ("Медианная latency, с", "latency_median_s"), ("p95 latency, с", "latency_p95_s"),
                       ("Входные токены", "input_tokens"), ("Выходные токены", "output_tokens"), ("Ошибки API", "api_errors")]:
        v1, v2 = (results[k]["metrics"][key] for k in ("model_1", "model_2"))
        fmt = lambda v: str(v) if isinstance(v, int) else f"{v:.3f}"
        print(f"| {title} | {fmt(v1)} | {fmt(v2)} |")
    for k in ("model_1", "model_2"):
        cm = results[k]["metrics"]["confusion_matrix"]
        print(f"\nConfusion matrix {k} (строки: эталон, столбцы: ответ): {cm['labels']}")
        for lab, row in zip(cm["labels"], cm["matrix"]):
            print(f"  {lab}: {row}")
        print("Accuracy по сложности:", {d: round(v, 2) for d, v in results[k]["metrics"]["accuracy_by_difficulty"].items()})
    print("\nПодробные результаты: results.json")


if __name__ == "__main__":
    main()
