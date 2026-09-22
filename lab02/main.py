import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI


EXPERIMENT = "User_promt"

SYSTEM_PROMPTS = [
    "Ты помощник по генерации идей для нового продукта или сервиса. Придумай ровно 5 вариантов названий с кратким обоснованием каждого в одном предложении. Начинай ответ сразу с пронумерованного списка, без вступления и без заключения."
]

USER_PROMPTS = [
    "1.	Предложи пять названий сервиса для автоматического анализа журналов событий информационной безопасности. Для каждого названия дай пояснение в одном предложении."
]

TEMPERATURES = [0.3]         
MAX_TOKENS_LIST = [3000]       


REPEATS = 1

JSON_PATH = "results.json"


def get_client():
    load_dotenv()
    return OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )


def ask_model(client, model, system_prompt, user_prompt, temperature, max_tokens):
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    start = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    latency = time.perf_counter() - start

    return response.choices[0].message.content, latency


def run_all(client, model):
    
    results = []

    for i, system_prompt in enumerate(SYSTEM_PROMPTS, start=1):
        system_id = f"sys_{i:02d}"

        for j, user_prompt in enumerate(USER_PROMPTS, start=1):
            user_id = f"usr_{j:02d}"

            for temperature in TEMPERATURES:
                for max_tokens in MAX_TOKENS_LIST:
                    for repeat in range(1, REPEATS + 1):
                        label = (
                            f"{system_id} | {user_id} | temp={temperature} | "
                            f"max_tokens={max_tokens} | repeat={repeat}"
                        )
                        print(f"[{label}] запрос...", end=" ")

                        try:
                            answer, latency = ask_model(
                                client, model, system_prompt, user_prompt,
                                temperature, max_tokens,
                            )
                            print(f"OK ({latency:.1f} с)")
                            results.append({
                                "model": model,
                                "experiment": EXPERIMENT,
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "system_id": system_id,
                                "system_prompt": system_prompt,
                                "user_id": user_id,
                                "user_prompt": user_prompt,
                                "repeat": repeat,
                                "latency_s": round(latency, 3),
                                "answer": answer,
                                "error": None,
                            })
                        except Exception as exc:
                            print(f"ОШИБКА: {exc}")
                            results.append({
                                "model": model,
                                "experiment": EXPERIMENT,
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "system_id": system_id,
                                "system_prompt": system_prompt,
                                "user_id": user_id,
                                "user_prompt": user_prompt,
                                "repeat": repeat,
                                "latency_s": None,
                                "answer": None,
                                "error": str(exc),
                            })
    return results


def save_results_json(new_results, json_path=JSON_PATH):
    
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            old_results = json.load(f)
    else:
        old_results = []

    all_results = old_results + new_results

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено в {json_path} (всего записей: {len(all_results)})")


if __name__ == "__main__":
    client = get_client()
    model = os.getenv("LLM_MODEL_1")
    if not model:
        raise SystemExit("LLM_MODEL_1 не задан в .env")

    results = run_all(client, model)
    save_results_json(results)