"""
ЛР №3. Проектирование промптов и системных инструкций для LLM.

Логика повторяет подход из ЛР №2 (тот же способ вызова API через
OpenAI-совместимый клиент, замер времени ответа, накопительное
сохранение результатов в JSON), но добавляет то, что требует ЛР3:

- версии промптов (v1 / v2 / v3) хранятся отдельно в prompts/*.txt
  и разделены на system prompt (правила) и user prompt (входные данные);
- тестовый набор вынесен в tests.json, а не зашит в код;
- для версии v3 и индивидуального промпта ответ проверяется на то,
  что это синтаксически корректный JSON;
- результаты сохраняются и в results.json (полные ответы),
  и в results.csv (компактная таблица для отчёта и ручного
  выставления баллов по критериям из методички).
"""

import csv
import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI


PROMPTS_DIR = "lab03/prompts"
TESTS_PATH = "lab03/tests.json"
RESULTS_JSON_PATH = "lab03/results.json"


MAIN_PROMPT_VERSIONS = ["v1", "v2", "v3"]


TEMPERATURE = 0.2


def get_client():
    load_dotenv()
    return OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL") or None,
    )


def load_prompt(version_name):
    path = os.path.join(PROMPTS_DIR, f"prompt_{version_name}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_tests(path=TESTS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ask(client, model, system_prompt, user_prompt, temperature=TEMPERATURE):
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    elapsed = time.perf_counter() - started
    return response.choices[0].message.content, elapsed


def try_parse_json(text):
    cleaned = text.strip()
    had_fence = False
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
        had_fence = True

    try:
        parsed = json.loads(cleaned)
        return True, had_fence, parsed
    except json.JSONDecodeError:
        return False, had_fence, None


def run_main_scenario(client, model, tests_data, versions=MAIN_PROMPT_VERSIONS):
    scenario = tests_data["main_scenario"]
    results = []

    system_prompts = {v: load_prompt(v) for v in versions}

    for version in versions:
        system_prompt = system_prompts[version]
        for test in scenario["tests"]:
            user_prompt = f"Событие: {test['input']}"
            label = f"[{version} | {test['id']} | {test['type']}]"
            print(f"{label} запрос...", end=" ")

            try:
                answer, latency = ask(client, model, system_prompt, user_prompt)
                is_json_valid = None
                had_fence = None
                if version == "v3":
                    is_json_valid, had_fence, _ = try_parse_json(answer)
                print(f"OK ({latency:.1f} c)" + (
                    f", json={'valid' if is_json_valid else 'INVALID'}"
                    if is_json_valid is not None else ""
                ))
                results.append({
                    "scenario": scenario["name"],
                    "version": version,
                    "test_id": test["id"],
                    "test_type": test["type"],
                    "input": test["input"],
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "latency_s": round(latency, 3),
                    "answer": answer,
                    "is_json_valid": is_json_valid,
                    "had_markdown_fence": had_fence,
                    "error": None,
                })
            except Exception as exc:
                print(f"ОШИБКА: {exc}")
                results.append({
                    "scenario": scenario["name"],
                    "version": version,
                    "test_id": test["id"],
                    "test_type": test["type"],
                    "input": test["input"],
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "latency_s": None,
                    "answer": None,
                    "is_json_valid": None,
                    "had_markdown_fence": None,
                    "error": str(exc),
                })
    return results


def run_individual_task(client, model, tests_data):
    """Прогоняет тесты индивидуального задания через отдельный промпт."""
    task = tests_data["individual_task"]
    system_prompt = load_prompt("individual")
    results = []

    for test in task["tests"]:
        user_prompt = f"Обращение: {test['input']}"
        label = f"[individual | {test['id']} | {test['type']}]"
        print(f"{label} запрос...", end=" ")

        try:
            answer, latency = ask(client, model, system_prompt, user_prompt)
            is_json_valid, had_fence, _ = try_parse_json(answer)
            print(f"OK ({latency:.1f} c), json={'valid' if is_json_valid else 'INVALID'}")
            results.append({
                "scenario": task["name"],
                "version": "individual",
                "test_id": test["id"],
                "test_type": test["type"],
                "input": test["input"],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "latency_s": round(latency, 3),
                "answer": answer,
                "is_json_valid": is_json_valid,
                "had_markdown_fence": had_fence,
                "error": None,
            })
        except Exception as exc:
            print(f"ОШИБКА: {exc}")
            results.append({
                "scenario": task["name"],
                "version": "individual",
                "test_id": test["id"],
                "test_type": test["type"],
                "input": test["input"],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "latency_s": None,
                "answer": None,
                "is_json_valid": None,
                "had_markdown_fence": None,
                "error": str(exc),
            })
    return results


def save_results_json(new_results, json_path=RESULTS_JSON_PATH):
    """Накопительное сохранение, как в ЛР2: дописываем к уже существующим результатам."""
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            old_results = json.load(f)
    else:
        old_results = []

    all_results = old_results + new_results

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено в {json_path} (всего записей: {len(all_results)})")
    return all_results


def print_json_validity_summary(results, version):
    subset = [r for r in results if r["version"] == version and r["error"] is None]
    if not subset:
        return
    valid = sum(1 for r in subset if r["is_json_valid"])
    print(f"\nВерсия '{version}': корректный JSON в {valid}/{len(subset)} ответах.")


if __name__ == "__main__":
    client = get_client()
    model = os.getenv("LLM_MODEL")
    if not model:
        raise SystemExit("LLM_MODEL не задан в .env")

    tests_data = load_tests()

    print("Основной сценарий: анализ события ИБ")
    main_results = run_main_scenario(client, model, tests_data)

    print("\nИндивидуальное задание: классификация обращения в техподдержку")
    individual_results = run_individual_task(client, model, tests_data)

    all_new_results = main_results + individual_results
    all_results = save_results_json(all_new_results)
    print_json_validity_summary(main_results, "v3")
    print_json_validity_summary(individual_results, "individual")
