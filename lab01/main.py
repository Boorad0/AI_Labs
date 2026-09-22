import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
LMM_BASE_URL = os.getenv("LMM_BASE_URL")
LMM_MODEL_1 = os.getenv("LMM_MODEL_1")
LMM_MODEL_2 = os.getenv("LMM_MODEL_2")

SYSTEM_PROMPT = "Ты — полезный ассистент."


REQUEST_TIMEOUT_SECONDS = 1800  

client = OpenAI(
    base_url=LMM_BASE_URL,
    api_key=API_TOKEN,
    timeout=REQUEST_TIMEOUT_SECONDS,
    max_retries=0, 
)


PROMPTS = [
    {
        "type": "Объяснение",
        "text": "Объясни студенту 4 курса разницу между хешированием и "
                "симметричным шифрованием. Не более 120 слов."
    },
    {
        "type": "Классификация",
        "text": "Определи тип события: «После 15 неудачных попыток входа "
                "с одного IP выполнен успешный вход администратора». "
                "Выбери: норма / подозрительно / критично и объясни."
    },
    {
        "type": "Суммаризация",
        "text": "Сожми следующий текст до 5 тезисов: «Информационная "
                "безопасность организации строится на трёх принципах — "
                "конфиденциальности, целостности и доступности данных. "
                "Нарушение любого из них может привести к финансовым и "
                "репутационным потерям. Для защиты применяются "
                "технические меры (шифрование, межсетевые экраны), "
                "организационные меры (политики, обучение персонала) "
                "и мониторинг инцидентов в реальном времени. Ключевую "
                "роль играет своевременное реагирование на инциденты "
                "и регулярный аудит систем.»"
    },
    {
        "type": "Структурирование",
        "text": "Извлеки из текста сущности и верни JSON с полями: ip, "
                "user, timestamp, event_type. Текст: «2024-06-01 14:32:10 "
                "пользователь admin выполнил вход с IP 192.168.1.15, "
                "событие: успешная аутентификация.»"
    },
    {
        "type": "Код",
        "text": "Напиши Python-функцию проверки SHA-256 строки "
                "(принимает строку и ожидаемый хеш, возвращает True/False)."
    },
    {
        "type": "Отладка",
        "text": "Найди ошибку в следующем фрагменте Python-кода и "
                "исправь её:\n\n"
                "def average(numbers):\n"
                "    total = 0\n"
                "    for n in numbers:\n"
                "        total =+ n\n"
                "    return total / len(numbers)"
    },
    {
        "type": "Аналитика",
        "text": "Предложи план первичного анализа подозрительного входа "
                "в корпоративную систему."
    },
    {
        "type": "ИБ",
        "text": "Назови 5 рисков использования внешней LLM для анализа "
                "внутренних документов организации."
    },
    {
        "type": "Деловой текст",
        "text": "Сформулируй краткое уведомление сотрудникам о запрете "
                "передачи паролей в AI-сервисы."
    },
    {
        "type": "Формат",
        "text": "Ответь только таблицей Markdown: риск | вероятность | "
                "ущерб | мера защиты. Оцени 4 риска AI-агента с доступом "
                "к почте."
    },
]


def ask_ai(question: str, model: str):
    started = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.7
        
    )
    elapsed = time.perf_counter() - started
    message = response.choices[0].message.content

    return message, elapsed


def run_benchmark(prompts, models):
    """Прогоняет все промпты через все модели, собирает результаты."""
    results = []

    for model in models:
        print(f"\n=== Модель: {model} ===")
        for i, item in enumerate(prompts, start=1):
            prompt_type = item["type"]
            prompt_text = item["text"]
            print(f"[{model}] Запрос {i}/{len(prompts)} ({prompt_type})...", end=" ")

            max_attempts = 2  # 1 основная попытка + 1 повтор при таймауте/сетевой ошибке
            last_error = None

            for attempt in range(1, max_attempts + 1):
                try:
                    answer, latency = ask_ai(prompt_text, model)
                    results.append({
                        "model": model,
                        "prompt_id": i,
                        "prompt_type": prompt_type,
                        "prompt_text": prompt_text,
                        "latency": round(latency, 3),
                        "answer": answer,
                        "error": None,
                    })
                    print(f"OK ({latency:.2f} с, попытка {attempt})")
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"ОШИБКА (попытка {attempt}/{max_attempts}): {exc}")
                    if attempt == max_attempts:
                        results.append({
                            "model": model,
                            "prompt_id": i,
                            "prompt_type": prompt_type,
                            "prompt_text": prompt_text,
                            "latency": None,
                            "answer": None,
                            "error": str(last_error),
                        })

    return results


def save_results_json(results, json_path="results.json"):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nРезультаты сохранены в: \n  {json_path}")



if __name__ == "__main__":
    models_to_test = [m for m in (LMM_MODEL_1, LMM_MODEL_2) if m ]

    if not models_to_test:
        raise SystemExit("Не заданы LMM_MODEL_1 / LMM_MODEL_2 в .env")

    all_results = run_benchmark(PROMPTS, models_to_test)
    save_results_json(all_results)