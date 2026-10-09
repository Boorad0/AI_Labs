import json
import re
import shutil
import subprocess

import psutil


def parse_json(text):
    """Достаёт JSON из ответа. Если JSON нет, возвращает None."""
    try:
        return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception:
        return None


def score_label(data, check):
    """Классификация: метка должна совпасть с эталоном."""
    return 2 if str(data.get("label", "")).strip().lower() == check["expected"] else 0


def score_fields(data, check):
    """Извлечение данных: считаем, сколько полей содержат ожидаемое значение."""
    found = sum(1 for key, value in check["expected"].items() if value in str(data.get(key, "")).lower())
    if found == len(check["expected"]):
        return 2
    return 1 if found * 2 >= len(check["expected"]) else 0


def score_risk(data, check):
    """Краткий анализ: уровень риска + хотя бы одно ключевое слово в рекомендации."""
    risk_ok = str(data.get("risk_level", "")).strip().lower() == check["expected"]
    action = str(data.get("recommended_action", "")).lower()
    words_ok = any(word in action for word in check["keywords"])
    return 2 if risk_ok and words_ok else 1 if risk_ok or words_ok else 0


def score_code(data, check):
    """Код: номер строки + хотя бы одно ключевое слово в описании ошибки."""
    try:
        line_ok = int(data.get("line")) in check["lines"]
    except (TypeError, ValueError):
        line_ok = False
    problem = str(data.get("problem", "")).lower()
    words_ok = any(word in problem for word in check["keywords"])
    return 2 if line_ok and words_ok else 1 if line_ok or words_ok else 0


SCORERS = {"label": score_label, "fields": score_fields, "risk": score_risk, "code": score_code}


def score(check, text):
    """Итоговая оценка ответа. Если ответ не является JSON, оценка 0."""
    data = parse_json(text)
    if data is None:
        return 0
    return SCORERS[check["kind"]](data, check)


def gpu_usage():
    """Видеопамять (МиБ) и загрузка GPU (%) через nvidia-smi. Если утилиты нет, возвращает None."""
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
        memory, load = out.split(",")
        return float(memory), float(load)
    except Exception:
        return None, None


def watch_resources(stop_event, peaks):
    """Работает в отдельном потоке: раз в секунду смотрит CPU, RAM, VRAM, GPU и запоминает максимумы в словаре peaks."""
    psutil.cpu_percent()                      
    while not stop_event.wait(1):            
        memory = psutil.virtual_memory()
        peaks["cpu_percent"] = max(peaks["cpu_percent"], psutil.cpu_percent())
        peaks["ram_used_gb"] = max(peaks["ram_used_gb"], round((memory.total - memory.available) / 1024 ** 3, 2))
        vram, gpu = gpu_usage()
        if vram is not None:
            peaks["vram_mib"] = max(peaks["vram_mib"] or 0, vram)
            peaks["gpu_percent"] = max(peaks["gpu_percent"] or 0, gpu)


def cost_usd(input_tokens, output_tokens, price_in, price_out):
    """C = Nвх * Pвх + Nвых * Pвых. Цены указаны в USD за 1 млн токенов."""
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
