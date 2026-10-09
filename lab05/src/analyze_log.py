import json
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL_1")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
ROOT = Path(__file__).resolve().parent.parent      # корень проекта, запуск возможен из любой папки
prompts = json.load(open(ROOT / "prompts.json", encoding="utf-8"))

path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "sample_auth.log")
log = open(path, encoding="utf-8").read()

if not MODEL:
    raise SystemExit("В .env не задано имя модели MODEL_1")
payload = {"model": MODEL, "stream": False, "options": {"temperature": 0, "num_predict": MAX_TOKENS},
           "messages": [{"role": "system", "content": prompts["log_system"]}, {"role": "user", "content": f"Журнал:\n{log}"}]}
if os.getenv("THINK", "").strip().lower() in ("true", "false"):
    payload["think"] = os.getenv("THINK").strip().lower() == "true"
try:
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=900)
    r.raise_for_status()
except Exception as exc:
    raise SystemExit(f"Ollama недоступен или модель не установлена: {exc}")
raw = re.sub(r"<think>.*?</think>", "", r.json()["message"]["content"], flags=re.S).strip()

try:
    result = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
except Exception:
    result = {"error": "ответ модели не является JSON", "raw": raw}

(ROOT / "results").mkdir(exist_ok=True)
with open(ROOT / "results" / "analysis.json", "w", encoding="utf-8") as f:
    json.dump({"file": path, "model": MODEL, "result": result}, f, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False, indent=2))
