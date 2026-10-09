import json
import os
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

LOCAL_URL = os.getenv("LOCAL_URL", "http://localhost:11434")
LOCAL_MODEL = os.getenv("LOCAL_MODEL")
LOCAL_THINK = os.getenv("LOCAL_THINK", "").strip().lower()      

CLOUD_MODEL = os.getenv("CLOUD_MODEL")
cloud_client = OpenAI(base_url=os.getenv("CLOUD_BASE_URL", "https://openrouter.ai/api/v1"),
                      api_key=os.getenv("CLOUD_API_KEY") or "not-set", max_retries=0, timeout=300)


def new_result():
    return {"text": "", "ttft_s": None, "latency_s": None, "input_tokens": None, "output_tokens": None, "error": ""}


def call_local(system, prompt):
    """Локальная модель через API Ollama. Ответ приходит частями (потоком), так можно измерить время первого токена."""
    result = new_result()
    body = {"model": LOCAL_MODEL, "stream": True,
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
    if LOCAL_THINK in ("true", "false"):
        body["think"] = LOCAL_THINK == "true"

    start = time.perf_counter()
    try:
        with requests.post(f"{LOCAL_URL}/api/chat", json=body, stream=True, timeout=900) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                part = json.loads(line)
                message = part.get("message", {})
                if result["ttft_s"] is None and (message.get("content") or message.get("thinking")):
                    result["ttft_s"] = time.perf_counter() - start       
                result["text"] += message.get("content", "")
                if part.get("done"):                                      
                    result["input_tokens"] = part.get("prompt_eval_count")
                    result["output_tokens"] = part.get("eval_count")
    except Exception as error:
        result["error"] = str(error)
    result["latency_s"] = time.perf_counter() - start
    return result


def call_cloud(system, prompt):
    """Облачная модель через OpenAI-совместимый API OpenRouter, тоже потоком."""
    result = new_result()
    start = time.perf_counter()
    try:
        stream = cloud_client.chat.completions.create(
            model=CLOUD_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            stream=True, stream_options={"include_usage": True},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                thinking = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
                if result["ttft_s"] is None and (delta.content or thinking):
                    result["ttft_s"] = time.perf_counter() - start
                result["text"] += delta.content or ""
            if getattr(chunk, "usage", None):                              # статистика токенов приходит в конце
                result["input_tokens"] = chunk.usage.prompt_tokens
                result["output_tokens"] = chunk.usage.completion_tokens
    except Exception as error:
        result["error"] = str(error)
    result["latency_s"] = time.perf_counter() - start
    return result
