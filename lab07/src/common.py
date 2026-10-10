"""Общие функции: чтение данных, получение эмбеддингов из Ollama, нарезка текста на фрагменты."""
import json
import os
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
TOP_K = int(os.getenv("TOP_K", "3"))


def read_jsonl(name):
    """Читает data/<name> — по одному JSON-объекту в строке."""
    lines = (ROOT / "data" / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def embed(texts):
    """Возвращает матрицу N x D с нормализованными векторами. Тексты отправляются в Ollama пачками по 16."""
    vectors = []
    for i in range(0, len(texts), 16):
        answer = requests.post(f"{OLLAMA_URL}/api/embed",
                               json={"model": EMBED_MODEL, "input": texts[i:i + 16]}, timeout=300)
        answer.raise_for_status()
        vectors += answer.json()["embeddings"]
    matrix = np.array(vectors, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)    # длина 1: косинус = скалярное произведение


def device():
    """Где сейчас лежит модель по данным Ollama (/api/ps): GPU, CPU или частично на GPU."""
    try:
        models = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10).json().get("models", [])
    except requests.RequestException:
        return "неизвестно"
    for m in models:
        if m["name"].split(":")[0] == EMBED_MODEL.split(":")[0]:
            if m["size_vram"] == 0:
                return "CPU"
            return "GPU" if m["size_vram"] >= m["size"] else f"GPU частично ({m['size_vram'] * 100 // m['size']}% в видеопамяти)"
    return "модель не загружена"


def make_chunks(text, size, overlap):
    """Режет текст на окна по size символов; соседние окна перекрываются на overlap символов."""
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]
