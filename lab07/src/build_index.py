"""Шаг 2: считает эмбеддинги корпуса и сохраняет их в results/corpus_vectors.npy.
Запуск: python src\build_index.py"""
import json
import time

import numpy as np

from common import EMBED_MODEL, ROOT, device, embed, read_jsonl


def main():
    corpus = read_jsonl("corpus.jsonl")
    embed(["прогрев"])                                       # первый запрос загружает модель, в замер он не входит
    start = time.perf_counter()
    vectors = embed([d["text"] for d in corpus])
    seconds = time.perf_counter() - start

    np.save(ROOT / "results" / "corpus_vectors.npy", vectors)
    info = {"model": EMBED_MODEL, "documents": len(corpus), "dimension": int(vectors.shape[1]),
            "normalized": True, "device": device(), "build_time_s": round(seconds, 3)}
    (ROOT / "results" / "index_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(info)


if __name__ == "__main__":
    main()
