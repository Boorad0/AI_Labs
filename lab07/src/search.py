r"""Шаг 3: поиск top-k по смыслу.
Запуск: python src\search.py "что делать если украли пароль"
С правами доступа: python src\search.py "..." --role staff   (staff видит только документы с access=all)"""
import argparse

import numpy as np

from common import ROOT, TOP_K, embed, read_jsonl


def search(query, corpus, vectors, k=TOP_K):
    """Возвращает k лучших документов: (индекс, сходство). Векторы нормализованы, поэтому сходство = скалярное произведение."""
    q = embed([query])[0]
    scores = vectors @ q
    order = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--role", choices=["security", "staff"], default="security")
    args = parser.parse_args()

    corpus = read_jsonl("corpus.jsonl")
    vectors = np.load(ROOT / "results" / "corpus_vectors.npy")
    if args.role == "staff":                                  # фильтр прав применяется ДО поиска
        keep = [i for i, d in enumerate(corpus) if d["access"] == "all"]
        corpus = [corpus[i] for i in keep]
        vectors = vectors[keep]
    for i, score in search(args.query, corpus, vectors, args.k):
        d = corpus[i]
        print(f"{score:.3f}  {d['id']}  [{d['category']}]  {d['text']}")


if __name__ == "__main__":
    main()
