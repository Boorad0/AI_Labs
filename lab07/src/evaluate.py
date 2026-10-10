r"""Шаги 4–6: ключевой поиск (TF-IDF) и семантический поиск на одних и тех же запросах, метрики, эксперимент с фрагментами.
Запуск: python src\evaluate.py   (сначала python src\build_index.py)
Результаты: results/search_results.csv, results/metrics.json"""
import csv
import json
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from common import EMBED_MODEL, ROOT, TOP_K, device, embed, make_chunks, read_jsonl

# схемы разбиения длинных документов: (размер фрагмента, перекрытие) в символах
SCHEMES = {"A": (400, 80), "B": (1000, 200)}


def tfidf_ranker(texts):
    """Ключевой поиск: слова запроса и документов сравниваются без учёта смысла."""
    vectorizer = TfidfVectorizer(lowercase=True)
    matrix = vectorizer.fit_transform(texts)

    def rank(query):
        scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
        return list(np.argsort(scores)[::-1])
    return rank


def embedding_ranker(vectors):
    """Семантический поиск: запрос превращается в вектор и сравнивается со всеми векторами документов."""
    def rank(query):
        scores = vectors @ embed([query])[0]
        return list(np.argsort(scores)[::-1])
    return rank


def evaluate(set_name, method, ranker, ids, queries, rows):
    """Для каждого запроса считает Precision@k, Recall@k, ранг первого верного результата и время; добавляет строки в rows."""
    ranker(queries[0]["query"])                                # прогрев, чтобы первый запрос не искажал время
    for q in queries:
        start = time.perf_counter()
        order = ranker(q["query"])
        ms = (time.perf_counter() - start) * 1000
        relevant = set(q["relevant"])
        top = [ids[i] for i in order[:TOP_K]]
        hits = sum(1 for d in top if d in relevant)
        ranks = [n + 1 for n, i in enumerate(order) if ids[i] in relevant]
        rows.append({"set": set_name, "method": method, "query_no": q["id"], "query": q["query"],
                     "top1": top[0], "top3": " ".join(top), "first_relevant_rank": ranks[0] if ranks else "",
                     "precision": hits / TOP_K, "recall": hits / len(relevant) if relevant else 0,
                     "reciprocal_rank": 1 / ranks[0] if ranks else 0, "time_ms": round(ms, 2)})


def summarize(rows):
    result = {}
    for key in sorted({(r["set"], r["method"]) for r in rows}):
        part = [r for r in rows if (r["set"], r["method"]) == key]
        mean = lambda name: round(sum(r[name] for r in part) / len(part), 4)
        result[f"{key[0]} | {key[1]}"] = {"queries": len(part), f"precision@{TOP_K}": mean("precision"),
                                         f"recall@{TOP_K}": mean("recall"), "mrr": mean("reciprocal_rank"),
                                         "mean_time_ms": mean("time_ms")}
    return result


def main():
    rows = []

    # 1. короткие документы: ключевой поиск и эмбеддинги
    corpus = read_jsonl("corpus.jsonl")
    queries = read_jsonl("queries.jsonl")
    ids = [d["id"] for d in corpus]
    vectors = np.load(ROOT / "results" / "corpus_vectors.npy")
    evaluate("короткие документы", "ключевой (TF-IDF)", tfidf_ranker([d["text"] for d in corpus]), ids, queries, rows)
    evaluate("короткие документы", "семантический", embedding_ranker(vectors), ids, queries, rows)

    # 2. длинные документы: две схемы разбиения на фрагменты
    long_docs = read_jsonl("long_docs.jsonl")
    long_queries = read_jsonl("long_queries.jsonl")
    chunk_info = {}
    for name, (size, overlap) in SCHEMES.items():
        chunks = [(f"{d['id']}-{n}", c) for d in long_docs for n, c in enumerate(make_chunks(d["text"], size, overlap), 1)]
        ids_c = [cid for cid, _ in chunks]
        start = time.perf_counter()
        vectors_c = embed([c for _, c in chunks])
        build = time.perf_counter() - start
        # верным считается фрагмент, в котором есть фраза-ответ
        queries_c = [{"id": q["id"], "query": q["query"], "relevant": [cid for cid, c in chunks if q["answer"] in c]}
                     for q in long_queries]
        chunk_info[name] = {"size": size, "overlap": overlap, "chunks": len(chunks), "build_time_s": round(build, 3),
                            "queries_without_relevant_chunk": [q["id"] for q in queries_c if not q["relevant"]]}
        queries_c = [q for q in queries_c if q["relevant"]]
        evaluate(f"длинные документы, схема {name}", "семантический", embedding_ranker(vectors_c), ids_c, queries_c, rows)

    with open(ROOT / "results" / "search_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    metrics = {"model": EMBED_MODEL, "device": device(), "top_k": TOP_K, "chunking": chunk_info, "metrics": summarize(rows)}
    (ROOT / "results" / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
