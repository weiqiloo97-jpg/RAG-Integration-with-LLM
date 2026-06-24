"""
ChromaDB RAG Evaluation — Extended (no API key required)
==========================================================
Tests: Vector search · BM25 · Hybrid · Reranking
Metrics: Precision@K · Recall@K · MRR · Hit Rate · Avg Similarity · Avg Query Time
"""

import time
import math
import json
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Config
CHROMA_PATH  = "./chroma_store"
TOP_K        = 5
HYBRID_ALPHA = 0.5   # vector weight; (1-ALPHA) goes to BM25


# CONNECTION

def connect():
    print("Loading models and connecting to ChromaDB ...")
    embed_model  = SentenceTransformer("all-MiniLM-L6-v2")
    rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    client       = chromadb.PersistentClient(path=CHROMA_PATH)
    collection   = client.get_collection("mflix")
    print(f"   Connected - {collection.count()} documents in collection\n")
    return embed_model, rerank_model, collection


# BM25 INDEX

def build_bm25_index(collection):
    print("Building BM25 index ...")
    result    = collection.get(include=["documents", "metadatas"])
    all_docs  = result["documents"]
    all_metas = result["metadatas"]
    all_ids   = result["ids"]
    tokenised = [doc.lower().split() for doc in all_docs]
    bm25      = BM25Okapi(tokenised)
    print(f"   BM25 index built over {len(all_docs)} documents\n")
    return bm25, all_docs, all_metas, all_ids


# RETRIEVAL METHODS

def vector_search(embed_model, collection, question, top_k=TOP_K, source_filter=None):
    embedding = embed_model.encode([question]).tolist()
    where     = {"source": source_filter} if source_filter else None
    results   = collection.query(
        query_embeddings=embedding, n_results=top_k, where=where,
        include=["documents", "metadatas", "distances"],
    )
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]
    scores    = [round(1 - d, 4) for d in distances]
    return docs, metas, scores


def bm25_search(bm25, all_docs, all_metas, question, top_k=TOP_K, source_filter=None):
    tokenised_query = question.lower().split()
    raw_scores      = bm25.get_scores(tokenised_query)
    indexed = [
        (i, float(raw_scores[i]))
        for i in range(len(all_docs))
        if (source_filter is None or all_metas[i].get("source") == source_filter)
    ]
    indexed.sort(key=lambda x: x[1], reverse=True)
    top       = indexed[:top_k]
    max_score = top[0][1] if top and top[0][1] > 0 else 1.0
    docs   = [all_docs[i]  for i, _ in top]
    metas  = [all_metas[i] for i, _ in top]
    scores = [round(s / max_score, 4) for _, s in top]
    return docs, metas, scores


def hybrid_search(embed_model, collection, bm25, all_docs, all_metas, all_ids,
                  question, top_k=TOP_K, source_filter=None, alpha=HYBRID_ALPHA):
    embedding  = embed_model.encode([question]).tolist()
    where      = {"source": source_filter} if source_filter else None
    fetch_n    = min(collection.count(), top_k * 4)
    vec_result = collection.query(
        query_embeddings=embedding, n_results=fetch_n, where=where,
        include=["documents", "metadatas", "distances"],
    )
    vec_ids   = vec_result["ids"][0]
    vec_dists = vec_result["distances"][0]
    vec_docs  = vec_result["documents"][0]
    vec_metas = vec_result["metadatas"][0]
    max_d     = max(vec_dists) if vec_dists else 1.0
    vec_score_map = {vid: round(1 - (d / max_d), 4) for vid, d in zip(vec_ids, vec_dists)}

    tokenised_query = question.lower().split()
    raw_bm25        = bm25.get_scores(tokenised_query)
    max_bm25        = max(raw_bm25) if max(raw_bm25) > 0 else 1.0
    bm25_score_map  = {
        all_ids[i]: round(float(raw_bm25[i]) / max_bm25, 4)
        for i in range(len(all_ids))
        if (source_filter is None or all_metas[i].get("source") == source_filter)
    }

    id_to_doc  = dict(zip(vec_ids, vec_docs))
    id_to_meta = dict(zip(vec_ids, vec_metas))
    for i, cid in enumerate(all_ids):
        id_to_doc.setdefault(cid,  all_docs[i])
        id_to_meta.setdefault(cid, all_metas[i])

    candidate_ids = set(vec_ids) | set(bm25_score_map.keys())
    fused = [
        (cid, alpha * vec_score_map.get(cid, 0.0) + (1 - alpha) * bm25_score_map.get(cid, 0.0))
        for cid in candidate_ids
    ]
    fused.sort(key=lambda x: x[1], reverse=True)
    top    = fused[:top_k]
    docs   = [id_to_doc[cid]  for cid, _ in top]
    metas  = [id_to_meta[cid] for cid, _ in top]
    scores = [round(s, 4)     for _, s  in top]
    return docs, metas, scores


def rerank_results(rerank_model, question, docs, metas, scores, top_k=TOP_K):
    if not docs:
        return docs, metas, scores
    pairs     = [(question, doc) for doc in docs]
    ce_scores = rerank_model.predict(pairs)
    sigmoid   = lambda x: 1 / (1 + math.exp(-x))
    reranked  = sorted(
        zip(docs, metas, [round(sigmoid(s), 4) for s in ce_scores]),
        key=lambda x: x[2], reverse=True,
    )[:top_k]
    r_docs, r_metas, r_scores = zip(*reranked) if reranked else ([], [], [])
    return list(r_docs), list(r_metas), list(r_scores)


# METRIC HELPERS

def is_hit(docs, metas, keyword):
    if keyword == "":
        return bool(docs)
    combined = " ".join(docs).lower()
    for m in metas:
        combined += " " + " ".join(str(v) for v in m.values()).lower()
    return keyword.lower() in combined


def precision_at_k(docs, metas, keyword, k):
    if not keyword:
        return 1.0 if docs else 0.0
    relevant = sum(
        1 for doc, meta in zip(docs[:k], metas[:k])
        if keyword.lower() in (doc + " ".join(str(v) for v in meta.values())).lower()
    )
    return relevant / min(k, len(docs))


def recall_at_k(docs, metas, keyword, total_relevant, k):
    if total_relevant == 0:
        return 1.0
    if not keyword:
        return 1.0 if docs else 0.0
    retrieved = sum(
        1 for doc, meta in zip(docs[:k], metas[:k])
        if keyword.lower() in (doc + " ".join(str(v) for v in meta.values())).lower()
    )
    return retrieved / total_relevant


def reciprocal_rank(docs, metas, keyword):
    if not keyword:
        return 1.0 if docs else 0.0
    for rank, (doc, meta) in enumerate(zip(docs, metas), start=1):
        if keyword.lower() in (doc + " ".join(str(v) for v in meta.values())).lower():
            return 1.0 / rank
    return 0.0


def count_relevant_in_collection(collection, keyword, source_filter=None):
    if not keyword:
        return 1
    where  = {"source": source_filter} if source_filter else None
    result = collection.get(where=where, include=["documents", "metadatas"])
    count  = sum(
        1 for doc, meta in zip(result["documents"], result["metadatas"])
        if keyword.lower() in (doc + " ".join(str(v) for v in meta.values())).lower()
    )
    return max(count, 1)


# EVALUATION SUITE

TEST_CASES = [
    {"query": "crime action gangster drama",          "keyword": "Crime",   "source": "movies",   "note": "Genre - Crime"},
    {"query": "romantic drama love story",            "keyword": "Drama",   "source": "movies",   "note": "Genre - Drama"},
    {"query": "comedy funny humour",                  "keyword": "Comedy",  "source": "movies",   "note": "Genre - Comedy"},
    {"query": "horror thriller suspense fear",        "keyword": "Horror",  "source": "movies",   "note": "Genre - Horror"},
    {"query": "american film united states",          "keyword": "USA",     "source": "movies",   "note": "Country - USA"},
    {"query": "french european cinema france",        "keyword": "France",  "source": "movies",   "note": "Country - France"},
    {"query": "award winning acclaimed best picture", "keyword": "win",     "source": "movies",   "note": "Awards text"},
    {"query": "great film loved this movie",          "keyword": "",        "source": "comments", "note": "Comment retrieval"},
]


def run_full_evaluation(embed_model, rerank_model, collection, bm25, all_docs, all_metas, all_ids):
    methods = {"Vector": [], "BM25": [], "Hybrid": [], "Hybrid+Rerank": []}

    print("\n" + "=" * 75)
    print("  FULL EVALUATION - All Retrieval Methods")
    print("=" * 75)

    for case in TEST_CASES:
        q, kw, src, note = case["query"], case["keyword"], case["source"], case["note"]
        total_rel = count_relevant_in_collection(collection, kw, src)

        print(f"\n  [{note}]  Query: \"{q}\"")
        print(f"  Total relevant in collection: {total_rel}")
        print("  " + "-" * 60)

        def record(name, docs, metas, scores, t):
            avg_sim = sum(scores) / len(scores) if scores else 0
            methods[name].append({
                "hit": is_hit(docs, metas, kw),
                "p":   precision_at_k(docs, metas, kw, TOP_K),
                "r":   recall_at_k(docs, metas, kw, total_rel, TOP_K),
                "rr":  reciprocal_rank(docs, metas, kw),
                "sim": avg_sim,
                "t":   t,
            })
            r = methods[name][-1]
            print(f"  {name:<16} | P@{TOP_K}: {r['p']:.2f}  R@{TOP_K}: {r['r']:.2f}"
                  f"  RR: {r['rr']:.2f}  Sim: {r['sim']:.4f}  Time: {t*1000:.1f}ms")

        t0 = time.perf_counter()
        docs_v, metas_v, scores_v = vector_search(embed_model, collection, q, source_filter=src)
        record("Vector", docs_v, metas_v, scores_v, time.perf_counter() - t0)

        t0 = time.perf_counter()
        docs_b, metas_b, scores_b = bm25_search(bm25, all_docs, all_metas, q, source_filter=src)
        record("BM25", docs_b, metas_b, scores_b, time.perf_counter() - t0)

        t0 = time.perf_counter()
        docs_h, metas_h, scores_h = hybrid_search(
            embed_model, collection, bm25, all_docs, all_metas, all_ids, q, source_filter=src)
        t_hyb = time.perf_counter() - t0
        record("Hybrid", docs_h, metas_h, scores_h, t_hyb)

        t0 = time.perf_counter()
        docs_r, metas_r, scores_r = rerank_results(rerank_model, q, docs_h, metas_h, scores_h)
        record("Hybrid+Rerank", docs_r, metas_r, scores_r, time.perf_counter() - t0 + t_hyb)

    # Summary table
    n = len(TEST_CASES)
    print("\n" + "=" * 75)
    print("  SUMMARY TABLE  (averages across all test cases)")
    print("=" * 75)
    print(f"  {'Method':<16} {'Hit Rate':>10} {'Precision@5':>12} {'Recall@5':>10}"
          f" {'MRR':>8} {'Avg Sim':>9} {'Avg Time':>10}")
    print("  " + "-" * 73)

    summary = {}
    for name, results in methods.items():
        hr    = sum(r["hit"] for r in results) / n * 100
        avg_p = sum(r["p"]   for r in results) / n
        avg_r = sum(r["r"]   for r in results) / n
        avg_m = sum(r["rr"]  for r in results) / n
        avg_s = sum(r["sim"] for r in results) / n
        avg_t = sum(r["t"]   for r in results) / n * 1000
        summary[name] = {
            "hit_rate_%": round(hr, 1),
            f"precision@{TOP_K}": round(avg_p, 4),
            f"recall@{TOP_K}":    round(avg_r, 4),
            "MRR":                round(avg_m, 4),
            "avg_similarity":     round(avg_s, 4),
            "avg_query_time_ms":  round(avg_t, 2),
        }
        print(f"  {name:<16} {hr:>9.0f}% {avg_p:>12.4f} {avg_r:>10.4f}"
              f" {avg_m:>8.4f} {avg_s:>9.4f} {avg_t:>8.1f}ms")

    print("=" * 75)
    out_path = "./rag_evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump({"system": "ChromaDB+SentenceTransformers", "results": summary}, f, indent=2)
    print(f"\n  Results saved to {out_path}\n")
    return summary


# ORIGINAL: DEMO QUERIES

def print_result_block(label, docs, metas, scores):
    print(f"\n  -- {label} --")
    for i, (doc, meta, score) in enumerate(zip(docs, metas, scores)):
        if meta.get("source") == "movies":
            print(f"  [{i+1}] {meta.get('title','?')} ({meta.get('year','?')})  Score: {score}")
            print(f"        Genres: {meta.get('genres','?')}  |  IMDB: {meta.get('imdb_rating','?')}")
        else:
            print(f"  [{i+1}] Comment by {meta.get('name','?')}  Score: {score}")
            print(f"        {doc[:100]}...")


def run_demo_queries(embed_model, rerank_model, collection, bm25, all_docs, all_metas, all_ids):
    demos = [
        ("crime drama with gangsters",        "movies"),
        ("French film from the 1930s",        "movies"),
        ("short western cowboy film",         "movies"),
        ("award winning documentary",         "movies"),
        ("user comments about a great movie", "comments"),
    ]
    print("\n" + "=" * 75)
    print("  DEMO QUERIES  (Vector / BM25 / Hybrid / Hybrid+Rerank)")
    print("=" * 75)
    for question, src in demos:
        print(f"\n" + "-" * 75)
        print(f"  Query: \"{question}\"  (source: {src})")
        docs_v, metas_v, scores_v = vector_search(embed_model, collection, question, top_k=3, source_filter=src)
        print_result_block("Vector (top 3)", docs_v, metas_v, scores_v)
        docs_b, metas_b, scores_b = bm25_search(bm25, all_docs, all_metas, question, top_k=3, source_filter=src)
        print_result_block("BM25 (top 3)", docs_b, metas_b, scores_b)
        docs_h, metas_h, scores_h = hybrid_search(
            embed_model, collection, bm25, all_docs, all_metas, all_ids, question, top_k=3, source_filter=src)
        print_result_block("Hybrid (top 3)", docs_h, metas_h, scores_h)
        docs_r, metas_r, scores_r = rerank_results(rerank_model, question, docs_h, metas_h, scores_h, top_k=3)
        print_result_block("Hybrid+Rerank (top 3)", docs_r, metas_r, scores_r)


# ORIGINAL: COMMENT LOOKUP

def get_comments_for_movie(collection, movie_oid):
    print(f"\nComments for movie_id: {movie_oid}")
    print("-" * 65)
    results = collection.get(
        where={"$and": [{"source": "comments"}, {"movie_id": movie_oid}]},
        include=["documents", "metadatas"],
    )
    if not results["ids"]:
        print("  No comments found for this movie_id.")
        return
    for doc, meta in zip(results["documents"], results["metadatas"]):
        print(f"  {meta.get('name','?')}  ({meta.get('date','?')})")
        print(f"     {doc[:200]}")
        print()


# ORIGINAL: INTERACTIVE MODE

def interactive_mode(embed_model, rerank_model, collection, bm25, all_docs, all_metas, all_ids):
    print("\n" + "=" * 75)
    print("  INTERACTIVE MODE  (type 'quit' to exit)")
    print("=" * 75)
    print("  Methods: vector | bm25 | hybrid | rerank\n")
    while True:
        try:
            q = input("Query: ").strip()
            if q.lower() in ("quit", "exit", "q", ""):
                if q.lower() in ("quit", "exit", "q"):
                    print("Bye!")
                    break
                continue
            src = input("Source filter [movies/comments/all]: ").strip().lower()
            src = src if src in ("movies", "comments") else None
            method = input("Method [vector/bm25/hybrid/rerank]: ").strip().lower()
            print()
            if method == "bm25":
                docs, metas, scores = bm25_search(bm25, all_docs, all_metas, q, source_filter=src)
            elif method == "hybrid":
                docs, metas, scores = hybrid_search(
                    embed_model, collection, bm25, all_docs, all_metas, all_ids, q, source_filter=src)
            elif method == "rerank":
                docs, metas, scores = hybrid_search(
                    embed_model, collection, bm25, all_docs, all_metas, all_ids, q, source_filter=src)
                docs, metas, scores = rerank_results(rerank_model, q, docs, metas, scores)
            else:
                docs, metas, scores = vector_search(embed_model, collection, q, source_filter=src)
            print_result_block(f"Results ({method or 'vector'})", docs, metas, scores)
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break


# MAIN

def main():
    print("=" * 75)
    print("  ChromaDB RAG Evaluation (no API key required)")
    print("=" * 75)

    embed_model, rerank_model, collection = connect()
    bm25, all_docs, all_metas, all_ids   = build_bm25_index(collection)

    run_demo_queries(embed_model, rerank_model, collection, bm25, all_docs, all_metas, all_ids)
    run_full_evaluation(embed_model, rerank_model, collection, bm25, all_docs, all_metas, all_ids)

    print("\n" + "=" * 75)
    print("  COMMENT LOOKUP by movie_id")
    print("=" * 75)
    get_comments_for_movie(collection, "573a1390f29313caabcd4323")

    interactive_mode(embed_model, rerank_model, collection, bm25, all_docs, all_metas, all_ids)


if __name__ == "__main__":
    main()