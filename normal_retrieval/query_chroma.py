import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_store"
TOP_K       = 5
# ─────────────────────────────────────────────────────────────────────────────


def connect():
    print("🔧 Loading model and connecting to ChromaDB ...")
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("mflix")
    print(f"   ✅ Connected — {collection.count()} documents in collection\n")
    return model, collection


def query(model, collection, question: str, top_k: int = TOP_K, source_filter: str = None):
    """
    Semantic search on the collection.

    Args:
        question:      Natural language query string
        top_k:         Number of results to return
        source_filter: 'movies' | 'comments' | None (search all)
    """
    embedding = model.encode([question]).tolist()
    where     = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"🔍 Query: \"{question}\"")
    if source_filter:
        print(f"   (filtered to: {source_filter})")
    print("─" * 65)

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        score = round(1 - dist, 4)   # cosine distance → similarity

        if meta.get("source") == "movies":
            print(f"[{i+1}] 🎬  {meta.get('title', '?')} ({meta.get('year', '?')})  — Score: {score}")
            print(f"      Genres:    {meta.get('genres', '—')}")
            print(f"      Directors: {meta.get('directors', '—')}")
            print(f"      Cast:      {meta.get('cast', '—')[:80]}")
            print(f"      IMDB:      {meta.get('imdb_rating', '—')}  |  RT Viewer: {meta.get('viewer_rating', '—')} ({meta.get('viewer_meter', '—')}%)")
            print(f"      Awards:    {meta.get('award_wins', 0)} wins, {meta.get('award_nominations', 0)} nominations")
        else:
            # comment
            movie_id = meta.get("movie_id", "")
            print(f"[{i+1}] 💬  Comment by {meta.get('name', '?')}  — Score: {score}")
            print(f"      Date:     {meta.get('date', '—')}")
            print(f"      Movie ID: {movie_id}")
            print(f"      Text:     {doc[:130]}...")

        print()

    return results


def run_evaluation(model, collection):
    """
    Hit-rate evaluation — checks whether the top-K results contain
    an expected keyword. Reports hit rate and average similarity score.
    """
    print("\n" + "=" * 65)
    print("  📊 EVALUATION — Hit Rate Test")
    print("=" * 65)

    # Test cases tailored to mflix movies.json fields (no plot text available)
    # Keywords are matched against genres, cast, directors, countries, etc.
    test_cases = [
        {
            "query":   "crime action gangster drama",
            "keyword": "Crime",
            "source":  "movies",
            "note":    "Genre match — Crime"
        },
        {
            "query":   "romantic drama love story",
            "keyword": "Drama",
            "source":  "movies",
            "note":    "Genre match — Drama"
        },
        {
            "query":   "comedy funny humour",
            "keyword": "Comedy",
            "source":  "movies",
            "note":    "Genre match — Comedy"
        },
        {
            "query":   "horror thriller suspense fear",
            "keyword": "Horror",
            "source":  "movies",
            "note":    "Genre match — Horror"
        },
        {
            "query":   "american film united states hollywood",
            "keyword": "USA",
            "source":  "movies",
            "note":    "Country match — USA"
        },
        {
            "query":   "french european cinema france",
            "keyword": "France",
            "source":  "movies",
            "note":    "Country match — France"
        },
        {
            "query":   "award winning acclaimed best picture",
            "keyword": "win",
            "source":  "movies",
            "note":    "Awards text match"
        },
        {
            "query":   "great film loved this movie",
            "keyword": "",            # any comment is a valid hit
            "source":  "comments",
            "note":    "Comment retrieval (any result = hit)"
        },
    ]

    hits   = 0
    total  = len(test_cases)
    scores = []

    for case in test_cases:
        embedding = model.encode([case["query"]]).tolist()
        where     = {"source": case["source"]} if case["source"] else None

        results = collection.query(
            query_embeddings=embedding,
            n_results=TOP_K,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Build one big string of everything returned (docs + all metadata values)
        combined = " ".join(results["documents"][0]).lower()
        for m in results["metadatas"][0]:
            combined += " " + " ".join(str(v) for v in m.values()).lower()

        keyword = case["keyword"].lower()
        hit     = (keyword == "") or (keyword in combined)

        avg_score = round(
            1 - sum(results["distances"][0]) / max(len(results["distances"][0]), 1),
            4
        )

        if hit:
            hits  += 1
            status = "✅ HIT "
        else:
            status = "❌ MISS"

        scores.append(avg_score)
        print(f"{status} | Score: {avg_score:.4f} | {case['note']}")
        print(f"        Query:    \"{case['query']}\"")
        if keyword:
            print(f"        Expected: \"{case['keyword']}\" in results")
        print()

    hit_rate    = hits / total * 100
    avg_quality = sum(scores) / len(scores)

    print("─" * 65)
    print(f"  Hit Rate:          {hits}/{total} = {hit_rate:.0f}%")
    print(f"  Avg Similarity:    {avg_quality:.4f}  (higher = more confident matches)")
    print()
    if hit_rate >= 80:
        print("  🟢 Retrieval quality: GOOD")
    elif hit_rate >= 60:
        print("  🟡 Retrieval quality: FAIR")
    else:
        print("  🔴 Retrieval quality: POOR — check data loaded correctly")
    print("=" * 65)


def run_demo_queries(model, collection):
    """Live demo queries covering movies and comments."""
    print("\n" + "=" * 65)
    print("  🎬 DEMO QUERIES")
    print("=" * 65)

    demos = [
        # (question, source_filter)
        ("crime drama with gangsters",             "movies"),
        ("French film from the 1930s",             "movies"),
        ("short western cowboy film",              "movies"),
        ("award winning documentary",              "movies"),
        ("user comments about a great movie",      "comments"),
    ]

    for question, source in demos:
        query(model, collection, question, top_k=3, source_filter=source)


def get_comments_for_movie(collection, movie_oid: str):
    """
    Fetch all comments linked to a specific movie by its _id OID string.
    Useful because comments.json stores movie_id as {"$oid": "..."}.
    """
    print(f"\n💬 Comments for movie_id: {movie_oid}")
    print("─" * 65)

    results = collection.get(
        where={"$and": [{"source": "comments"}, {"movie_id": movie_oid}]},
        include=["documents", "metadatas"],
    )

    if not results["ids"]:
        print("  No comments found for this movie_id.")
        return

    for doc, meta in zip(results["documents"], results["metadatas"]):
        print(f"  👤 {meta.get('name', '?')}  ({meta.get('date', '?')})")
        print(f"     {doc[:200]}")
        print()


def main():
    print("=" * 65)
    print("  ChromaDB Query & Evaluation — sample_mflix")
    print("=" * 65)

    model, collection = connect()

    # 1. Demo queries
    run_demo_queries(model, collection)

    # 2. Structured evaluation
    run_evaluation(model, collection)

    # 3. Example: look up comments for a specific movie
    #    Replace the OID below with any _id.$oid from your movies.json
    print("\n" + "=" * 65)
    print("  🔗 COMMENT LOOKUP by movie_id")
    print("=" * 65)
    example_movie_oid = "573a1390f29313caabcd4323"   # change to any valid OID
    get_comments_for_movie(collection, example_movie_oid)

    # 4. Interactive mode
    print("\n" + "=" * 65)
    print("  💬 INTERACTIVE MODE  (type 'quit' to exit)")
    print("=" * 65)
    print("  Source options: movies | comments | (Enter for all)\n")

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
            print()
            query(model, collection, q, top_k=TOP_K, source_filter=src)

        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break


if __name__ == "__main__":
    main()