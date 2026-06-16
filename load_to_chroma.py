import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_FOLDER = "sample_mflix"
CHROMA_PATH     = "./chroma_store"
BATCH_SIZE      = 100
# ─────────────────────────────────────────────────────────────────────────────

def safe_str(val):
    """Convert any MongoDB extended JSON value to a plain string."""
    if val is None:
        return ""
    if isinstance(val, dict):
        if "$oid" in val:
            return str(val["$oid"])
        if "$date" in val:
            d = val["$date"]
            if isinstance(d, dict) and "$numberLong" in d:
                return str(d["$numberLong"])
            return str(d)
        if "$binary" in val:
            return "[binary]"
        return str(val)
    if isinstance(val, list):
        return ", ".join(safe_str(v) for v in val)
    return str(val)


def extract_id(doc):
    """Pull the plain string _id from a MongoDB document."""
    raw = doc.get("_id", "")
    if isinstance(raw, dict) and "$oid" in raw:
        return raw["$oid"]
    return str(raw)


def build_movie_text(doc):
    """
    movies.json has NO plot/fullplot fields.
    Build a rich text blob from available fields for embedding.
    """
    parts = []

    title = doc.get("title", "")
    if title:
        parts.append(f"Title: {title}")

    genres = doc.get("genres", [])
    if genres:
        parts.append(f"Genres: {', '.join(genres)}")

    cast = doc.get("cast", [])
    if cast:
        parts.append(f"Cast: {', '.join(cast[:6])}")   # top 6 cast members

    directors = doc.get("directors", [])
    if directors:
        parts.append(f"Directors: {', '.join(directors)}")

    writers = doc.get("writers", [])
    if writers:
        # Strip the "(role)" suffix e.g. "Jules Renard (novel)" → "Jules Renard"
        clean_writers = [w.split("(")[0].strip() for w in writers[:4]]
        parts.append(f"Writers: {', '.join(clean_writers)}")

    countries = doc.get("countries", [])
    if countries:
        parts.append(f"Countries: {', '.join(countries)}")

    languages = doc.get("languages", [])
    if languages:
        parts.append(f"Languages: {', '.join(languages)}")

    year = doc.get("year", "")
    if year:
        parts.append(f"Year: {year}")

    awards = doc.get("awards", {})
    if isinstance(awards, dict) and awards.get("text"):
        parts.append(f"Awards: {awards['text']}")

    return ". ".join(parts) if parts else title or "unknown"


def load_movies(collection, model):
    path = os.path.join(DATABASE_FOLDER, "movies.json")
    if not os.path.exists(path):
        print(f"⚠️  {path} not found, skipping movies.")
        return

    print(f"\n📂 Loading {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    print(f"   Found {len(docs)} movie documents")

    ids, texts, metadatas = [], [], []

    for doc in docs:
        doc_id = "movie_" + extract_id(doc)
        text   = build_movie_text(doc)

        # imdb can be missing or malformed
        imdb        = doc.get("imdb", {})
        imdb_rating = 0.0
        imdb_votes  = 0
        if isinstance(imdb, dict):
            try:
                imdb_rating = float(imdb.get("rating", 0) or 0)
            except (TypeError, ValueError):
                imdb_rating = 0.0
            try:
                imdb_votes = int(imdb.get("votes", 0) or 0)
            except (TypeError, ValueError):
                imdb_votes = 0

        # tomatoes viewer rating
        tomatoes       = doc.get("tomatoes", {})
        viewer_rating  = 0.0
        viewer_meter   = 0
        if isinstance(tomatoes, dict):
            viewer = tomatoes.get("viewer", {})
            if isinstance(viewer, dict):
                try:
                    viewer_rating = float(viewer.get("rating", 0) or 0)
                except (TypeError, ValueError):
                    viewer_rating = 0.0
                try:
                    viewer_meter = int(viewer.get("meter", 0) or 0)
                except (TypeError, ValueError):
                    viewer_meter = 0

        awards = doc.get("awards", {})
        award_wins = 0
        award_nominations = 0
        if isinstance(awards, dict):
            try:
                award_wins = int(awards.get("wins", 0) or 0)
            except (TypeError, ValueError):
                award_wins = 0
            try:
                award_nominations = int(awards.get("nominations", 0) or 0)
            except (TypeError, ValueError):
                award_nominations = 0

        meta = {
            "source":             "movies",
            "title":              safe_str(doc.get("title", "")),
            "year":               int(doc["year"]) if isinstance(doc.get("year"), int) else 0,
            "genres":             safe_str(doc.get("genres", [])),
            "directors":          safe_str(doc.get("directors", [])),
            "cast":               safe_str(doc.get("cast", [])),
            "countries":          safe_str(doc.get("countries", [])),
            "languages":          safe_str(doc.get("languages", [])),
            "rated":              safe_str(doc.get("rated", "")),
            "runtime":            int(doc["runtime"]) if isinstance(doc.get("runtime"), int) else 0,
            "num_mflix_comments": int(doc.get("num_mflix_comments", 0) or 0),
            "imdb_rating":        imdb_rating,
            "imdb_votes":         imdb_votes,
            "viewer_rating":      viewer_rating,
            "viewer_meter":       viewer_meter,
            "award_wins":         award_wins,
            "award_nominations":  award_nominations,
            "type":               safe_str(doc.get("type", "movie")),
        }

        ids.append(doc_id)
        texts.append(text)
        metadatas.append(meta)

    _insert_batches(collection, model, ids, texts, metadatas, label="movies")


def load_comments(collection, model):
    path = os.path.join(DATABASE_FOLDER, "comments.json")
    if not os.path.exists(path):
        print(f"⚠️  {path} not found, skipping comments.")
        return

    print(f"\n📂 Loading {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    print(f"   Found {len(docs)} comment documents")

    ids, texts, metadatas = [], [], []

    for doc in docs:
        text = doc.get("text", "") or ""
        if not text.strip():
            continue   # skip blank comments

        doc_id = "comment_" + extract_id(doc)

        meta = {
            "source":   "comments",
            "name":     safe_str(doc.get("name", "")),
            "email":    safe_str(doc.get("email", "")),
            "movie_id": safe_str(doc.get("movie_id", "")),   # links to movie _id
            "date":     safe_str(doc.get("date", "")),
        }

        ids.append(doc_id)
        texts.append(text)
        metadatas.append(meta)

    _insert_batches(collection, model, ids, texts, metadatas, label="comments")


def _insert_batches(collection, model, ids, texts, metadatas, label):
    """Embed in batches and upsert into ChromaDB."""
    total    = len(ids)
    inserted = 0

    for start in range(0, total, BATCH_SIZE):
        end         = min(start + BATCH_SIZE, total)
        batch_ids   = ids[start:end]
        batch_texts = texts[start:end]
        batch_meta  = metadatas[start:end]

        print(f"   Embedding {label} [{start+1}–{end}/{total}] ...", end="", flush=True)
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_meta,
        )
        inserted += len(batch_ids)
        print(" ✓")

    print(f"   ✅ {inserted} {label} inserted into ChromaDB")


def main():
    print("=" * 55)
    print("  ChromaDB Loader — sample_mflix dataset")
    print("=" * 55)

    print("\n🔧 Loading embedding model (all-MiniLM-L6-v2) ...")
    print("   (first run downloads ~90 MB, instant after that)")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("   ✅ Model ready")

    print(f"\n🗄️  Connecting to ChromaDB at {CHROMA_PATH} ...")
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="mflix",
        metadata={"hnsw:space": "cosine"},
    )
    print(f"   ✅ Collection 'mflix' ready (existing docs: {collection.count()})")

    load_movies(collection, model)
    load_comments(collection, model)

    print(f"\n{'=' * 55}")
    print(f"  🎉 Done! Total docs in ChromaDB: {collection.count()}")
    print(f"  Saved to: {os.path.abspath(CHROMA_PATH)}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()