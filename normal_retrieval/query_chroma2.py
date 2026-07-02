import sys
import chromadb
from sentence_transformers import SentenceTransformer

# Reconfigure stdout to support UTF-8 on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def query_vector_db(query_text, top_k=3):
    print("=" * 60)
    print(f"Querying: \"{query_text}\"")
    print("=" * 60)

    db_path = "./my_local_database"
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("langchain")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embed_model.encode([query_text]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    if not results or not results["ids"] or not results["ids"][0]:
        print("No results found.")
        return

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    # No manual sort needed — Chroma already returns these ascending (closest first)

    for i in range(len(ids)):
        meta = metadatas[i] or {}
        distance = round(distances[i], 4)  # lower = more similar

        print(f"\n[{i+1}] ID: {ids[i]} | L2 Distance: {distance}")
        print(f"    Source File: {meta.get('source_file', 'unknown')}")
        print(f"    Page: {meta.get('source_page', 'unknown')} | Company: {meta.get('company', 'unknown')}")
        print(f"    Document Snippet:")
        doc_lines = documents[i].strip().split('\n')
        for line in doc_lines[:6]:
            print(f"      {line}")
        if len(doc_lines) > 6:
            print("      ...")
    print("\n" + "=" * 60)

def main():
    # Verification/check dimension (matching query_chroma2.py original check)
    client = chromadb.PersistentClient(path="./my_local_database")
    collection = client.get_collection("langchain")
    sample = collection.get(limit=1, include=["embeddings"])
    
    if sample and sample["embeddings"] is not None:
        print("OK, dimension:", len(sample["embeddings"][0]))
    else:
        print("Error: Could not retrieve sample embedding.")
        return

    # Run a demo query
    demo_query = "HP company notices"
    query_vector_db(demo_query, top_k=3)

if __name__ == "__main__":
    main()