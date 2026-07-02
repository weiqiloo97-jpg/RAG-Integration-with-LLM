#connects to my_local_database
#fetches all 2,396 documents and their metadata from lanagchain collection
#deletes the old corrupted langchain collection and creates a new langchain collection
#re-adds the documents in batches 

import os
import chromadb
from sentence_transformers import SentenceTransformer

def main():
    print("=" * 60)
    print("  ChromaDB Index Rebuilder - Fixing my_local_database")
    print("=" * 60)

    db_path = "./my_local_database"
    if not os.path.exists(db_path):
        print(f"Error: Database directory {db_path} does not exist!")
        return

    print(f"\n1. Connecting to database at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)
    
    # Retrieve existing collection
    try:
        collection = client.get_collection("langchain")
        original_count = collection.count()
        print(f"   Found collection 'langchain' with {original_count} documents.")
    except Exception as e:
        print(f"   Error getting 'langchain' collection: {e}")
        return

    if original_count == 0:
        print("   No documents to rebuild! Exiting.")
        return

    print("\n2. Fetching all documents and metadatas from SQLite...")
    # Fetch all data without 'embeddings' to avoid the native index error
    all_data = collection.get(include=["documents", "metadatas"])
    ids = all_data["ids"]
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    print(f"   Successfully retrieved {len(ids)} records from database.")

    print("\n3. Loading local embedding model (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("   Model loaded successfully.")

    print("\n4. Recreating 'langchain' collection...")
    # Delete the old collection to clear the corrupt HNSW files
    print("   Deleting old collection...")
    client.delete_collection("langchain")
    
    # Re-create a clean collection
    print("   Creating new collection...")
    collection = client.create_collection(
        name="langchain",
        metadata={"hnsw:space": "l2"} # Using same metric as original config
    )

    print("\n5. Re-embedding and indexing documents in batches...")
    batch_size = 100
    total = len(ids)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_ids = ids[start:end]
        batch_texts = documents[start:end]
        batch_metas = metadatas[start:end]

        print(f"   Processing batch [{start+1}–{end}/{total}]...", end="", flush=True)
        
        # Explicitly encode documents to avoid DefaultEmbeddingFunction overhead
        batch_embeddings = embed_model.encode(batch_texts, show_progress_bar=False).tolist()
        
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metas
        )
        print(" Done.")

    print(f"\nDone! Collection 'langchain' successfully rebuilt.")
    print(f"   Total documents in collection: {collection.count()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
