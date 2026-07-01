import chromadb
client = chromadb.PersistentClient(path="./my_local_database")
collection = client.get_collection("langchain")
sample = collection.get(limit=1, include=["embeddings"])
print("OK, dimension:", len(sample["embeddings"][0]))