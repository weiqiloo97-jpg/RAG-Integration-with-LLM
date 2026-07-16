
import chromadb

from sentence_transformers import SentenceTransformer
import sys

def log(msg):
    print(msg)
    sys.stdout.flush()

log("Connecting to chromadb...")
client = chromadb.PersistentClient(path="./my_local_database")
collection = client.get_collection("langchain")
log("Connected. Count: " + str(collection.count()))

log("Loading SentenceTransformer model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
log("Model loaded successfully.")

log("Encoding query...")
question = "HP company notices"
embedding = embed_model.encode([question]).tolist()
log("Query encoded successfully. Dimension: " + str(len(embedding[0])))

log("Running query...")
results = collection.query(
    query_embeddings=embedding,
    n_results=3,
    include=["documents", "metadatas", "distances"]
)
log("Query completed successfully!")
log(str(results))
