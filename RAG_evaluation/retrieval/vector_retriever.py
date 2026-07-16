import chromadb
from sentence_transformers import SentenceTransformer

class VectorRetriever:
    def __init__(self, db_path="./chroma_store", collection_name="mflix", model_name="all-MiniLM-L6-v2"):
        """
        Initializes the Vector Retriever.
        Reuses the existing database path, collection, and embedding model.
        """
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)
        self.model = SentenceTransformer(model_name)
        
    def retrieve(self, query: str, top_k: int = 5, source_filter: str = "movies") -> list:
        """
        Runs dense semantic search and returns a standardized list of document dicts.
        """
        query_embedding = self.model.encode([query]).tolist()
        
        where = {"source": source_filter} if source_filter else None
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results or not results["ids"] or not results["ids"][0]:
            return []
            
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        retrieved_docs = []
        for i in range(len(ids)):
            # convert distance to similarity score. Space is 'cosine', so similarity = 1 - distance
            similarity = 1.0 - distances[i]
            retrieved_docs.append({
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i] or {},
                "score": round(similarity, 4)
            })
            
        return retrieved_docs
