import numpy as np
import chromadb
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, db_path="./chroma_store", collection_name="mflix", source_filter="movies"):
        """
        Initializes the BM25 Retriever by fetching all documents from ChromaDB
        and building the BM25 index in memory.
        """
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)
        self.source_filter = source_filter
        self._build_index()
        
    def _build_index(self):
        where = {"source": self.source_filter} if self.source_filter else None
        
        # Fetch all documents and metadata from ChromaDB
        results = self.collection.get(
            where=where,
            include=["documents", "metadatas"]
        )
        
        self.ids = results["ids"]
        self.documents = results["documents"]
        self.metadatas = results["metadatas"]
        
        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        Retrieves top_k documents based on BM25 keyword score.
        Scores are normalized to the [0, 1] range relative to the maximum score in the query results.
        """
        if not self.documents:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices by score in descending order
        indices = np.argsort(scores)[::-1]
        
        retrieved_docs = []
        max_score = float(max(scores)) if len(scores) > 0 and max(scores) > 0 else 1.0
        
        for idx in indices[:top_k]:
            raw_score = float(scores[idx])
            normalized_score = raw_score / max_score if max_score > 0 else 0.0
            retrieved_docs.append({
                "id": self.ids[idx],
                "text": self.documents[idx],
                "metadata": self.metadatas[idx] or {},
                "score": round(normalized_score, 4)
            })
            
        return retrieved_docs
