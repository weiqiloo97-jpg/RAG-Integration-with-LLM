from .vector_retriever import VectorRetriever
from .bm25_retriever import BM25Retriever

class HybridRetriever:
    def __init__(self, vector_retriever: VectorRetriever, bm25_retriever: BM25Retriever, alpha: float = 0.5):
        """
        Initializes the Hybrid Retriever.
        Fuses the scores of the dense vector retriever and BM25 retriever.
        alpha: weight given to dense vector search score (0.0 <= alpha <= 1.0).
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
        
    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        Retrieves top_k documents by fusing results from both Vector and BM25.
        Fuses candidates using: score = alpha * vector_score + (1 - alpha) * bm25_score.
        """
        # Fetch candidate pools from both retrievers
        # Retrieving a slightly larger pool ensures better fusion coverage
        fetch_n = max(top_k * 3, 15)
        
        vector_results = self.vector_retriever.retrieve(query, top_k=fetch_n)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=fetch_n)
        
        scores_map = {}
        docs_map = {}
        
        # Process vector results
        for doc in vector_results:
            doc_id = doc["id"]
            scores_map[doc_id] = {"vector": doc["score"], "bm25": 0.0}
            docs_map[doc_id] = doc
            
        # Process BM25 results
        for doc in bm25_results:
            doc_id = doc["id"]
            if doc_id in scores_map:
                scores_map[doc_id]["bm25"] = doc["score"]
            else:
                scores_map[doc_id] = {"vector": 0.0, "bm25": doc["score"]}
                docs_map[doc_id] = doc
                
        # Calculate combined score
        fused_results = []
        for doc_id, scores in scores_map.items():
            fused_score = self.alpha * scores["vector"] + (1.0 - self.alpha) * scores["bm25"]
            fused_results.append({
                "id": doc_id,
                "text": docs_map[doc_id]["text"],
                "metadata": docs_map[doc_id]["metadata"],
                "score": round(fused_score, 4)
            })
            
        # Sort by fused score descending and take top_k
        fused_results.sort(key=lambda x: x["score"], reverse=True)
        return fused_results[:top_k]
