from .hybrid_retriever import HybridRetriever
from .reranker import Reranker

class RAGFlowRetriever:
    def __init__(self, hybrid_retriever: HybridRetriever, reranker: Reranker, vector_weight: float = 0.3, similarity_threshold: float = 0.2):
        """
        Initializes the RAGFlow-inspired Retriever.
        Uses hybrid retrieval with standard weights (0.3 vector / 0.7 keyword),
        applies a similarity threshold, and then reranks using Cross-Encoder.
        """
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.vector_weight = vector_weight
        self.similarity_threshold = similarity_threshold
        
    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        Runs the RAGFlow retrieval flow:
        1. Hybrid retrieval with vector_weight (0.3) and keyword_weight (0.7).
        2. Filters out candidate documents below similarity_threshold.
        3. Reranks the remaining candidates with Cross-Encoder.
        """
        # Save current hybrid alpha and adjust it for RAGFlow weights
        original_alpha = self.hybrid_retriever.alpha
        self.hybrid_retriever.alpha = self.vector_weight
        
        # Retrieve candidates (larger pool for fusion and pruning)
        candidates = self.hybrid_retriever.retrieve(query, top_k=max(top_k * 4, 20))
        
        # Restore hybrid alpha
        self.hybrid_retriever.alpha = original_alpha
        
        # Pruning based on similarity threshold
        pruned_candidates = [doc for doc in candidates if doc["score"] >= self.similarity_threshold]
        
        # Fallback to prevent returning empty results if threshold is too strict
        if not pruned_candidates and candidates:
            pruned_candidates = candidates[:top_k]
            
        # Rerank the filtered candidates
        reranked = self.reranker.rerank(query, pruned_candidates, top_k=top_k)
        return reranked
