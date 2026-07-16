import math
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the Cross-Encoder Reranker using the specified pre-trained model.
        """
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query: str, documents: list, top_k: int = 5) -> list:
        """
        Reranks a list of standard document dicts for a query.
        documents: list of dicts with {"id", "text", "metadata", "score"}
        Returns the top_k documents sorted by their reranked Cross-Encoder scores.
        """
        if not documents:
            return []
            
        pairs = [(query, doc["text"]) for doc in documents]
        ce_scores = self.model.predict(pairs)
        
        def sigmoid(x):
            try:
                return 1.0 / (1.0 + math.exp(-x))
            except OverflowError:
                return 0.0 if x < 0 else 1.0
                
        reranked_docs = []
        for i, doc in enumerate(documents):
            raw_score = float(ce_scores[i])
            normalized_score = sigmoid(raw_score)
            
            doc_copy = doc.copy()
            doc_copy["score"] = round(normalized_score, 4)
            reranked_docs.append(doc_copy)
            
        reranked_docs.sort(key=lambda x: x["score"], reverse=True)
        return reranked_docs[:top_k]
