import time
import numpy as np

def is_relevant(retrieved_doc: dict, ground_truth_docs: list) -> bool:
    """
    Checks if a retrieved document matches any of the ground-truth documents.
    Supports ID matching, title matching, substring matching, and word overlap.
    """
    ret_text = retrieved_doc.get("text", "").lower()
    ret_id = retrieved_doc.get("id", "").lower()
    ret_title = retrieved_doc.get("metadata", {}).get("title", "").lower()
    
    for gt in ground_truth_docs:
        if not gt:
            continue
        gt_lower = str(gt).lower()
        
        # 1. Exact ID match
        if ret_id == gt_lower or ret_id.replace("movie_", "") == gt_lower:
            return True
            
        # 2. Title match
        if ret_title and (ret_title == gt_lower or ret_title in gt_lower or gt_lower in ret_title):
            return True
            
        # 3. Content substring match
        if gt_lower in ret_text or ret_text in gt_lower:
            return True
            
        # 4. Token overlap (Jaccard similarity for robust alignment)
        ret_words = set(w for w in ret_text.split() if len(w) > 3)
        gt_words = set(w for w in gt_lower.split() if len(w) > 3)
        if ret_words and gt_words:
            overlap = len(ret_words.intersection(gt_words)) / len(ret_words.union(gt_words))
            if overlap > 0.4:
                return True
                
    return False

def compute_cosine_similarity(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """Computes cosine similarity between two embeddings."""
    dot_product = np.dot(query_emb, doc_emb)
    norm_query = np.linalg.norm(query_emb)
    norm_doc = np.linalg.norm(doc_emb)
    if norm_query == 0 or norm_doc == 0:
        return 0.0
    return float(dot_product / (norm_query * norm_doc))

def compute_tier1_metrics(retrieved_docs: list, ground_truth_docs: list, query: str, query_time: float, embed_model) -> dict:
    """
    Computes Tier 1 retrieval metrics:
    - Hit Rate
    - Precision@5
    - Recall@5
    - MRR
    - Average Similarity
    - Average Query Time (passed in from runner)
    """
    # Restrict to top 5
    top_5 = retrieved_docs[:5]
    
    # Track which ground-truth documents (by index) are matched
    matched_gt_indices = set()
    relevant_retrieved_count = 0
    
    for doc in top_5:
        matched_any = False
        for idx, gt in enumerate(ground_truth_docs):
            # Check if this document is relevant to this specific ground-truth item
            if is_relevant(doc, [gt]):
                matched_gt_indices.add(idx)
                matched_any = True
        if matched_any:
            relevant_retrieved_count += 1
            
    # 1. Hit Rate
    hit = 1.0 if len(matched_gt_indices) > 0 else 0.0
            
    # 2. Precision@5
    precision_at_5 = relevant_retrieved_count / 5.0
    
    # 3. Recall@5
    total_relevant = len(ground_truth_docs)
    recall_at_5 = len(matched_gt_indices) / float(total_relevant) if total_relevant > 0 else 1.0
    
    # 4. MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, doc in enumerate(top_5, start=1):
        if is_relevant(doc, ground_truth_docs):
            mrr = 1.0 / rank
            break
            
    # 5. Average Similarity
    # Encode query and retrieved documents to calculate cosine similarity
    similarities = []
    if top_5 and embed_model:
        query_emb = embed_model.encode([query])[0]
        doc_texts = [doc["text"] for doc in top_5]
        doc_embs = embed_model.encode(doc_texts)
        for doc_emb in doc_embs:
            similarities.append(compute_cosine_similarity(query_emb, doc_emb))
            
    avg_similarity = float(np.mean(similarities)) if similarities else 0.0
    
    return {
        "Hit Rate": hit,
        "Precision@5": precision_at_5,
        "Recall@5": recall_at_5,
        "MRR": mrr,
        "Average Similarity": round(avg_similarity, 4),
        "Query Time": round(query_time, 4)
    }
