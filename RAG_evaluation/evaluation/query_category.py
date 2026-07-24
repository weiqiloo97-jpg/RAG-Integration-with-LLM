"""
query_category.py
=================
Rule-based query classifier and per-category performance aggregator.

Query Categories
----------------
1. Version Query      — query explicitly targets a specific version number/release
2. Comparison Query   — query asks to compare two or more versions / features
3. Feature/Change Query — query asks what changed, was introduced, or was fixed
4. Fact Query         — general factual question (fallback)

Usage
-----
    from evaluation.query_category import classify_query, compute_category_metrics

    category = classify_query("What changed in Bootstrap v5.3.1?")
    # → "Feature/Change Query"

    rows = compute_category_metrics(query_results_list, queries_data)
    # → list[dict] → write to query_category_results.csv
"""

import re
from typing import List, Dict

# ---------------------------------------------------------------------------
# Pattern sets for classification (checked in priority order)
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b", re.IGNORECASE)

_COMPARISON_KEYWORDS = [
    "compare", "comparison", "vs", "versus", "difference between",
    "differ", "between versions", "which version", "better than",
]

_CHANGE_KEYWORDS = [
    "changed", "introduced", "modified", "new in", "added in",
    "fixed in", "what changed", "update to", "upgrade", "deprecated",
    "removed", "improvement", "enhancement",
]

_VERSION_QUERY_KEYWORDS = [
    "release notes", "changelog", "release", "patch notes",
    "what is in", "what was in",
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_query(query_text: str) -> str:
    """
    Classifies a query into one of four categories using keyword heuristics.

    Priority order (highest to lowest):
      1. Comparison Query
      2. Feature/Change Query
      3. Version Query
      4. Fact Query (default)

    Parameters
    ----------
    query_text : str

    Returns
    -------
    str  — one of "Comparison Query", "Feature/Change Query",
           "Version Query", "Fact Query"
    """
    q_lower = query_text.lower()

    # 1. Comparison (highest priority — must check before version/change)
    for kw in _COMPARISON_KEYWORDS:
        if kw in q_lower:
            return "Comparison Query"

    # 2. Feature/Change
    for kw in _CHANGE_KEYWORDS:
        if kw in q_lower:
            return "Feature/Change Query"

    # 3. Version Query — has explicit version number or release vocab
    if _VERSION_RE.search(q_lower):
        return "Version Query"
    for kw in _VERSION_QUERY_KEYWORDS:
        if kw in q_lower:
            return "Version Query"

    # 4. Default
    return "Fact Query"


# ---------------------------------------------------------------------------
# Per-category metric aggregator
# ---------------------------------------------------------------------------

def compute_category_metrics(
    query_results_list: List[Dict],
    queries_data: List[Dict],
) -> List[Dict]:
    """
    Groups query results by category and computes per-category retrieval metrics.

    Retrieval is considered 'successful' for a query when:
      - retrieved_chunks > 0  AND  temporal_leak == False

    This mirrors the definition used in the main evaluation pipeline.

    Parameters
    ----------
    query_results_list : list[dict]
        Rows from the Stage 1 loop (must have: query_id, query, retrieved_chunks,
        temporal_leak, retrieval_latency_ms).
    queries_data : list[dict]
        Original JSON query items (for additional metadata if needed).

    Returns
    -------
    list[dict]  — one row per category, sorted by retrieval_accuracy descending.
        Keys: category, number_of_queries, retrieval_accuracy,
              temporal_leakage_rate, average_latency_ms
    """
    # Build lookup from query_id → query_text (in case needed)
    id_to_query: Dict[str, str] = {
        item.get("query_id", ""): item.get("query", "")
        for item in queries_data
    }

    # Accumulators
    categories = ["Fact Query", "Version Query", "Comparison Query", "Feature/Change Query"]
    buckets: Dict[str, Dict] = {
        cat: {
            "total": 0,
            "successful": 0,
            "leaked": 0,
            "latencies": [],
        }
        for cat in categories
    }

    for result in query_results_list:
        q_text = result.get("query", "") or id_to_query.get(result.get("query_id", ""), "")
        category = classify_query(q_text)

        ret_chunks = int(result.get("retrieved_chunks", 0))
        temporal_leak = bool(result.get("temporal_leak", False))
        latency_ms = float(result.get("retrieval_latency_ms", 0.0))

        bucket = buckets[category]
        bucket["total"] += 1
        bucket["latencies"].append(latency_ms)

        if temporal_leak:
            bucket["leaked"] += 1

        # Successful = returned chunks AND no temporal leak
        if ret_chunks > 0 and not temporal_leak:
            bucket["successful"] += 1

    # Build output rows
    rows: List[Dict] = []
    for cat in categories:
        b = buckets[cat]
        total = b["total"]
        if total == 0:
            continue

        retrieval_accuracy = round((b["successful"] / total) * 100.0, 2)
        temporal_leakage_rate = round((b["leaked"] / total) * 100.0, 2)
        avg_latency = round(sum(b["latencies"]) / len(b["latencies"]), 3) if b["latencies"] else 0.0

        rows.append({
            "category": cat,
            "number_of_queries": total,
            "retrieval_accuracy": retrieval_accuracy,
            "temporal_leakage_rate": temporal_leakage_rate,
            "average_latency_ms": avg_latency,
        })

    # Sort by retrieval_accuracy descending (easiest → hardest)
    rows.sort(key=lambda r: r["retrieval_accuracy"], reverse=True)
    return rows
