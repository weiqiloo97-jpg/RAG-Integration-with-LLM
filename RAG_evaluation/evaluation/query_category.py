"""
query_category.py
=================
Rule-based query classifier and per-category performance aggregator for
Versioned RAG evaluation.

Query Categories
----------------
1. Version-Specific Query  — Queries explicitly asking about a specific version.
2. Comparison Query        — Queries comparing multiple versions.
3. Feature/Change Query    — Queries asking about modifications, additions, or updates without a specific version anchor.
4. General Fact Query      — General technical questions without explicit version comparison.
5. Cross-Version Query     — Queries requiring information across multiple versions.

Usage
-----
    from evaluation.query_category import classify_query, compute_category_metrics, print_distribution

    category = classify_query("What CSS utilities were introduced in Bootstrap v5.3.2?")
    # → "Version-Specific Query"

    rows = compute_category_metrics(query_results_list, queries_data)
    # → list[dict] → write to query_category_results.csv

    print_distribution(rows)
    # → prints category distribution summary to console
"""

import re
import numpy as np
from typing import List, Dict

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Matches single version strings (e.g., v5.3.2, v2.4.7, 5.3.1, 3.5.4)
_VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b", re.IGNORECASE)

# Matches TWO or more distinct version strings in the same query
_MULTI_VERSION_RE = re.compile(
    r"\bv?\d+\.\d+(?:\.\d+)?\b.*\bv?\d+\.\d+(?:\.\d+)?\b", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

_COMPARISON_KEYWORDS = [
    "compare", "comparison", "vs", "versus",
    "difference between", "differences between",
    "differ", "between versions", "which version",
    "better than", "worse than", "over version",
]

_CROSS_VERSION_KEYWORDS = [
    "across versions", "across all versions", "throughout versions",
    "evolution of", "history of", "over time", "multiple versions",
    "all versions", "each version", "version to version",
]

# Change verbs — used to identify Feature/Change queries when no version string is present
_CHANGE_VERBS = [
    "introduced", "modified", "changed", "added", "fixed",
    "removed", "deprecated", "updated", "improved", "enhanced",
    "what changed", "new in", "added in", "fixed in",
]

_FEATURE_CHANGE_KEYWORDS = [
    "what changed", "what is new", "what was added",
    "recently introduced", "latest changes", "latest update",
    "update to", "upgrade", "migration",
]

# Release / version vocabulary
_VERSION_VOCAB = [
    "release notes", "changelog", "patch notes",
    "release", "what is in version", "what was in version",
]

# Descriptions for CSV output
_CATEGORY_DESCRIPTIONS = {
    "Version-Specific Query": "Queries targeting a specific document version",
    "Comparison Query": "Queries requiring comparison between versions",
    "Feature/Change Query": "Queries about introduced or modified features",
    "General Fact Query": "General technical questions",
    "Cross-Version Query": "Queries requiring information across multiple versions",
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_query(query_text: str) -> str:
    """
    Classifies a query into one of five categories using keyword heuristics.

    Priority order:
      1. Comparison Query        — explicit comparison keywords OR multiple version numbers
      2. Cross-Version Query     — cross-version / evolution keywords
      3. Version-Specific Query  — explicit single version number or release vocab
      4. Feature/Change Query    — feature / change keywords without a specific version anchor
      5. General Fact Query      — general technical question fallback

    Parameters
    ----------
    query_text : str

    Returns
    -------
    str  — category name
    """
    q_lower = query_text.lower()

    has_version = bool(_VERSION_RE.search(q_lower))
    has_multi_version = bool(_MULTI_VERSION_RE.search(q_lower))

    # 1. Comparison Query
    if has_multi_version:
        return "Comparison Query"
    for kw in _COMPARISON_KEYWORDS:
        if kw in q_lower:
            return "Comparison Query"

    # 2. Cross-Version Query
    for kw in _CROSS_VERSION_KEYWORDS:
        if kw in q_lower:
            return "Cross-Version Query"

    # 3. Version-Specific Query
    if has_version:
        return "Version-Specific Query"
    for kw in _VERSION_VOCAB:
        if kw in q_lower:
            return "Version-Specific Query"

    # 4. Feature/Change Query
    for kw in _CHANGE_VERBS:
        if kw in q_lower:
            return "Feature/Change Query"
    for kw in _FEATURE_CHANGE_KEYWORDS:
        if kw in q_lower:
            return "Feature/Change Query"

    # 5. General Fact Query
    return "General Fact Query"


# ---------------------------------------------------------------------------
# Per-category metric aggregator
# ---------------------------------------------------------------------------

_ALL_CATEGORIES = [
    "Version-Specific Query",
    "Comparison Query",
    "Feature/Change Query",
    "General Fact Query",
    "Cross-Version Query",
]


def compute_category_metrics(
    query_results_list: List[Dict],
    queries_data: List[Dict],
) -> List[Dict]:
    """
    Groups query results by category and computes per-category retrieval metrics.

    Columns produced:
      category, number_of_queries, successful_retrievals, retrieval_accuracy,
      temporal_leakage_rate, average_latency_ms, p50_latency_ms, p95_latency_ms,
      average_retrieved_chunks, description

    Parameters
    ----------
    query_results_list : list[dict]
        Rows from the Stage 1 loop.
    queries_data : list[dict]
        Original JSON query items.

    Returns
    -------
    list[dict]
        One row per category present in the queries, sorted by number_of_queries descending.
    """
    id_to_query: Dict[str, str] = {
        item.get("query_id", ""): item.get("query", "")
        for item in queries_data
    }

    buckets: Dict[str, Dict] = {
        cat: {
            "total": 0,
            "successful": 0,
            "leaked": 0,
            "latencies": [],
            "chunks": [],
        }
        for cat in _ALL_CATEGORIES
    }

    for result in query_results_list:
        q_text = (
            result.get("query", "")
            or id_to_query.get(result.get("query_id", ""), "")
        )
        category = classify_query(q_text)

        ret_chunks = int(result.get("retrieved_chunks", 0))
        temporal_leak = bool(result.get("temporal_leak", False))
        latency_ms = float(result.get("retrieval_latency_ms", 0.0))

        b = buckets[category]
        b["total"] += 1
        b["latencies"].append(latency_ms)
        b["chunks"].append(ret_chunks)

        if temporal_leak:
            b["leaked"] += 1

        if ret_chunks > 0 and not temporal_leak:
            b["successful"] += 1

    rows: List[Dict] = []
    for cat in _ALL_CATEGORIES:
        b = buckets[cat]
        total = b["total"]
        if total == 0:
            continue

        lats = np.array(b["latencies"])
        retrieval_accuracy = round((b["successful"] / total) * 100.0, 1)
        temporal_leakage_rate = round((b["leaked"] / total) * 100.0, 1)
        avg_latency = round(float(np.mean(lats)), 1)
        p50_latency = round(float(np.percentile(lats, 50)), 1)
        p95_latency = round(float(np.percentile(lats, 95)), 1)
        avg_chunks = round(float(np.mean(b["chunks"])), 1)

        rows.append({
            "category": cat,
            "number_of_queries": total,
            "successful_retrievals": b["successful"],
            "retrieval_accuracy": retrieval_accuracy,
            "temporal_leakage_rate": temporal_leakage_rate,
            "average_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "average_retrieved_chunks": avg_chunks,
            "description": _CATEGORY_DESCRIPTIONS.get(cat, ""),
        })

    rows.sort(key=lambda r: r["number_of_queries"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Console distribution printer
# ---------------------------------------------------------------------------

def print_distribution(rows: List[Dict]) -> None:
    """
    Prints a formatted category distribution summary to stdout.

    Parameters
    ----------
    rows : list[dict]
        Output of compute_category_metrics().
    """
    total_queries = sum(r["number_of_queries"] for r in rows)
    if total_queries == 0:
        print("Query Category Distribution:")
        print("--------------------------------")
        print("No queries evaluated.")
        return

    print("\nQuery Category Distribution:")
    print("--------------------------------")
    for r in rows:
        cat = r["category"]
        n = r["number_of_queries"]
        pct = round((n / total_queries) * 100.0, 1)
        pct_str = f"{int(pct)}%" if pct.is_integer() else f"{pct}%"
        print(f"{cat:<24} {n} queries ({pct_str})")
