"""
error_analysis.py
=================
Analyses retrieval quality and classifies each query into a failure category
to explain *why* retrieval fell short, beyond a bare accuracy number.

Failure Categories
------------------
1. No Relevant Chunk Retrieved   — retrieved_chunks == 0
2. Wrong Version Retrieved       — chunks returned but version does not match expected
3. Sparse Coverage               — fewer chunks returned than the top_k requested,
                                   indicating thin KB coverage for that query
4. Ambiguous Query               — query text is generic / lacks a version anchor
5. Wrong Chunk Retrieved         — version matches but content is a low-confidence match
6. Retrieval Correct but LLM Answer Incorrect  — Stage 2 only

Note on 'Sparse Coverage'
-------------------------
When the ChromaDB exact-version filter is used, the retriever can only return
documents that exist for a given version. Versions with little ingested content
(e.g. minor patch releases) return only 1 chunk instead of the requested 5.
This is a real quality signal: fewer chunks → less context for the LLM.

Usage
-----
    from evaluation.error_analysis import generate_error_analysis

    rows, summary = generate_error_analysis(query_results_list, queries_data)
    # rows   → list[dict] → write to error_analysis.csv
    # summary → dict  → print to console
"""

import re
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Version pattern: v<major>.<minor>.<patch> (e.g. v5.3.1, v3.5.4)
_VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b", re.IGNORECASE)

# Words that strongly anchor a query to a specific version / release
_VERSION_ANCHOR_WORDS = {
    "version", "release", "update", "patch", "changelog",
    "introduced", "modified", "changed", "new in", "fixed in",
}


def _is_ambiguous(query_text: str) -> bool:
    """
    Heuristic: a query is 'ambiguous' if it neither contains an explicit
    version number nor any version-anchor vocabulary.
    """
    q_lower = query_text.lower()
    if _VERSION_RE.search(q_lower):
        return False
    for word in _VERSION_ANCHOR_WORDS:
        if word in q_lower:
            return False
    return True


def _version_matches(expected_version: str, retrieved_version_str: str) -> bool:
    """
    Returns True if the expected version appears in the retrieved version string.
    retrieved_version_str may be comma-separated (e.g. "v5.3.1, v5.3.2").
    """
    if not retrieved_version_str or retrieved_version_str.strip().lower() in ("none", ""):
        return False
    retrieved_versions = [v.strip() for v in retrieved_version_str.split(",")]
    return expected_version.strip() in retrieved_versions


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

def classify_failure(
    query_text: str,
    expected_version: str,
    retrieved_version_str: str,
    retrieved_chunks: int,
    temporal_leak: bool,
    llm_incorrect: bool = False,
    sparse_threshold: int = 5,
) -> tuple:
    """
    Classify a single retrieval into a failure/quality category.

    Parameters
    ----------
    query_text : str
    expected_version : str
    retrieved_version_str : str
        Comma-separated string of retrieved versions.
    retrieved_chunks : int
        Number of chunks returned by the retriever.
    temporal_leak : bool
        True if at least one retrieved chunk is from a newer version.
    llm_incorrect : bool
        True when Stage-2 LLM generated an incorrect answer despite correct retrieval.
    sparse_threshold : int
        top_k requested. Retrieved < threshold is considered sparse coverage.

    Returns
    -------
    (category, notes) : Tuple[str, str]
    """
    # 1. No chunks at all
    if retrieved_chunks == 0:
        return (
            "No Relevant Chunk Retrieved",
            "Retriever returned 0 chunks — no matching content in the KB.",
        )

    # 2. Stage 2: correct retrieval but wrong LLM answer
    if llm_incorrect and _version_matches(expected_version, retrieved_version_str):
        return (
            "Retrieval Correct but LLM Answer Incorrect",
            "Retrieved the correct version chunk but the LLM generated an incorrect answer.",
        )

    # 3. Wrong version (temporal leak or explicit version mismatch)
    if temporal_leak or not _version_matches(expected_version, retrieved_version_str):
        return (
            "Wrong Version Retrieved",
            f"Expected {expected_version} but retrieved: {retrieved_version_str}.",
        )

    # 4. Sparse coverage — correct version but very few chunks returned
    if retrieved_chunks < sparse_threshold:
        if _is_ambiguous(query_text):
            return (
                "Ambiguous Query",
                f"Query lacks a version anchor; only {retrieved_chunks}/{sparse_threshold} chunks returned.",
            )
        return (
            "Sparse Coverage",
            f"Only {retrieved_chunks}/{sparse_threshold} chunks retrieved — limited KB coverage for this version.",
        )

    # 5. Ambiguous query (full chunk count but generic query)
    if _is_ambiguous(query_text):
        return (
            "Ambiguous Query",
            "Query lacks a version anchor or specific topic keyword.",
        )

    # 6. Default: correct version + full chunks but content may be off-topic
    return (
        "Wrong Chunk Retrieved",
        "Version matched and chunks returned, but content may be off-topic for the query.",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_error_analysis(
    query_results_list: List[Dict],
    queries_data: List[Dict],
    stage2_incorrect_ids: set = None,
    top_k: int = 5,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Iterates over all query results, identifies quality issues, and classifies each.

    A retrieval is considered to have a quality issue if ANY of these are true:
      - retrieved_chunks == 0          (total miss)
      - temporal_leak == True          (version leak)
      - retrieved_chunks < top_k       (sparse KB coverage — fewer docs than requested)

    The last criterion is the most impactful for version-filtered pipelines:
    when a version has only 1–2 chunks indexed, the LLM receives very little
    context, which degrades answer quality even if the version is "correct".

    Parameters
    ----------
    query_results_list : list[dict]
        Each dict must have keys: query_id, query, expected_version,
        retrieved_version, retrieved_chunks, temporal_leak, retrieval_latency_ms.
    queries_data : list[dict]
        Original query JSON items (for any additional metadata if needed).
    stage2_incorrect_ids : set[str], optional
        Set of query_ids where Stage 2 LLM gave an incorrect answer.
    top_k : int
        The top_k used during retrieval (default 5). Used to detect sparse coverage.

    Returns
    -------
    rows : list[dict]
        One row per query with a quality issue — written to error_analysis.csv.
    summary : dict[str, float]
        Category → percentage share of total issues.
    """
    if stage2_incorrect_ids is None:
        stage2_incorrect_ids = set()

    rows: List[Dict] = []

    for result in query_results_list:
        q_id = result.get("query_id", "")
        q_text = result.get("query", "")
        exp_ver = result.get("expected_version", "")
        ret_ver = result.get("retrieved_version", "none")
        ret_chunks = int(result.get("retrieved_chunks", 0))
        temporal_leak = bool(result.get("temporal_leak", False))
        llm_incorrect = q_id in stage2_incorrect_ids

        # Quality issue: no chunks, version leak, OR sparse (fewer than top_k)
        is_issue = (ret_chunks == 0) or temporal_leak or (ret_chunks < top_k)

        if not is_issue:
            continue

        category, notes = classify_failure(
            query_text=q_text,
            expected_version=exp_ver,
            retrieved_version_str=ret_ver,
            retrieved_chunks=ret_chunks,
            temporal_leak=temporal_leak,
            llm_incorrect=llm_incorrect,
            sparse_threshold=top_k,
        )

        rows.append({
            "query_id": q_id,
            "query": q_text,
            "expected_version": exp_ver,
            "retrieved_version": ret_ver,
            "retrieved_chunks": ret_chunks,
            "failure_category": category,
            "notes": notes,
        })

    # Build summary percentages
    total_issues = len(rows)
    category_counts: Dict[str, int] = {}
    for row in rows:
        cat = row["failure_category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    summary: Dict[str, float] = {}
    if total_issues > 0:
        for cat, count in category_counts.items():
            summary[cat] = round((count / total_issues) * 100.0, 1)

    return rows, summary
