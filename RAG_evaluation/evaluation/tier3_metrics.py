"""
tier3_metrics.py
================
Tier 3 performance metrics specific to Versioned RAG systems.

Metrics:
  1. Temporal Leakage Rate        — fraction of queries returning chunks from
                                    a newer version than the target version.
  2. Change Detection Accuracy    — F1 of the system's ability to identify
                                    truly-changed content across consecutive
                                    version pairs (checksum vs. embedding sim).
  3. Update Efficiency            — ratio of re-embedded chunks to total chunks
                                    per version ingestion cycle.
  4. p50 / p95 / p99 Query Latency — percentile latencies from the retriever.
"""

import datetime
import sqlite3
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Version date helper (mirrors versionedrag_generator._version_date)
# ---------------------------------------------------------------------------

_VERSION_DATES = {
    "v5.2.3": datetime.date(2022, 11, 22),
    "v5.3.1": datetime.date(2023, 7, 26),
    "v5.3.2": datetime.date(2023, 9, 14),
    "v5.3.3": datetime.date(2024, 2, 20),
    "v5.3.4": datetime.date(2025, 6, 1),
    "v5.3.5": datetime.date(2025, 6, 8),
}


def _vdate(version: str) -> datetime.date:
    return _VERSION_DATES.get(version, datetime.date(1970, 1, 1))


# ---------------------------------------------------------------------------
# 1. Temporal Leakage Rate
# ---------------------------------------------------------------------------

def compute_temporal_leakage_rate(
    query_results: list,
) -> dict:
    """
    Measures what fraction of queries return at least one chunk from a version
    that is *newer* than the query's target version (temporal leakage).

    Parameters
    ----------
    query_results : list of dicts
        Each dict has:
          {
            "query": str,
            "target_version": str,        # version the query is pinned to
            "retrieved_docs": list[dict], # each doc has metadata.version
          }

    Returns
    -------
    dict:
        {
          "temporal_leakage_rate": float,   # 0.0–1.0 (lower is better)
          "leaky_queries": int,
          "total_queries": int,
          "details": list[dict]             # per-query breakdown
        }
    """
    leaky = 0
    total = len(query_results)
    details = []

    for entry in query_results:
        target_v = entry.get("target_version", "")
        target_date = _vdate(target_v)
        docs = entry.get("retrieved_docs", [])

        leaked_versions = []
        for doc in docs:
            doc_version = doc.get("metadata", {}).get("version", "")
            doc_date_str = doc.get("metadata", {}).get("version_date", "")
            # Parse version_date from ISO string stored in metadata
            try:
                doc_date = datetime.date.fromisoformat(doc_date_str)
            except (ValueError, TypeError):
                doc_date = _vdate(doc_version)

            if doc_date > target_date:
                leaked_versions.append(doc_version)

        has_leak = len(leaked_versions) > 0
        if has_leak:
            leaky += 1

        details.append({
            "query": entry.get("query", ""),
            "target_version": target_v,
            "leaked_versions": list(set(leaked_versions)),
            "leaky": has_leak,
        })

    rate = leaky / total if total > 0 else 0.0
    return {
        "temporal_leakage_rate": round(rate, 4),
        "leaky_queries": leaky,
        "total_queries": total,
        "details": details,
    }


# ---------------------------------------------------------------------------
# 2. Change Detection Accuracy
# ---------------------------------------------------------------------------

def compute_change_detection_accuracy(
    version_pairs: list,
    sqlite_path: str,
    chroma_collection,
    embed_model,
    sim_threshold: float = 0.95,
) -> dict:
    """
    For each consecutive version pair (v_old, v_new), compares the system's
    embedding-similarity-based change detection against ground-truth checksum diffs.

    Ground-truth: a chunk is "truly changed" if its checksum differs between versions
                  for the same (source_file, chunk_index).

    Detected:     a chunk is "detected changed" if the cosine similarity between
                  the old and new version embeddings falls below `sim_threshold`.

    Returns the macro-averaged F1 across all version pairs.

    Parameters
    ----------
    version_pairs : list of (str, str)
        e.g. [("v5.2.3", "v5.3.1"), ("v5.3.1", "v5.3.2"), ...]
    sqlite_path : str
        Path to the SQLite metadata database.
    chroma_collection : chromadb Collection
        The versioned_kb ChromaDB collection (for fetching embeddings).
    embed_model : SentenceTransformer
        For re-encoding chunks if embeddings are not stored (fallback).
    sim_threshold : float
        Cosine similarity below which a chunk is considered "detected changed".

    Returns
    -------
    dict:
        {
          "change_detection_f1": float,
          "change_detection_precision": float,
          "change_detection_recall": float,
          "pair_results": list[dict],   # per-pair breakdown
          "sim_threshold_used": float,
        }
    """
    pair_results = []
    all_f1 = []

    conn = sqlite3.connect(sqlite_path)

    for v_old, v_new in version_pairs:
        # Fetch chunk records for both versions
        old_chunks = {
            row[1]: (row[0], row[2])  # (source_file, chunk_index) -> (chunk_id, checksum)
            for row in conn.execute(
                "SELECT chunk_id, source_file, chunk_index, checksum "
                "FROM chunk_versions WHERE version = ?",
                (v_old,)
            )
        }
        new_chunks = {
            row[1]: (row[0], row[2])
            for row in conn.execute(
                "SELECT chunk_id, source_file, chunk_index, checksum "
                "FROM chunk_versions WHERE version = ?",
                (v_new,)
            )
        }

        # Find common logical keys (same source_file + chunk_index exist in both versions)
        # Key = (source_file, chunk_index) stored as "source_file::idx"
        old_rows = {
            (row[1], row[2]): (row[0], row[3])
            for row in conn.execute(
                "SELECT chunk_id, source_file, chunk_index, checksum "
                "FROM chunk_versions WHERE version = ?",
                (v_old,)
            )
        }
        new_rows = {
            (row[1], row[2]): (row[0], row[3])
            for row in conn.execute(
                "SELECT chunk_id, source_file, chunk_index, checksum "
                "FROM chunk_versions WHERE version = ?",
                (v_new,)
            )
        }

        common_keys = set(old_rows.keys()) & set(new_rows.keys())
        if not common_keys:
            pair_results.append({
                "pair": f"{v_old}→{v_new}",
                "common_chunks": 0,
                "f1": None,
                "precision": None,
                "recall": None,
                "note": "No overlapping logical chunks — fully new version",
            })
            continue

        tp = fp = fn = 0
        chunk_details = []

        for key in common_keys:
            old_id, old_checksum = old_rows[key]
            new_id, new_checksum = new_rows[key]

            # Ground truth: truly changed if checksums differ
            truly_changed = old_checksum != new_checksum

            # Detected changed: fetch embeddings from ChromaDB and compute cosine sim
            detected_changed = _check_embedding_change(
                chroma_collection, old_id, new_id, embed_model, sim_threshold
            )

            if truly_changed and detected_changed:
                tp += 1
            elif not truly_changed and detected_changed:
                fp += 1
            elif truly_changed and not detected_changed:
                fn += 1
            # tn: not changed, not detected → correct ignore (not counted in F1)

            chunk_details.append({
                "key": f"{key[0]}::{key[1]}",
                "truly_changed": truly_changed,
                "detected_changed": detected_changed,
            })

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        all_f1.append(f1)

        pair_results.append({
            "pair": f"{v_old}→{v_new}",
            "common_chunks": len(common_keys),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    conn.close()

    macro_f1 = float(np.mean(all_f1)) if all_f1 else 0.0
    macro_prec = float(np.mean([r["precision"] for r in pair_results if "precision" in r])) if pair_results else 0.0
    macro_rec = float(np.mean([r["recall"] for r in pair_results if "recall" in r])) if pair_results else 0.0

    return {
        "change_detection_f1": round(macro_f1, 4),
        "change_detection_precision": round(macro_prec, 4),
        "change_detection_recall": round(macro_rec, 4),
        "sim_threshold_used": sim_threshold,
        "pair_results": pair_results,
    }


def _check_embedding_change(
    collection,
    old_id: str,
    new_id: str,
    embed_model,
    threshold: float,
) -> bool:
    """
    Fetches embeddings for old_id and new_id from ChromaDB.
    Returns True if cosine similarity < threshold (i.e., content changed).
    Falls back to False if either ID is not found.
    """
    try:
        old_result = collection.get(ids=[old_id], include=["embeddings", "documents"])
        new_result = collection.get(ids=[new_id], include=["embeddings", "documents"])

        if not old_result["embeddings"] or not new_result["embeddings"]:
            # Fallback: re-encode documents if embeddings not stored
            if old_result["documents"] and new_result["documents"]:
                old_emb = embed_model.encode([old_result["documents"][0]])[0]
                new_emb = embed_model.encode([new_result["documents"][0]])[0]
            else:
                return False
        else:
            old_emb = np.array(old_result["embeddings"][0])
            new_emb = np.array(new_result["embeddings"][0])

        # Cosine similarity
        dot = np.dot(old_emb, new_emb)
        norm_old = np.linalg.norm(old_emb)
        norm_new = np.linalg.norm(new_emb)
        if norm_old == 0 or norm_new == 0:
            return False
        sim = float(dot / (norm_old * norm_new))
        return sim < threshold

    except Exception:
        return False


# ---------------------------------------------------------------------------
# 3. Update Efficiency
# ---------------------------------------------------------------------------

def compute_update_efficiency(ingestion_stats: dict) -> dict:
    """
    Computes update efficiency per version and overall.

    Update Efficiency = updated_chunks / total_chunks
    Ideal score: close to 0 for minor patch releases (few real changes),
                 higher for major feature releases.

    Parameters
    ----------
    ingestion_stats : dict
        From VersionedKBManager.ingestion_stats:
        { version: {"total": int, "updated": int, "skipped": int} }

    Returns
    -------
    dict:
        {
          "overall_efficiency": float,    # weighted average across versions
          "per_version": dict[str, float],
          "interpretation": str
        }
    """
    per_version = {}
    total_chunks = 0
    total_updated = 0

    for version, stats in ingestion_stats.items():
        t = stats.get("total", 0)
        u = stats.get("updated", 0)
        eff = u / t if t > 0 else 0.0
        per_version[version] = round(eff, 4)
        total_chunks += t
        total_updated += u

    overall = total_updated / total_chunks if total_chunks > 0 else 0.0

    if overall < 0.2:
        interp = "Highly efficient — system re-embeds <20% of chunks on update"
    elif overall < 0.5:
        interp = "Moderate efficiency — roughly half of chunks re-embedded"
    else:
        interp = "Low efficiency — majority of chunks re-embedded (expected for first ingest)"

    return {
        "overall_efficiency": round(overall, 4),
        "per_version": per_version,
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# 4. p50 / p95 / p99 Query Latency
# ---------------------------------------------------------------------------

def compute_latency_percentiles(latencies: list) -> dict:
    """
    Computes p50, p95, and p99 query latency percentiles.

    Parameters
    ----------
    latencies : list of float
        Wall-clock seconds per query, from VersionedRAGRetriever.latencies.

    Returns
    -------
    dict:
        {
          "p50_ms": float,
          "p95_ms": float,
          "p99_ms": float,
          "mean_ms": float,
          "min_ms": float,
          "max_ms": float,
          "count": int,
        }
    """
    if not latencies:
        return {
            "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0,
            "mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "count": 0
        }

    arr = np.array(latencies) * 1000.0  # convert to milliseconds

    return {
        "p50_ms":  round(float(np.percentile(arr, 50)), 3),
        "p95_ms":  round(float(np.percentile(arr, 95)), 3),
        "p99_ms":  round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(np.mean(arr)), 3),
        "min_ms":  round(float(np.min(arr)), 3),
        "max_ms":  round(float(np.max(arr)), 3),
        "count":   len(latencies),
    }
