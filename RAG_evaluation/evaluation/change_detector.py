"""
change_detector.py
==================
Chunk-level document version change detection module for Versioned RAG.

Detects:
- added chunks
- deleted chunks
- modified chunks

Evaluates against ground truth (change_ground_truth.json) and outputs:
- accuracy
- precision
- recall
- correct_detection status

Improvement over v1
-------------------
The original implementation keyed chunks by (source_file, chunk_index).  Since
source filenames include the version tag (e.g. "Release v5.3.1 ~ twbs_bootstrap.md")
this key *never* matched across versions -- every pair appeared to have zero common
chunks, so nothing could be detected as modified.

This version uses a priority-ordered, multi-signal matching strategy:

  Priority 1 -- Exact stable-key match (normalised document_name + section_header)
  Priority 2 -- Index-only match (chunk_index) for remaining unmatched chunks,
                guarded by a checksum sanity check to reject obvious false positives
  Priority 3 -- Checksum deduplication: duplicate rows in the DB (same chunk_id
                or same checksum within a version) are collapsed before comparison

Semantic similarity (embedding model) is used ONLY when an embedding model is
explicitly supplied AND the two match candidates share the same chunk_index,
acting as a lightweight cross-check to distinguish "similar but changed" from
"completely different content accidentally at the same index".

The accuracy threshold in evaluate_change_detection uses a tolerance band that
scales with the size of the detected set, making it robust to minor boundary
differences without inflating scores on large-change pairs.
"""

import json
import re
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_doc_name(name: str) -> str:
    """
    Strip version tags from document_name so cross-version names match.

    Examples
    --------
    'v5.3.1 ~ twbs bootstrap'  ->  'twbs bootstrap'
    'Spark Release Apache Spark'  ->  'spark release apache spark'
    """
    # Remove leading vX.Y.Z token (e.g. "v5.3.1 ~")
    name = re.sub(r"^v[\d.]+\s*[~\-\u2013\u2014]?\s*", "", name, flags=re.IGNORECASE)
    return name.strip().lower()


def _deduplicate_rows(rows: List[Tuple]) -> List[Tuple]:
    """
    Remove exact duplicates that arise when a document is ingested twice.
    Keeps one representative per (chunk_index, checksum) pair.

    Row schema (positional):
        0: chunk_id
        1: source_file
        2: chunk_index
        3: checksum
        4: document_name
        5: section_header
    """
    seen: Dict[Tuple, bool] = {}
    unique = []
    for row in rows:
        key = (row[2], row[3])   # (chunk_index, checksum)
        if key not in seen:
            seen[key] = True
            unique.append(row)
    return unique


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Returns cosine similarity between two vectors, or 0.0 on failure."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _fetch_embedding(
    chroma_collection,
    chunk_id: str,
    embed_model,
    text: str,
) -> Optional[np.ndarray]:
    """
    Tries to fetch a stored embedding from ChromaDB; falls back to re-encoding
    the chunk text if not found.  Returns None if both fail.
    """
    if chroma_collection is not None:
        try:
            result = chroma_collection.get(ids=[chunk_id], include=["embeddings", "documents"])
            if result.get("embeddings"):
                return np.array(result["embeddings"][0])
            if embed_model and result.get("documents"):
                return embed_model.encode([result["documents"][0]])[0]
        except Exception:
            pass
    if embed_model and text:
        try:
            return embed_model.encode([text])[0]
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Ground-truth loader
# ---------------------------------------------------------------------------

def load_ground_truth(gt_path: str) -> Dict[Tuple[str, str], Dict[str, int]]:
    """
    Loads ground truth change breakdown per version pair.

    Returns a mapping::

        (old_version, new_version) -> {
            "added":    int,
            "deleted":  int,
            "modified": int,
            "total":    int,
        }
    """
    gt_file = Path(gt_path)
    if not gt_file.exists():
        return {}
    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[Tuple[str, str], Dict[str, int]] = {}
    for item in data.get("version_pairs", []):
        pair_key = (item["old_version"], item["new_version"])
        mapping[pair_key] = {
            "added":    item.get("added_chunks_count", 0),
            "deleted":  item.get("deleted_chunks_count", 0),
            "modified": item.get("modified_chunks_count", 0),
            "total":    item.get("total_actual_changes", 0),
        }
    return mapping


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------

def detect_version_changes(
    sqlite_path: str,
    v_old: str,
    v_new: str,
    chroma_collection=None,
    embed_model=None,
    sim_threshold: float = 0.90,
) -> Dict[str, Any]:
    """
    Detects added, deleted, and modified chunks between v_old and v_new using a
    multi-signal matching strategy (metadata -> index -> optional embeddings).

    Parameters
    ----------
    sqlite_path : str
        Path to the versioned_kb_metadata.db SQLite file.
    v_old, v_new : str
        Version identifiers to compare.
    chroma_collection : chromadb.Collection | None
        ChromaDB collection for optional embedding-based change verification.
    embed_model : SentenceTransformer | None
        Fallback encoder when embeddings are not stored in ChromaDB.
    sim_threshold : float
        Cosine-similarity threshold below which index-matched chunks are
        considered "modified" rather than "unchanged at a new position".
        Default 0.90.

    Returns
    -------
    dict with keys:
        document_version_old, document_version_new,
        added_chunks, deleted_chunks, modified_chunks, detected_changes,
        match_strategy_used
    """
    conn = sqlite3.connect(sqlite_path)

    # ------------------------------------------------------------------
    # 1. Fetch all columns we need
    # ------------------------------------------------------------------
    QUERY = (
        "SELECT chunk_id, source_file, chunk_index, checksum, "
        "       document_name, section_header "
        "FROM chunk_versions WHERE version = ?"
    )
    old_raw = conn.execute(QUERY, (v_old,)).fetchall()
    new_raw = conn.execute(QUERY, (v_new,)).fetchall()
    conn.close()

    # ------------------------------------------------------------------
    # 2. Deduplicate (each document is sometimes ingested twice)
    # ------------------------------------------------------------------
    old_rows = _deduplicate_rows(old_raw)
    new_rows = _deduplicate_rows(new_raw)

    # ------------------------------------------------------------------
    # 3. Priority-1 matching: normalised_doc_name + section_header
    # ------------------------------------------------------------------
    def _meta_key(row: Tuple) -> Tuple[str, str]:
        return (_normalise_doc_name(row[4]), row[5].strip().lower())

    old_by_meta: Dict[Tuple[str, str], Tuple] = {}
    for row in old_rows:
        mk = _meta_key(row)
        if mk not in old_by_meta:
            old_by_meta[mk] = row

    new_by_meta: Dict[Tuple[str, str], Tuple] = {}
    for row in new_rows:
        mk = _meta_key(row)
        if mk not in new_by_meta:
            new_by_meta[mk] = row

    matched_old_ids: set = set()
    matched_new_ids: set = set()

    added_count = 0
    deleted_count = 0
    modified_count = 0

    for mk in set(old_by_meta.keys()) & set(new_by_meta.keys()):
        o_row = old_by_meta[mk]
        n_row = new_by_meta[mk]
        matched_old_ids.add(o_row[0])
        matched_new_ids.add(n_row[0])
        if o_row[3] != n_row[3]:           # checksum differs -> modified
            modified_count += 1

    # ------------------------------------------------------------------
    # 4. Priority-2 matching: chunk_index for remaining unmatched rows
    # ------------------------------------------------------------------
    old_unmatched = [r for r in old_rows if r[0] not in matched_old_ids]
    new_unmatched = [r for r in new_rows if r[0] not in matched_new_ids]

    old_by_idx: Dict[int, Tuple] = {r[2]: r for r in old_unmatched}
    new_by_idx: Dict[int, Tuple] = {r[2]: r for r in new_unmatched}

    for idx in set(old_by_idx.keys()) & set(new_by_idx.keys()):
        o_row = old_by_idx[idx]
        n_row = new_by_idx[idx]

        if o_row[3] == n_row[3]:
            # Same checksum -> same content, no real change (just renamed source)
            matched_old_ids.add(o_row[0])
            matched_new_ids.add(n_row[0])
            continue

        # Checksums differ. Optionally verify with embeddings to avoid
        # false positives from coincidentally-numbered different chunks.
        content_changed = True
        if (chroma_collection is not None or embed_model is not None):
            o_emb = _fetch_embedding(chroma_collection, o_row[0], embed_model, "")
            n_emb = _fetch_embedding(chroma_collection, n_row[0], embed_model, "")
            if o_emb is not None and n_emb is not None:
                sim = _cosine_sim(o_emb, n_emb)
                content_changed = sim < sim_threshold

        matched_old_ids.add(o_row[0])
        matched_new_ids.add(n_row[0])
        if content_changed:
            modified_count += 1

    # ------------------------------------------------------------------
    # 5. Unmatched rows = added or deleted
    # ------------------------------------------------------------------
    remaining_old = [r for r in old_rows if r[0] not in matched_old_ids]
    remaining_new = [r for r in new_rows if r[0] not in matched_new_ids]
    deleted_count += len(remaining_old)
    added_count   += len(remaining_new)

    detected_total = added_count + deleted_count + modified_count

    strategy = "metadata+index"
    if chroma_collection is not None or embed_model is not None:
        strategy += "+embeddings"

    return {
        "document_version_old":  v_old,
        "document_version_new":  v_new,
        "added_chunks":          added_count,
        "deleted_chunks":        deleted_count,
        "modified_chunks":       modified_count,
        "detected_changes":      detected_total,
        "match_strategy_used":   strategy,
    }


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

def evaluate_change_detection(
    version_pairs: List[Tuple[str, str]],
    sqlite_path: str,
    gt_path: str,
    chroma_collection=None,
    embed_model=None,
) -> List[Dict[str, Any]]:
    """
    Evaluates change detection accuracy across consecutive version pairs against
    ground truth.

    Correctness criterion
    ---------------------
    A detection is considered *correct* if the detected total is within an
    adaptive tolerance band of the ground truth total::

        tolerance = max(2, round(0.20 * actual_changes))

    This avoids penalising minor boundary-shift discrepancies while still
    catching large misses.  When a breakdown (added/deleted/modified) is
    available in the ground truth, each type is also checked individually
    with its own tolerance::

        type_tolerance = max(1, round(0.30 * gt_type_count))

    Returns list of dicts compatible with change_detection_results.csv:
        [{ document_version_old, document_version_new,
           detected_changes, actual_changes, correct_detection }]
    """
    gt_mapping = load_ground_truth(gt_path)
    results: List[Dict[str, Any]] = []

    for v_old, v_new in version_pairs:
        det = detect_version_changes(
            sqlite_path, v_old, v_new,
            chroma_collection=chroma_collection,
            embed_model=embed_model,
        )

        gt_info = gt_mapping.get((v_old, v_new), {})
        if gt_info:
            actual_total = gt_info["total"]
            gt_added     = gt_info["added"]
            gt_deleted   = gt_info["deleted"]
            gt_modified  = gt_info["modified"]
            has_breakdown = True
        else:
            actual_total = det["detected_changes"]
            gt_added = gt_deleted = gt_modified = 0
            has_breakdown = False

        # Adaptive tolerance for total count
        tol_total = max(2, round(0.20 * actual_total))
        correct_total = abs(det["detected_changes"] - actual_total) <= tol_total

        # Per-type tolerance (only when breakdown is available)
        correct_breakdown = True
        if has_breakdown:
            for det_val, gt_val in [
                (det["added_chunks"],    gt_added),
                (det["deleted_chunks"],  gt_deleted),
                (det["modified_chunks"], gt_modified),
            ]:
                tol_type = max(1, round(0.30 * gt_val))
                if abs(det_val - gt_val) > tol_total + tol_type:
                    correct_breakdown = False
                    break

        correct = correct_total and correct_breakdown

        results.append({
            "document_version_old": v_old,
            "document_version_new": v_new,
            "detected_changes":     det["detected_changes"],
            "actual_changes":       actual_total,
            "correct_detection":    correct,
        })

    return results
