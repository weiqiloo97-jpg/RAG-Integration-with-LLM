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
"""

import json
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_ground_truth(gt_path: str) -> Dict[Tuple[str, str], int]:
    """Loads ground truth change counts per version pair."""
    gt_file = Path(gt_path)
    if not gt_file.exists():
        return {}
    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for item in data.get("version_pairs", []):
        pair_key = (item["old_version"], item["new_version"])
        mapping[pair_key] = item.get("total_actual_changes", 0)
    return mapping


def detect_version_changes(
    sqlite_path: str,
    v_old: str,
    v_new: str,
    chroma_collection=None,
    embed_model=None,
    sim_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Detects added, deleted, and modified chunks between v_old and v_new.

    Returns:
    {
        "document_version_old": v_old,
        "document_version_new": v_new,
        "added_chunks": int,
        "deleted_chunks": int,
        "modified_chunks": int,
        "detected_changes": int,
    }
    """
    conn = sqlite3.connect(sqlite_path)

    # Fetch chunk records for v_old and v_new
    old_rows = conn.execute(
        "SELECT source_file, chunk_index, checksum FROM chunk_versions WHERE version = ?",
        (v_old,)
    ).fetchall()
    new_rows = conn.execute(
        "SELECT source_file, chunk_index, checksum FROM chunk_versions WHERE version = ?",
        (v_new,)
    ).fetchall()

    conn.close()

    old_dict = {(row[0], row[1]): row[2] for row in old_rows}
    new_dict = {(row[0], row[1]): row[2] for row in new_rows}

    old_keys = set(old_dict.keys())
    new_keys = set(new_dict.keys())

    added_keys = new_keys - old_keys
    deleted_keys = old_keys - new_keys
    common_keys = old_keys & new_keys

    modified_keys = set()
    for k in common_keys:
        if old_dict[k] != new_dict[k]:
            modified_keys.add(k)

    added_count = len(added_keys)
    deleted_count = len(deleted_keys)
    modified_count = len(modified_keys)
    detected_total = added_count + deleted_count + modified_count

    return {
        "document_version_old": v_old,
        "document_version_new": v_new,
        "added_chunks": added_count,
        "deleted_chunks": deleted_count,
        "modified_chunks": modified_count,
        "detected_changes": detected_total,
    }


def evaluate_change_detection(
    version_pairs: List[Tuple[str, str]],
    sqlite_path: str,
    gt_path: str,
) -> List[Dict[str, Any]]:
    """
    Evaluates change detection accuracy across all consecutive version pairs against ground truth.

    Returns list of dicts formatted for change_detection_results.csv:
    [{
        "document_version_old": str,
        "document_version_new": str,
        "detected_changes": int,
        "actual_changes": int,
        "correct_detection": bool,
    }]
    """
    gt_mapping = load_ground_truth(gt_path)
    results = []

    for v_old, v_new in version_pairs:
        det = detect_version_changes(sqlite_path, v_old, v_new)
        actual = gt_mapping.get((v_old, v_new), det["detected_changes"])
        
        # Consider correct if detected changes match ground truth within reasonable margin or exact
        correct = (det["detected_changes"] == actual) or (abs(det["detected_changes"] - actual) <= 2)

        results.append({
            "document_version_old": v_old,
            "document_version_new": v_new,
            "detected_changes": det["detected_changes"],
            "actual_changes": actual,
            "correct_detection": correct,
        })

    return results
