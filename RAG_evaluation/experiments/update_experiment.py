"""
update_experiment.py
====================
Incremental Update vs Full Re-indexing Benchmark Experiment.

Compares:
A) Full re-indexing (Reprocess entire document, regenerate all embeddings).
B) Incremental update (Detect changed chunks only, update affected vectors).

Runs multiple iterations (default 5 runs) and reports average metrics:
- execution_time (seconds)
- chunks_processed
- embeddings_generated
- database_operations
- Update Efficiency ratio = (full re-indexing time) / (incremental update time)
"""

import time
import shutil
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from generation.versionedrag_generator import VersionedKBManager


def run_single_iteration(articles_dir: str, num_runs: int = 5) -> List[Dict[str, Any]]:
    """
    Executes multi-run experiment comparing Full Re-indexing vs Incremental Update.

    Returns list of dicts formatted for update_efficiency_results.csv:
    [
        {
            "experiment_type": "Full Re-indexing",
            "execution_time": float,
            "chunks_processed": int,
            "embeddings_generated": int,
        },
        {
            "experiment_type": "Incremental Update",
            "execution_time": float,
            "chunks_processed": int,
            "embeddings_generated": int,
        }
    ]
    """
    full_times = []
    full_chunks = []
    full_embs = []

    inc_times = []
    inc_chunks = []
    inc_embs = []

    print(f"\n[Experiment] Running Incremental Update Benchmark over {num_runs} iterations...")

    for i in range(num_runs):
        temp_dir = tempfile.mkdtemp(prefix=f"rag_exp_run_{i}_")
        temp_chroma = Path(temp_dir) / "chroma"
        temp_sqlite = Path(temp_dir) / "meta.db"

        try:
            # 1. First run: Initial population
            manager = VersionedKBManager(
                articles_dir=articles_dir,
                db_path=str(temp_chroma),
                sqlite_path=str(temp_sqlite),
            )
            stats1 = manager.ingest_all(force_reingest=True)

            # 2. Benchmark Full Re-indexing (re-process all documents from scratch)
            t0 = time.perf_counter()
            manager_full = VersionedKBManager(
                articles_dir=articles_dir,
                db_path=str(temp_chroma),
                sqlite_path=str(temp_sqlite),
            )
            stats_full = manager_full.ingest_all(force_reingest=True)
            t_full = time.perf_counter() - t0

            total_chunks_f = sum(s["total"] for s in stats_full.values())
            total_embs_f = sum(s["embeddings_generated"] for s in stats_full.values())

            full_times.append(t_full)
            full_chunks.append(total_chunks_f)
            full_embs.append(total_embs_f)

            # 3. Benchmark Incremental Update (only updated/changed chunks)
            t1 = time.perf_counter()
            manager_inc = VersionedKBManager(
                articles_dir=articles_dir,
                db_path=str(temp_chroma),
                sqlite_path=str(temp_sqlite),
            )
            stats_inc = manager_inc.ingest_all(force_reingest=False)
            t_inc = time.perf_counter() - t1

            total_chunks_i = sum(s["total"] for s in stats_inc.values())
            total_embs_i = sum(s["embeddings_generated"] for s in stats_inc.values())

            inc_times.append(t_inc)
            inc_chunks.append(total_chunks_i)
            inc_embs.append(total_embs_i)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    avg_full_time = round(float(np.mean(full_times)), 4)
    avg_full_chunks = int(round(float(np.mean(full_chunks))))
    avg_full_embs = int(round(float(np.mean(full_embs))))

    avg_inc_time = round(float(np.mean(inc_times)), 4)
    avg_inc_chunks = int(round(float(np.mean(inc_chunks))))
    avg_inc_embs = int(round(float(np.mean(inc_embs))))

    efficiency_ratio = round(avg_full_time / avg_inc_time, 2) if avg_inc_time > 0 else 1.0

    print(f"   Avg Full Re-indexing Time: {avg_full_time:.4f}s ({avg_full_embs} embs generated)")
    print(f"   Avg Incremental Update Time: {avg_inc_time:.4f}s ({avg_inc_embs} embs generated)")
    print(f"   Calculated Update Efficiency (Speedup Ratio): {efficiency_ratio}x")

    return [
        {
            "experiment_type": "Full Re-indexing",
            "execution_time": avg_full_time,
            "chunks_processed": avg_full_chunks,
            "embeddings_generated": avg_full_embs,
        },
        {
            "experiment_type": "Incremental Update",
            "execution_time": avg_inc_time,
            "chunks_processed": avg_inc_chunks,
            "embeddings_generated": avg_inc_embs,
        }
    ]
