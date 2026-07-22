"""
update_experiment.py
====================
Incremental Update vs Full Re-indexing Benchmark Experiment.

Benchmark sequence:
1. Create initial KB snapshot once in an isolated temporary environment.
2. Measure full re-indexing:
   - Clear database
   - Re-ingest all documents
   - Record time, embeddings generated, database operations.
3. Restore original KB snapshot.
4. Simulate document version update:
   - Modify/add/delete selected versioned documents.
   - Run chunk-level change detection.
   - Update only affected chunks.
5. Measure incremental update:
   - Record update time
   - Number of chunks re-embedded
   - Database operations.

Returns metric row matching CSV specification:
full_reindex_time, incremental_update_time, chunks_processed_full, chunks_processed_incremental, update_efficiency_ratio
"""

import os
import gc
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from generation.versionedrag_generator import VersionedKBManager


def run_update_benchmark(articles_dir: str) -> Dict[str, Any]:
    """
    Executes isolated benchmark comparing Full Re-indexing vs Incremental Update.
    Does NOT modify or corrupt the primary workspace knowledge base.
    """
    temp_dir = tempfile.mkdtemp(prefix="rag_update_bench_")
    try:
        temp_dir_path = Path(temp_dir)
        temp_articles_dir = temp_dir_path / "articles"
        temp_db_path = temp_dir_path / "chroma_db"
        temp_sqlite_path = temp_dir_path / "meta.db"

        snapshot_db_path = temp_dir_path / "snapshot_chroma"
        snapshot_sqlite_path = temp_dir_path / "snapshot_meta.db"

        # Copy original articles to isolated work directory
        shutil.copytree(articles_dir, temp_articles_dir)

        # -------------------------------------------------------------
        # 1. Create initial KB snapshot once
        # -------------------------------------------------------------
        print("\n[Benchmark Step 1] Creating initial KB snapshot once...")
        manager_init = VersionedKBManager(
            articles_dir=str(temp_articles_dir),
            db_path=str(temp_db_path),
            sqlite_path=str(temp_sqlite_path),
            use_embedding_cache=False,
        )
        manager_init.ingest_all(force_reingest=True)
        del manager_init
        gc.collect()

        # Save snapshot copy
        shutil.copytree(temp_db_path, snapshot_db_path, dirs_exist_ok=True)
        shutil.copy2(temp_sqlite_path, snapshot_sqlite_path)
        print("   [Step 1 Complete] Snapshot created successfully.")

        # -------------------------------------------------------------
        # 4. Simulate document version update (Prepare changes)
        # -------------------------------------------------------------
        print("   [Simulate Update] Modifying, adding, and deleting selected versioned documents...")
        md_files = sorted(list(temp_articles_dir.glob("*.md")))

        # A) Modify selected document
        if md_files:
            mod_file = md_files[0]
            orig_text = mod_file.read_text(encoding="utf-8", errors="replace")
            mod_file.write_text(
                orig_text + "\n\n## Patch Update Section\n\nAdded updated paragraph for incremental update test.\n",
                encoding="utf-8"
            )

        # B) Add selected new document
        new_file = temp_articles_dir / "Release_v5.3.6_patch.md"
        new_file.write_text(
            "# Release v5.3.6 Patch\n\nNew version patch document containing updated features and bug fixes.\n",
            encoding="utf-8"
        )

        # C) Delete selected document
        if len(md_files) > 1:
            del_file = md_files[-1]
            os.remove(del_file)

        # -------------------------------------------------------------
        # 2. Measure full re-indexing
        # -------------------------------------------------------------
        print("[Benchmark Step 2] Measuring Full Re-indexing...")
        shutil.rmtree(temp_db_path, ignore_errors=True)
        if temp_sqlite_path.exists():
            os.remove(temp_sqlite_path)

        t0 = time.perf_counter()
        manager_full = VersionedKBManager(
            articles_dir=str(temp_articles_dir),
            db_path=str(temp_db_path),
            sqlite_path=str(temp_sqlite_path),
            use_embedding_cache=False,
        )
        stats_full = manager_full.ingest_all(force_reingest=True)
        t_full = time.perf_counter() - t0

        chunks_full = sum(s["embeddings_generated"] for s in stats_full.values())
        db_ops_full = sum(s["db_operations"] for s in stats_full.values())
        del manager_full
        gc.collect()

        print(f"   Full Re-indexing: time={t_full:.4f}s, chunks_embedded={chunks_full}, db_ops={db_ops_full}")

        # -------------------------------------------------------------
        # 3. Restore original KB snapshot
        # -------------------------------------------------------------
        print("[Benchmark Step 3] Restoring original KB snapshot...")
        shutil.rmtree(temp_db_path, ignore_errors=True)
        if temp_sqlite_path.exists():
            os.remove(temp_sqlite_path)

        shutil.copytree(snapshot_db_path, temp_db_path, dirs_exist_ok=True)
        shutil.copy2(snapshot_sqlite_path, temp_sqlite_path)
        print("   [Step 3 Complete] Snapshot restored.")

        # -------------------------------------------------------------
        # 5. Measure incremental update
        # -------------------------------------------------------------
        print("[Benchmark Step 5] Measuring Incremental Update (chunk-level change detection)...")
        t1 = time.perf_counter()
        manager_inc = VersionedKBManager(
            articles_dir=str(temp_articles_dir),
            db_path=str(temp_db_path),
            sqlite_path=str(temp_sqlite_path),
            use_embedding_cache=False,
        )
        stats_inc = manager_inc.ingest_all(force_reingest=False)
        t_inc = time.perf_counter() - t1

        chunks_inc = sum(s["embeddings_generated"] for s in stats_inc.values())
        db_ops_inc = sum(s["db_operations"] for s in stats_inc.values())
        del manager_inc
        gc.collect()

        print(f"   Incremental Update: time={t_inc:.4f}s, chunks_reembedded={chunks_inc}, db_ops={db_ops_inc}")

        efficiency_ratio = round(t_full / t_inc, 2) if t_inc > 0 else 1.0

        res = {
            "full_reindex_time": round(t_full, 4),
            "incremental_update_time": round(t_inc, 4),
            "chunks_processed_full": chunks_full,
            "chunks_processed_incremental": chunks_inc,
            "update_efficiency_ratio": efficiency_ratio,
        }

        print(f"\n   Calculated Update Efficiency (Speedup Ratio): {efficiency_ratio}x")
        return res

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_single_iteration(articles_dir: str, num_runs: int = 1) -> List[Dict[str, Any]]:
    """
    Backward-compatible wrapper function for run_update_benchmark.
    Returns list of dicts formatted for DataFrame export to CSV.
    """
    res = run_update_benchmark(articles_dir=articles_dir)
    return [res]
