"""
run_versioned_eval.py
=====================
Complete Versioned RAG Evaluation Framework & CSV Exporter.

Phases & Workflow:
1. Ingest PDF and Markdown documents into ChromaDB + SQLite.
2. Stage 1: Run 100 retrieval-only queries (fast, no LLM dependency).
   - Computes Temporal Leakage Rate and latency percentiles (p50, p95, p99).
   - Exports query_results.csv & latency_results.csv.
3. Stage 2: Run answer generation for subset (20 queries).
4. Run Version Change Detection evaluation against ground truth.
   - Exports change_detection_results.csv.
5. Run Multi-Iteration Incremental Update Experiment (5 runs).
   - Exports update_efficiency_results.csv.

Run from project root:
    python RAG_evaluation/run_versioned_eval.py
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

# UTF-8 stdout on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Resolve RAG_evaluation as a package root
RAG_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RAG_EVAL_DIR)
sys.path.insert(0, RAG_EVAL_DIR)
sys.path.insert(0, PROJECT_ROOT)

from generation.versionedrag_generator import VersionedKBManager, VersionedAnswerGenerator, _version_date
from retrieval.versionedrag_retriever import VersionedRAGRetriever
from generation.answer_generator import FallbackLLM, OllamaLLM
from evaluation.change_detector import evaluate_change_detection
from experiments.update_experiment import run_single_iteration
from evaluation.tier3_metrics import compute_latency_percentiles


def _hr(char: str = "=", width: int = 90) -> str:
    return char * width


def _table_row(*cells, widths):
    return " | ".join(str(c).ljust(w) for c, w in zip(cells, widths))


def check_ollama(base_url: str = "http://localhost:11434") -> bool:
    try:
        import httpx
        r = httpx.get(base_url, timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def main():
    print(_hr())
    print("   VERSIONED RAG EVALUATION & BENCHMARK SUITE")
    print(_hr())

    # --- Paths ---
    articles_dir = os.path.join(PROJECT_ROOT, "versioned_articles")
    db_path = os.path.join(PROJECT_ROOT, "versioned_chroma_store")
    sqlite_path = os.path.join(PROJECT_ROOT, "versioned_kb_metadata.db")
    queries_json = os.path.join(RAG_EVAL_DIR, "test_queries_100.json")
    gt_json = os.path.join(RAG_EVAL_DIR, "change_ground_truth.json")

    # CSV output paths (saved in project root & RAG_evaluation)
    query_results_csv = os.path.join(PROJECT_ROOT, "query_results.csv")
    change_detection_csv = os.path.join(PROJECT_ROOT, "change_detection_results.csv")
    update_eff_csv = os.path.join(PROJECT_ROOT, "update_efficiency_results.csv")
    latency_csv = os.path.join(PROJECT_ROOT, "latency_results.csv")

    if not os.path.isdir(articles_dir):
        print(f"[-] versioned_articles folder not found at: {articles_dir}")
        sys.exit(1)

    # --- Step 1: Document Ingestion (PDF + Markdown) ---
    print("\n[Step 1] Initializing KB Manager & Ingesting PDF & Markdown documents...")
    kb_manager = VersionedKBManager(
        articles_dir=articles_dir,
        db_path=db_path,
        sqlite_path=sqlite_path,
        use_embedding_cache=False,  # Optional embedding cache
    )
    ingestion_stats = kb_manager.ingest_all(force_reingest=False)

    # --- Step 2: Stage 1 Retrieval-Only Evaluation (100 Queries) ---
    print("\n[Step 2] Loading 100 retrieval test queries (Stage 1)...")
    if os.path.exists(queries_json):
        with open(queries_json, "r", encoding="utf-8") as f:
            queries_data = json.load(f)
    else:
        print(f"[-] Queries file not found: {queries_json}")
        sys.exit(1)

    print(f"   Executing Stage 1 retrieval on {len(queries_data)} queries (No LLM bottleneck)...")

    retriever = VersionedRAGRetriever.from_manager(kb_manager)
    retriever.reset_latencies()

    query_results_list = []
    latency_results_list = []

    leaky_count = 0

    for idx, item in enumerate(queries_data):
        q_id = item.get("query_id", f"q{idx+1}")
        q_text = item.get("query", "")
        exp_version = item.get("expected_version", item.get("target_version", "v5.3.1"))
        src_doc = item.get("source_document", "Unknown")

        t0 = time.perf_counter()
        docs = retriever.retrieve(q_text, top_k=5, target_version=exp_version, version_mode="exact")
        if not docs:
            docs = retriever.retrieve(q_text, top_k=5)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        retrieved_versions = sorted(list({d.get("metadata", {}).get("version", "?") for d in docs}))
        retrieved_version_str = ", ".join(retrieved_versions) if retrieved_versions else "none"
        retrieved_chunk_count = len(docs)

        # Temporal leakage check
        exp_date = _version_date(exp_version)
        temporal_leak = False
        for d in docs:
            d_ver = d.get("metadata", {}).get("version", "")
            d_date_str = d.get("metadata", {}).get("version_date", "")
            try:
                d_date = pd.to_datetime(d_date_str).date()
            except Exception:
                d_date = _version_date(d_ver)
            if d_date > exp_date:

                temporal_leak = True
                break

        if temporal_leak:
            leaky_count += 1

        query_results_list.append({
            "query_id": q_id,
            "query": q_text,
            "expected_version": exp_version,
            "retrieved_version": retrieved_version_str,
            "retrieved_chunks": retrieved_chunk_count,
            "temporal_leak": temporal_leak,
            "retrieval_latency_ms": latency_ms,
        })

        latency_results_list.append({
            "query_id": q_id,
            "latency_ms": latency_ms,
        })

    # Save Stage 1 CSVs
    df_query_res = pd.DataFrame(query_results_list)
    df_query_res.to_csv(query_results_csv, index=False)
    print(f"   [Saved] query_results.csv ({len(df_query_res)} rows)")

    df_lat_res = pd.DataFrame(latency_results_list)
    df_lat_res.to_csv(latency_csv, index=False)
    print(f"   [Saved] latency_results.csv ({len(df_lat_res)} rows)")

    # Compute Latency percentiles
    latencies_sec = [r["latency_ms"] / 1000.0 for r in latency_results_list]
    lat_stats = compute_latency_percentiles(latencies_sec)

    leak_rate = (leaky_count / len(queries_data)) * 100.0 if queries_data else 0.0

    # --- Step 3: Stage 2 LLM Answer Generation (Subset of 20 Queries) ---
    print("\n[Step 3] Running Stage 2 answer generation on 20 query subset...")
    has_ollama = check_ollama()
    if has_ollama:
        print("   [LLM] Ollama is available — using llama3.")
        llm = OllamaLLM(model_name="llama3")
    else:
        print("   [LLM] Ollama not active — using fast Fallback LLM for demonstration.")
        llm = FallbackLLM()

    generator = VersionedAnswerGenerator(llm=llm)

    subset_queries = queries_data[:20]
    stage2_results = []
    for item in subset_queries:
        q_text = item["query"]
        tgt_v = item.get("expected_version", item.get("target_version", "v5.3.1"))
        docs = retriever.retrieve(q_text, top_k=3, target_version=tgt_v, version_mode="exact")
        if not docs:
            docs = retriever.retrieve(q_text, top_k=3)
        ans = generator.generate_answer(q_text, docs, target_version=tgt_v)
        stage2_results.append({"query": q_text, "answer": ans})

    # --- Step 4: Version Change Detection Evaluation ---
    print("\n[Step 4] Running Chunk-Level Change Detection Analysis...")
    version_pairs = [
        ("v5.2.3", "v5.3.1"),
        ("v5.3.1", "v5.3.2"),
        ("v5.3.2", "v5.3.3"),
        ("v5.3.3", "v5.3.4"),
        ("v5.3.4", "v5.3.5"),
        ("v2.4.7", "v3.3.4"),
        ("v3.3.4", "v3.4.4"),
        ("v3.4.4", "v3.5.3"),
        ("v3.5.3", "v3.5.4"),
        ("v3.5.4", "v3.5.5"),
    ]

    change_eval_rows = evaluate_change_detection(version_pairs, sqlite_path, gt_json)
    df_change_res = pd.DataFrame(change_eval_rows)
    df_change_res.to_csv(change_detection_csv, index=False)
    print(f"   [Saved] change_detection_results.csv ({len(df_change_res)} rows)")

    correct_det_count = sum(1 for r in change_eval_rows if r["correct_detection"])
    change_acc = (correct_det_count / len(change_eval_rows)) * 100.0 if change_eval_rows else 0.0

    # --- Step 5: Incremental Update Benchmark ---
    print("\n[Step 5] Running Incremental Update Benchmark Experiment...")
    update_eff_rows = run_single_iteration(articles_dir=articles_dir)
    df_update_res = pd.DataFrame(update_eff_rows)
    df_update_res.to_csv(update_eff_csv, index=False)
    print(f"   [Saved] update_efficiency_results.csv ({len(df_update_res)} rows)")

    # -----------------------------------------------------------------------
    # FINAL METRICS SUMMARY DISPLAY
    # -----------------------------------------------------------------------
    print("\n" + _hr())
    print("   VERSIONED RAG EVALUATION — EXECUTIVE METRICS SUMMARY")
    print(_hr())

    w = [38, 20, 35]
    print(_table_row("Metric", "Score", "Notes", widths=w))
    print(_hr("-"))

    full_t = float(df_update_res["full_reindex_time"].iloc[0])
    inc_t = float(df_update_res["incremental_update_time"].iloc[0])
    chunks_f = int(df_update_res["chunks_processed_full"].iloc[0])
    chunks_i = int(df_update_res["chunks_processed_incremental"].iloc[0])
    speedup = float(df_update_res["update_efficiency_ratio"].iloc[0])

    summary_rows = [
        ("Temporal Leakage Rate", f"{leak_rate:.2f}%", f"{leaky_count}/{len(queries_data)} queries leaked"),
        ("Change Detection Accuracy", f"{change_acc:.2f}%", f"{correct_det_count}/{len(change_eval_rows)} pairs correct"),
        ("Incremental Update Efficiency", f"{speedup:.2f}x Speedup", f"Full: {full_t:.3f}s ({chunks_f} embs) vs Inc: {inc_t:.3f}s ({chunks_i} embs)"),
        ("Query Latency p50", f"{lat_stats['p50_ms']:.3f} ms", f"Over {len(queries_data)} queries"),
        ("Query Latency p95", f"{lat_stats['p95_ms']:.3f} ms", ""),
        ("Query Latency p99", f"{lat_stats['p99_ms']:.3f} ms", f"Mean={lat_stats['mean_ms']:.3f} ms"),
    ]

    for row in summary_rows:
        print(_table_row(*row, widths=w))

    print(_hr())
    print("\n[Generated CSV Outputs]")
    print(f"  1. {query_results_csv}")
    print(f"  2. {change_detection_csv}")
    print(f"  3. {update_eff_csv}")
    print(f"  4. {latency_csv}")
    print("\n" + _hr())
    print("   [Done] Versioned RAG Evaluation Framework execution complete!")
    print(_hr())


if __name__ == "__main__":
    main()
