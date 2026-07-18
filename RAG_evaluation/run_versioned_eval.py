"""
run_versioned_eval.py
=====================
Standalone Tier 3 Versioned RAG evaluation runner.

Pipeline:
  1. Ingest versioned Bootstrap release notes (.md) into ChromaDB + SQLite
  2. Run sample queries (each pinned to a specific version)
  3. Compute all four Tier 3 metrics
  4. Print formatted table summaries

Run from project root:
    python RAG_evaluation/run_versioned_eval.py
"""

import os
import sys
import httpx
import numpy as np

# UTF-8 stdout on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Resolve RAG_evaluation as a package root
RAG_EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT  = os.path.dirname(RAG_EVAL_DIR)
sys.path.insert(0, RAG_EVAL_DIR)
sys.path.insert(0, PROJECT_ROOT)

from generation.versionedrag_generator import VersionedKBManager, VersionedAnswerGenerator
from retrieval.versionedrag_retriever   import VersionedRAGRetriever
from generation.answer_generator        import OllamaLLM
from evaluation.tier3_metrics import (
    compute_temporal_leakage_rate,
    compute_change_detection_accuracy,
    compute_update_efficiency,
    compute_latency_percentiles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_ollama(base_url: str = "http://localhost:11434") -> bool:
    try:
        r = httpx.get(base_url, timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _hr(char: str = "=", width: int = 90) -> str:
    return char * width


def _table_row(*cells, widths):
    return " | ".join(str(c).ljust(w) for c, w in zip(cells, widths))


# ---------------------------------------------------------------------------
# Sample queries  (version-pinned)
# ---------------------------------------------------------------------------

VERSIONED_QUERIES = [
    {
        "query": "What color mode improvements were introduced in Bootstrap v5.3.1?",
        "target_version": "v5.3.1",
    },
    {
        "query": "What CSS fixes were included in Bootstrap v5.2.3?",
        "target_version": "v5.2.3",
    },
    {
        "query": "What breaking change related to Sass was fixed in v5.3.3?",
        "target_version": "v5.3.3",
    },
    {
        "query": "What was the hotfix reason for Bootstrap v5.3.5?",
        "target_version": "v5.3.5",
    },
    {
        "query": "What JavaScript fixes were made to the selector engine in v5.3.3?",
        "target_version": "v5.3.3",
    },
    {
        "query": "List the key CSS changes made in Bootstrap v5.3.4.",
        "target_version": "v5.3.4",
    },
    {
        "query": "What new Sass variables were added in v5.3.1?",
        "target_version": "v5.3.1",
    },
    {
        "query": "What accessibility improvements were delivered in Bootstrap v5.3.2?",
        "target_version": "v5.3.2",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(_hr())
    print("   VERSIONED RAG EVALUATION — TIER 3 PERFORMANCE METRICS")
    print(_hr())

    # --- Paths ---
    articles_dir = os.path.join(PROJECT_ROOT, "versioned_articles")
    db_path      = os.path.join(PROJECT_ROOT, "versioned_chroma_store")
    sqlite_path  = os.path.join(PROJECT_ROOT, "versioned_kb_metadata.db")

    if not os.path.isdir(articles_dir):
        print(f"[-] versioned_articles folder not found at: {articles_dir}")
        sys.exit(1)

    # --- LLM ---
    print("\n[LLM] Connecting to Ollama ...")
    if not check_ollama():
        print("[-] ERROR: Ollama is not running on http://localhost:11434.")
        print("    Start it with:  ollama serve")
        print("    Then ensure the model is pulled:  ollama pull llama3")
        sys.exit(1)
    print("   [OK] Ollama is reachable — using llama3.")
    llm = OllamaLLM(model_name="llama3")

    # --- Step 1: Ingest versioned KB ---
    print("\n[Step 1] Initialising VersionedKBManager ...")
    kb_manager = VersionedKBManager(
        articles_dir=articles_dir,
        db_path=db_path,
        sqlite_path=sqlite_path,
    )
    ingestion_stats = kb_manager.ingest_all(force_reingest=False)

    # --- Step 2: Init retriever + generator ---
    print("\n[Step 2] Initialising VersionedRAGRetriever (shared ChromaDB client) ...")
    retriever = VersionedRAGRetriever.from_manager(kb_manager)

    generator = VersionedAnswerGenerator(llm=llm)

    # --- Step 3: Run queries ---
    print(f"\n[Step 3] Running {len(VERSIONED_QUERIES)} versioned queries ...")
    query_results  = []   # for temporal leakage
    sample_outputs = []   # for display table

    for i, item in enumerate(VERSIONED_QUERIES):
        q      = item["query"]
        tgt_v  = item["target_version"]

        docs = retriever.retrieve(q, top_k=5, target_version=tgt_v, version_mode="exact")

        # Fallback: if exact version filter returns nothing, retrieve without filter
        if not docs:
            docs = retriever.retrieve(q, top_k=5)

        answer = generator.generate_answer(q, docs, target_version=tgt_v)

        retrieved_versions = list({d.get("metadata", {}).get("version", "?") for d in docs})

        query_results.append({
            "query": q,
            "target_version": tgt_v,
            "retrieved_docs": docs,
        })
        sample_outputs.append({
            "query":              q,
            "target_version":     tgt_v,
            "retrieved_versions": retrieved_versions,
            "answer":             answer,
        })
        print(f"   [{i+1}/{len(VERSIONED_QUERIES)}] {tgt_v} | {q[:55]}...")

    # --- Step 4: Compute Tier 3 metrics ---
    print("\n[Step 4] Computing Tier 3 metrics ...")

    # 4a. Temporal Leakage Rate
    leakage = compute_temporal_leakage_rate(query_results)

    # 4b. Change Detection Accuracy
    version_pairs = kb_manager.get_consecutive_version_pairs()
    change_det = compute_change_detection_accuracy(
        version_pairs=version_pairs,
        sqlite_path=sqlite_path,
        chroma_collection=kb_manager.collection,
        embed_model=kb_manager.embed_model,
        sim_threshold=0.95,
    )

    # 4c. Update Efficiency
    update_eff = compute_update_efficiency(ingestion_stats)

    # 4d. Latency percentiles
    latency = compute_latency_percentiles(retriever.latencies)

    # -----------------------------------------------------------------------
    # OUTPUT — Metrics Summary Table
    # -----------------------------------------------------------------------
    print("\n" + _hr())
    print("   TIER 3 VERSIONED RAG — METRICS SUMMARY")
    print(_hr())

    # Row widths
    w = [38, 20, 35]
    print(_table_row("Metric", "Score", "Notes", widths=w))
    print(_hr("-"))

    def _fmt_pct(v): return f"{v * 100:.2f}%"
    def _fmt_f(v):   return f"{v:.4f}"

    rows = [
        (
            "Temporal Leakage Rate",
            _fmt_pct(leakage["temporal_leakage_rate"]),
            f"{leakage['leaky_queries']}/{leakage['total_queries']} queries leaked (lower=better)",
        ),
        (
            "Change Detection — F1",
            _fmt_f(change_det["change_detection_f1"]),
            f"Precision={change_det['change_detection_precision']:.4f}  "
            f"Recall={change_det['change_detection_recall']:.4f}",
        ),
        (
            "Change Detection — Precision",
            _fmt_f(change_det["change_detection_precision"]),
            f"Threshold = {change_det['sim_threshold_used']}",
        ),
        (
            "Change Detection — Recall",
            _fmt_f(change_det["change_detection_recall"]),
            "",
        ),
        (
            "Update Efficiency (overall)",
            _fmt_pct(update_eff["overall_efficiency"]),
            update_eff["interpretation"][:35],
        ),
        (
            "Query Latency  p50",
            f"{latency['p50_ms']:.3f} ms",
            f"Over {latency['count']} queries",
        ),
        (
            "Query Latency  p95",
            f"{latency['p95_ms']:.3f} ms",
            "",
        ),
        (
            "Query Latency  p99",
            f"{latency['p99_ms']:.3f} ms",
            f"Mean={latency['mean_ms']:.3f} ms  Max={latency['max_ms']:.3f} ms",
        ),
    ]

    for row in rows:
        print(_table_row(*row, widths=w))

    print(_hr())

    # -----------------------------------------------------------------------
    # Per-version Update Efficiency sub-table
    # -----------------------------------------------------------------------
    print("\n   Update Efficiency — Per Version")
    print(_hr("-", 60))
    w2 = [12, 10, 10, 10, 28]
    print(_table_row("Version", "Total", "Updated", "Skipped", "Efficiency", widths=w2))
    print(_hr("-", 60))
    for version, stats in ingestion_stats.items():
        eff = update_eff["per_version"].get(version, 0.0)
        print(_table_row(
            version,
            stats["total"],
            stats["updated"],
            stats["skipped"],
            f"{eff * 100:.1f}% re-embedded",
            widths=w2,
        ))
    print(_hr("-", 60))

    # -----------------------------------------------------------------------
    # Change Detection per-pair sub-table
    # -----------------------------------------------------------------------
    if change_det["pair_results"]:
        print("\n   Change Detection — Per Version Pair")
        print(_hr("-", 75))
        w3 = [16, 14, 6, 6, 6, 10, 10, 10]
        print(_table_row("Pair", "Common Chunks", "TP", "FP", "FN",
                         "Precision", "Recall", "F1", widths=w3))
        print(_hr("-", 75))
        for pr in change_det["pair_results"]:
            if "f1" in pr and pr["f1"] is not None:
                print(_table_row(
                    pr["pair"],
                    pr.get("common_chunks", "-"),
                    pr.get("true_positives", "-"),
                    pr.get("false_positives", "-"),
                    pr.get("false_negatives", "-"),
                    f"{pr['precision']:.4f}",
                    f"{pr['recall']:.4f}",
                    f"{pr['f1']:.4f}",
                    widths=w3,
                ))
            else:
                print(_table_row(
                    pr["pair"], pr.get("common_chunks", 0),
                    "-", "-", "-", "-", "-",
                    pr.get("note", ""),
                    widths=w3,
                ))
        print(_hr("-", 75))

    # -----------------------------------------------------------------------
    # OUTPUT — Sample Query Results Table
    # -----------------------------------------------------------------------
    print("\n" + _hr())
    print("   SAMPLE QUERY RESULTS — VERSIONED RAG ANSWERS")
    print(_hr())

    for idx, out in enumerate(sample_outputs):
        q_short   = out["query"]
        tgt       = out["target_version"]
        ret_v     = ", ".join(sorted(out["retrieved_versions"])) or "none"
        answer    = out["answer"]

        print(f"\n  [{idx+1}] Query       : {q_short}")
        print(f"       Target Ver  : {tgt}")
        print(f"       Ret. Ver(s) : {ret_v}")

        # Temporal leak flag
        lk_detail = leakage["details"][idx] if idx < len(leakage["details"]) else {}
        if lk_detail.get("leaky"):
            leaked = ", ".join(lk_detail.get("leaked_versions", []))
            print(f"       ⚠ LEAKAGE  : chunks from newer version(s): {leaked}")
        else:
            print(f"       Leakage    : none")

        # Wrap answer to 80 chars
        max_w = 80
        words = answer.split()
        lines = []
        line  = ""
        for w in words:
            if len(line) + len(w) + 1 <= max_w:
                line = (line + " " + w).strip()
            else:
                lines.append(line)
                line = w
        if line:
            lines.append(line)
        answer_display = ("\n" + " " * 19).join(lines)
        print(f"       Answer     : {answer_display}")
        print("  " + "-" * 88)

    # -----------------------------------------------------------------------
    # Temporal leakage detail
    # -----------------------------------------------------------------------
    print("\n   Temporal Leakage — Per Query Detail")
    print(_hr("-", 78))
    w4 = [10, 55, 10]
    print(_table_row("Target", "Query (truncated)", "Leaked?", widths=w4))
    print(_hr("-", 78))
    for d in leakage["details"]:
        flag = "YES ⚠" if d["leaky"] else "No"
        print(_table_row(d["target_version"], d["query"][:55], flag, widths=w4))
    print(_hr("-", 78))

    print("\n" + _hr())
    print("   [Finished] Versioned RAG Tier 3 evaluation complete!")
    print(_hr())


if __name__ == "__main__":
    main()
