# Versioned RAG System & Evaluation Framework

A Version-Aware Retrieval-Augmented Generation (RAG) framework with PDF & Markdown ingestion, chunk-level change detection, multi-run incremental update benchmarking, and 100-query statistical evaluation.

---

## 1. Installation

Install all required Python dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies
- `pypdf`: PDF text extraction and page parsing.
- `chromadb`: Vector database for embedding storage and similarity retrieval.
- `sentence-transformers`: Local embedding model (`all-MiniLM-L6-v2`).
- `pandas` & `numpy`: Data processing and latency percentile calculations.
- `httpx`: Optional connection to Ollama LLM.

---

## 2. Ingestion Pipeline

The pipeline supports both **PDF** and **Markdown** documents located in `versioned_articles/`:

- **PDF Ingestion**: `ingestion/pdf_processor.py` extracts page text, identifies section headers, and chunks document text.
- **Rich Metadata Preservation**:
  - `document_name`
  - `document_version`
  - `page_number`
  - `section_header`
  - `chunk_id`
  - `content_hash`
  - `timestamp`
- **Optional Embedding Cache**: Toggleable via `use_embedding_cache=True` in `VersionedKBManager` to cache chunk embeddings by SHA-256 hash.

---

## 3. How to Run Indexing & Evaluation

Run the master evaluation runner:

```bash
python RAG_evaluation/run_versioned_eval.py
```

### What this command executes:
1. **Document Ingestion**: Parses `.pdf` (Spark release notes) and `.md` (Bootstrap releases) from `versioned_articles/` into ChromaDB (`versioned_chroma_store`) and SQLite (`versioned_kb_metadata.db`).
2. **Stage 1 Evaluation (100 Queries)**: Executes 100 version-pinned queries retrieval-only without LLM overhead. Calculates Temporal Leakage Rate and latency percentiles (p50, p95, p99).
3. **Stage 2 Evaluation (20 Queries)**: Generates natural language answers for a 20-query subset (using local Ollama `llama3` or fallback LLM).
4. **Version Change Detection Analysis**: Evaluates detected added/deleted/modified chunks against `change_ground_truth.json`.
5. **Incremental Update Experiment**: Runs 3-5 iterations comparing **Full Re-indexing** vs **Incremental Update** and reports averaged execution time, processed chunks, generated embeddings, and Update Efficiency speedup ratio.

---

## 4. Generated CSV Reports

Results are automatically saved to the root directory as CSV files suitable for statistical analysis:

1. **`query_results.csv`**:
   - `query_id`, `query`, `expected_version`, `retrieved_version`, `retrieved_chunks`, `temporal_leak`, `retrieval_latency_ms`
2. **`change_detection_results.csv`**:
   - `document_version_old`, `document_version_new`, `detected_changes`, `actual_changes`, `correct_detection`
3. **`update_efficiency_results.csv`**:
   - `experiment_type`, `execution_time`, `chunks_processed`, `embeddings_generated`
4. **`latency_results.csv`**:
   - `query_id`, `latency_ms`

---

## 5. System Limitations & Future Work

- **Complex PDF Layouts**: PDF text extraction uses `pypdf`. Multicolumn tables or scanned image PDFs may require OCR (e.g. `pytesseract` or `pdfplumber`).
- **Semantic Change Thresholding**: Embedding similarity change detection threshold is currently set to `0.95`. Dynamic thresholding based on chunk length can further refine precision.
