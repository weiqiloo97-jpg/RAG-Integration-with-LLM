# RAG Evaluation Framework

A comprehensive Retrieval-Augmented Generation (RAG) evaluation framework that compares different RAG architectures, including traditional RAG and Version-Aware RAG.

The system supports document ingestion, vector retrieval, version-aware retrieval, and multi-tier performance evaluation to analyse retrieval accuracy, temporal consistency, efficiency, and latency.

---

# 1. Project Overview

This project investigates whether version-aware retrieval improves the performance of traditional RAG systems when handling evolving documents.

The framework compares multiple RAG approaches:

## Baseline RAG
A standard vector-based Retrieval-Augmented Generation pipeline.

Workflow:

```
Documents
    |
Chunking
    |
Embedding Generation
    |
Vector Database
    |
Similarity Retrieval
    |
LLM Answer Generation
```

---

## GraphRAG

A retrieval approach enhanced with knowledge relationships between document entities and metadata.

The system constructs additional structural information to improve retrieval over related concepts.

---

## VersionRAG

A version-aware RAG system designed for evolving document collections.

Unlike traditional RAG, VersionRAG preserves document versions and tracks changes between releases.

Key features:

- Version-aware document storage
- Chunk-level change detection
- Incremental knowledge base updates
- Temporal leakage prevention
- Efficient re-indexing of modified content only

Example:

```
Apache Spark v3.4.4
        |
        |
        v
Apache Spark v3.5.4
```

Instead of rebuilding the entire database, VersionRAG identifies changed chunks and updates only affected information.

---

# 2. Evaluation Framework

The project evaluates RAG performance using three evaluation tiers.

---

# Tier 1: Retrieval Performance Evaluation

Measures whether the retriever returns relevant information.

Metrics include:

### Retrieval Accuracy
Evaluates whether retrieved documents match the expected answer source.

### Hit Rate
Measures whether the correct document appears within retrieved results.

### Precision / Recall
Analyses retrieval quality.

### Mean Reciprocal Rank (MRR)
Measures how highly the correct result is ranked.

---

# Tier 2: Generation Quality Evaluation

Evaluates the final generated answers from the LLM.

Metrics include:

### Faithfulness
Whether the generated answer is supported by retrieved context.

### Answer Relevance
Whether the response correctly answers the query.

### Context Utilisation
Measures whether retrieved information contributes to the final answer.

---

# Tier 3: Version-Aware Evaluation

Additional metrics designed specifically for VersionRAG.

## Temporal Leakage Rate

Measures whether the system retrieves information from incorrect document versions.

Example:

Query:

```
What changed in Bootstrap v5.3.4?
```

Incorrect retrieval:

```
Bootstrap v5.3.5 information
```

This is considered temporal leakage.

---

## Change Detection Accuracy

Evaluates whether VersionRAG correctly identifies:

- Added chunks
- Deleted chunks
- Modified chunks

between document versions.

---

## Incremental Update Efficiency

Compares:

Traditional approach:

```
Delete database
        |
Re-index everything
```

against:

VersionRAG:

```
Detect changed chunks
        |
Update only affected content
```

Metrics:

- Execution time
- Number of processed chunks
- Number of regenerated embeddings
- Speed improvement ratio

---

## Query Latency

Measures retrieval speed using:

- p50 latency
- p95 latency
- p99 latency

These statistics evaluate normal and worst-case retrieval performance.

---

# 3. Build Instruction

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Install required packages:
```powershell
pip install -r requirements.txt
```

Run from project root:

```powershell
python RAG_evaluation/run_versioned_eval.py
```

The program will:

1. Load and preprocess documents
2. Generate embeddings
3. Build vector indexes
4. Execute retrieval experiments
5. Run RAG evaluation metrics
6. Generate performance reports
- Retrieval performance

while comparing traditional RAG approaches against VersionRAG.
