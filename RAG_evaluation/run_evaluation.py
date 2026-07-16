import os
import sys
import json
import time
import csv
import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

# Reconfigure stdout to support UTF-8 on Windows terminal to avoid UnicodeEncodeErrors
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Import modules from our package
from retrieval.vector_retriever import VectorRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from generation.answer_generator import AnswerGenerator, OllamaLLM, FallbackLLM
from evaluation.tier1_metrics import compute_tier1_metrics
from evaluation.tier2_metrics import compute_tier2_metrics, introduce_typos

def check_ollama_status(base_url="http://localhost:11434") -> bool:
    """Checks if the local Ollama server is running and accessible."""
    try:
        response = httpx.get(base_url, timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False

def main():
    print("=" * 70)
    print("   RAG EVALUATION PIPELINE - COMPARATIVE ANALYSIS")
    print("=" * 70)

    # 1. Paths & Configurations
    db_path = "./chroma_store"
    collection_name = "mflix"
    queries_file = "./RAG_evaluation/test_queries.json"
    
    if not os.path.exists(queries_file):
        print(f"[-] Error: Test queries file not found at {queries_file}")
        return

    print(f"[File] Loading test queries from: {queries_file}")
    with open(queries_file, "r", encoding="utf-8") as f:
        test_queries = json.load(f)
    print(f"   Loaded {len(test_queries)} evaluation queries.")

    # 2. Initialize Models & Database Connections
    print("\n[DB] Connecting to ChromaDB & loading SentenceTransformer model...")
    try:
        # Load embedding model once to share across retrievers and evaluations
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        rerank_model = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Initialize retrievers
        vec_retriever = VectorRetriever(db_path=db_path, collection_name=collection_name, model_name="all-MiniLM-L6-v2")
        bm25_retriever = BM25Retriever(db_path=db_path, collection_name=collection_name, source_filter="movies")
        hybrid_retriever = HybridRetriever(vector_retriever=vec_retriever, bm25_retriever=bm25_retriever, alpha=0.5)
        
        print("   [Success] Retrievers successfully initialized.")
    except Exception as e:
        print(f"[-] Error initializing retrieval models/database: {str(e)}")
        return

    # 3. Setup LLM Wrapper
    print("\n[LLM] Setting up LLM Wrapper...")
    ollama_model = "llama3"
    if check_ollama_status():
        print(f"   [Success] Local Ollama service found. Using '{ollama_model}' model.")
        llm = OllamaLLM(model_name=ollama_model)
    else:
        print("   [Warning] Local Ollama service not detected on http://localhost:11434.")
        print("   [Warning] Falling back to FallbackLLM (local heuristics) to complete evaluation.")
        llm = FallbackLLM()
        
    generator = AnswerGenerator(llm=llm)

    # Define RAG Models to evaluate (removed RAGFlow RAG)
    rag_pipelines = [
        ("Vector RAG", vec_retriever),
        ("BM25 RAG", bm25_retriever),
        ("Hybrid RAG", hybrid_retriever),
        ("Hybrid + Rerank", None)
    ]

    tier1_all_results = {}
    tier2_all_results = {}

    # 4. Evaluation Loop
    for name, retriever in rag_pipelines:
        print(f"\n[Run] Evaluating: {name} ...")
        
        t1_runs = []
        t2_runs = []
        
        for q_idx, item in enumerate(test_queries):
            query = item["question"]
            ground_truth = item["ground_truth"]
            gt_docs = item["relevant_documents"]
            
            # Retrieve documents
            t0 = time.perf_counter()
            if name == "Hybrid + Rerank":
                candidates = hybrid_retriever.retrieve(query, top_k=20)
                retrieved_docs = rerank_model.rerank(query, candidates, top_k=5)
            else:
                retrieved_docs = retriever.retrieve(query, top_k=5)
            query_time = time.perf_counter() - t0
            
            # Generate answer
            generated_answer = generator.generate_answer(query, retrieved_docs)
            
            # For robustness calculation, we perturb query and retrieve again
            perturbed_query = introduce_typos(query)
            if name == "Hybrid + Rerank":
                candidates_p = hybrid_retriever.retrieve(perturbed_query, top_k=20)
                retrieved_docs_p = rerank_model.rerank(perturbed_query, candidates_p, top_k=5)
            else:
                retrieved_docs_p = retriever.retrieve(perturbed_query, top_k=5)
                
            orig_ids = [d["id"] for d in retrieved_docs]
            pert_ids = [d["id"] for d in retrieved_docs_p]
            
            # Compute Tier 1
            t1 = compute_tier1_metrics(retrieved_docs, gt_docs, query, query_time, embed_model)
            t1_runs.append(t1)
            
            # Compute Tier 2
            t2 = compute_tier2_metrics(
                query=query,
                ground_truth=ground_truth,
                generated_answer=generated_answer,
                retrieved_docs=retrieved_docs,
                llm=llm,
                original_retrieved_ids=orig_ids,
                perturbed_retrieved_ids=pert_ids,
                embed_model=embed_model
            )
            t2_runs.append(t2)
            
            print(f"   [{q_idx+1}/{len(test_queries)}] Query: \"{query[:40]}...\" | HR: {t1['Hit Rate']} | Faithfulness: {t2['Faithfulness']}")

        # Average metrics
        tier1_averages = {
            "Hit Rate": np.mean([r["Hit Rate"] for r in t1_runs]),
            "Precision@5": np.mean([r["Precision@5"] for r in t1_runs]),
            "Recall@5": np.mean([r["Recall@5"] for r in t1_runs]),
            "MRR": np.mean([r["MRR"] for r in t1_runs]),
            "Average Similarity": np.mean([r["Average Similarity"] for r in t1_runs]),
            "Query Time": np.mean([r["Query Time"] for r in t1_runs])
        }
        
        tier2_averages = {
            "Faithfulness": np.mean([r["Faithfulness"] for r in t2_runs]),
            "Answer Relevancy": np.mean([r["Answer Relevancy"] for r in t2_runs]),
            "Context Precision": np.mean([r["Context Precision"] for r in t2_runs]),
            "Context Recall": np.mean([r["Context Recall"] for r in t2_runs]),
            "Noise Sensitivity": np.mean([r["Noise Sensitivity"] for r in t2_runs]),
            "Robustness": np.mean([r["Robustness"] for r in t2_runs])
        }
        
        tier1_all_results[name] = tier1_averages
        tier2_all_results[name] = tier2_averages

    # 5. Output CSVs
    t1_csv_path = "./RAG_evaluation/tier1_results.csv"
    t2_csv_path = "./RAG_evaluation/tier2_results.csv"

    print("\n[Save] Writing Tier 1 results to CSV...")
    with open(t1_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Hit Rate", "Precision@5", "Recall@5", "MRR", "Average Similarity", "Query Time"])
        for model_name, metrics in tier1_all_results.items():
            writer.writerow([
                model_name,
                f"{metrics['Hit Rate']:.4f}",
                f"{metrics['Precision@5']:.4f}",
                f"{metrics['Recall@5']:.4f}",
                f"{metrics['MRR']:.4f}",
                f"{metrics['Average Similarity']:.4f}",
                f"{metrics['Query Time']:.4f}"
            ])
    print(f"   Saved to: {t1_csv_path}")

    print("\n[Save] Writing Tier 2 results to CSV...")
    with open(t2_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall", "Noise Sensitivity", "Robustness"])
        for model_name, metrics in tier2_all_results.items():
            writer.writerow([
                model_name,
                f"{metrics['Faithfulness']:.4f}",
                f"{metrics['Answer Relevancy']:.4f}",
                f"{metrics['Context Precision']:.4f}",
                f"{metrics['Context Recall']:.4f}",
                f"{metrics['Noise Sensitivity']:.4f}",
                f"{metrics['Robustness']:.4f}"
            ])
    print(f"   Saved to: {t2_csv_path}")

    # 6. Summary Reports in Terminal
    print("\n" + "=" * 80)
    print("   TIER 1 RETRIEVAL METRICS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<18} | {'Hit Rate':<10} | {'Prec@5':<8} | {'Recall@5':<10} | {'MRR':<8} | {'Avg Sim':<8} | {'Time (s)':<8}")
    print("-" * 80)
    for model_name, metrics in tier1_all_results.items():
        print(f"{model_name:<18} | {metrics['Hit Rate']:<10.2%} | {metrics['Precision@5']:<8.4f} | {metrics['Recall@5']:<10.4f} | {metrics['MRR']:<8.4f} | {metrics['Average Similarity']:<8.4f} | {metrics['Query Time']:<8.4f}")
    
    print("\n" + "=" * 80)
    print("   TIER 2 GENERATION & CONTEXT METRICS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<18} | {'Faithful':<8} | {'Relevancy':<9} | {'Ctx Prec':<8} | {'Ctx Rec':<8} | {'Noise Sens':<10} | {'Robustness':<10}")
    print("-" * 80)
    for model_name, metrics in tier2_all_results.items():
        print(f"{model_name:<18} | {metrics['Faithfulness']:<8.4f} | {metrics['Answer Relevancy']:<9.4f} | {metrics['Context Precision']:<8.4f} | {metrics['Context Recall']:<8.4f} | {metrics['Noise Sensitivity']:<10.4f} | {metrics['Robustness']:<10.4f}")
    print("=" * 80)
    print("\n[Finished] Evaluation complete!")

if __name__ == "__main__":
    main()
