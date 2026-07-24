"""
versionedrag_retriever.py
=========================
Version-aware retriever for the Versioned RAG Tier 3 evaluation.

Queries the "versioned_kb" ChromaDB collection with optional version filtering.
Records per-query wall-clock latency for p50/p95/p99 percentile computation.

IMPORTANT: ChromaDB PersistentClient holds a file lock on the store directory.
To avoid deadlocks, use `VersionedRAGRetriever.from_manager(kb_manager)` in the
evaluation runner so the retriever reuses the KB manager's already-open client
instead of opening a second one against the same directory.
"""

import time
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer


class VersionedRAGRetriever:
    """
    Dense semantic retriever scoped to the versioned ChromaDB collection.

    Parameters
    ----------
    db_path : str
        Path to the ChromaDB persistent store (same as chroma_store used by
        the existing VectorRetriever, but different collection name).
    collection_name : str
        ChromaDB collection for versioned KB chunks (default: "versioned_kb").
    model_name : str
        SentenceTransformer model name for query embedding.
    """

    def __init__(
        self,
        db_path: str = "./chroma_store",
        collection_name: str = "versioned_kb",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embed_model = SentenceTransformer(model_name)

        # Latency log — populated by each retrieve() call
        self.latencies: list = []   # wall-clock seconds per query

    @classmethod
    def from_manager(cls, kb_manager) -> "VersionedRAGRetriever":
        """
        Factory: creates a retriever that REUSES the collection and embed_model
        already held open by a VersionedKBManager instance.

        This avoids opening a second ChromaDB PersistentClient against the same
        directory (which causes a file-lock deadlock on Windows/Linux).

        Parameters
        ----------
        kb_manager : VersionedKBManager
            An already-initialised VersionedKBManager.

        Returns
        -------
        VersionedRAGRetriever
        """
        instance = cls.__new__(cls)
        instance.client = None          # no separate client
        instance.collection = kb_manager.collection
        instance.embed_model = kb_manager.embed_model
        instance.latencies = []
        return instance

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        target_version: Optional[str] = None,
        version_mode: str = "exact",
    ) -> list:
        """
        Retrieves top-k chunks from the versioned ChromaDB collection.

        Parameters
        ----------
        query : str
            Natural-language question.
        top_k : int
            Number of chunks to return.
        target_version : str or None
            If provided, filters results by version:
            - "exact"   → only chunks where metadata.version == target_version
            - "history" → chunks where metadata.version_date <= target_version's date
        version_mode : str
            "exact" or "history" (see target_version description above).

        Returns
        -------
        list of dicts:
            [{
                "id": str,
                "text": str,
                "metadata": dict,   # includes version, version_date, source_file
                "score": float      # cosine similarity (1 - distance)
            }]
        """
        t0 = time.perf_counter()

        query_embedding = self.embed_model.encode([query]).tolist()

        # Build ChromaDB where-filter
        where_filter = self._build_filter(target_version, version_mode)

        # Request more candidates if filtering is enabled so we get top_k after filter
        n_query = min(top_k * 3, max(top_k, self.collection.count())) if where_filter else top_k
        if n_query == 0:
            self.latencies.append(time.perf_counter() - t0)
            return []

        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_query, self.collection.count()),
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # Fallback: query without filter if filter fails (e.g., no matching docs)
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )

        elapsed = time.perf_counter() - t0
        self.latencies.append(elapsed)

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved = []
        for i in range(len(ids)):
            similarity = 1.0 - distances[i]
            retrieved.append({
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i] or {},
                "score": round(similarity, 4),
            })

        return retrieved[:top_k]

    # ------------------------------------------------------------------
    # Timed retrieval (returns docs + per-stage timing breakdown)
    # ------------------------------------------------------------------

    def retrieve_timed(
        self,
        query: str,
        top_k: int = 5,
        target_version=None,
        version_mode: str = "exact",
    ) -> tuple:
        """
        Identical to ``retrieve()`` but additionally measures and returns
        per-stage execution times.

        Returns
        -------
        docs : list
            Same format as ``retrieve()``.
        timing : dict
            {
                "embedding_ms": float,   # query embedding generation
                "retrieval_ms": float,   # ChromaDB vector query
                "reranking_ms": float,   # always 0.0 (no reranker in pipeline)
                "llm_ms":       float,   # always 0.0 (filled by caller for Stage 2)
                "total_ms":     float,
            }
        """
        import time as _time

        t_start = _time.perf_counter()

        # --- Stage: Embedding Generation ---
        t0 = _time.perf_counter()
        query_embedding = self.embed_model.encode([query]).tolist()
        embedding_ms = (_time.perf_counter() - t0) * 1000.0

        # --- Stage: Build filter ---
        where_filter = self._build_filter(target_version, version_mode)
        n_query = (
            min(top_k * 3, max(top_k, self.collection.count()))
            if where_filter
            else top_k
        )

        if n_query == 0:
            total_ms = (_time.perf_counter() - t_start) * 1000.0
            self.latencies.append(total_ms / 1000.0)
            timing = {
                "embedding_ms": round(embedding_ms, 3),
                "retrieval_ms": 0.0,
                "reranking_ms": 0.0,
                "llm_ms": 0.0,
                "total_ms": round(total_ms, 3),
            }
            return [], timing

        # --- Stage: Vector Retrieval ---
        t0 = _time.perf_counter()
        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_query, self.collection.count()),
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        retrieval_ms = (_time.perf_counter() - t0) * 1000.0

        total_ms = (_time.perf_counter() - t_start) * 1000.0
        self.latencies.append(total_ms / 1000.0)

        timing = {
            "embedding_ms": round(embedding_ms, 3),
            "retrieval_ms": round(retrieval_ms, 3),
            "reranking_ms": 0.0,
            "llm_ms": 0.0,
            "total_ms": round(total_ms, 3),
        }

        if not results or not results["ids"] or not results["ids"][0]:
            return [], timing

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved = []
        for i in range(len(ids)):
            similarity = 1.0 - distances[i]
            retrieved.append({
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i] or {},
                "score": round(similarity, 4),
            })

        return retrieved[:top_k], timing

    # ------------------------------------------------------------------
    # Filter builder
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        target_version: Optional[str],
        version_mode: str,
    ) -> Optional[dict]:
        """
        Builds a ChromaDB metadata where-filter dict.

        For "exact" mode  → {"version": {"$eq": target_version}}
        For "history" mode → no ChromaDB filter (post-filter in Python is simpler
                              given ChromaDB's limited datetime operators).
        Returns None if no filter needed.
        """
        if not target_version:
            return None
        if version_mode == "exact":
            return {"version": {"$eq": target_version}}
        # history mode: no server-side filter; caller can post-filter if needed
        return None

    # ------------------------------------------------------------------
    # Latency stats
    # ------------------------------------------------------------------

    def get_latency_stats(self) -> dict:
        """
        Returns latency percentile stats (in seconds) across all recorded queries.
        """
        import numpy as np
        if not self.latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}
        arr = np.array(self.latencies)
        return {
            "p50": round(float(np.percentile(arr, 50)), 6),
            "p95": round(float(np.percentile(arr, 95)), 6),
            "p99": round(float(np.percentile(arr, 99)), 6),
            "count": len(self.latencies),
        }

    def reset_latencies(self):
        """Clears the latency log."""
        self.latencies = []
