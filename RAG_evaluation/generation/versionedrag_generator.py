"""
versionedrag_generator.py
=========================
Versioned Knowledge Base Manager and Answer Generator for Tier 3 evaluation.

Architecture:
  versioned_articles/*.md
    ↓ parse + chunk
  VersionedKBManager
    ├── SQLite  (versioned_kb_metadata.db)  — version, checksum, changed flag
    └── ChromaDB collection ("versioned_kb") — per-chunk embeddings

VersionedAnswerGenerator wraps the existing AnswerGenerator / LLM with
version-aware prompt prefixing.
"""

import os
import re
import hashlib
import sqlite3
import datetime
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from generation.answer_generator import AnswerGenerator, LLMWrapper, FallbackLLM


# ---------------------------------------------------------------------------
# Version ordering map  (used to compare versions chronologically)
# ---------------------------------------------------------------------------
_VERSION_DATES = {
    "v5.2.3": datetime.date(2022, 11, 22),
    "v5.3.1": datetime.date(2023, 7, 26),
    "v5.3.2": datetime.date(2023, 9, 14),
    "v5.3.3": datetime.date(2024, 2, 20),
    "v5.3.4": datetime.date(2025, 6, 1),    # "last week" from today, approx
    "v5.3.5": datetime.date(2025, 6, 8),    # "last week" from today, approx
}


def _version_date(version: str) -> datetime.date:
    """Returns a datetime.date for the given version string, or epoch if unknown."""
    return _VERSION_DATES.get(version, datetime.date(1970, 1, 1))


def _parse_version_from_filename(filename: str) -> str:
    """
    Extracts version string like 'v5.3.1' from filenames of the form
    'Release v5.3.1 · twbs_bootstrap.md'.
    """
    match = re.search(r"(v\d+\.\d+\.\d+)", filename)
    return match.group(1) if match else "unknown"


def _chunk_text(text: str, max_chars: int = 800) -> list:
    """
    Splits markdown text into chunks by double-newline paragraphs.
    Merges short paragraphs into the previous chunk, and splits
    oversized paragraphs at sentence boundaries.
    Returns a list of non-empty string chunks.
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_paragraphs = re.split(r"\n{2,}", text)

    chunks = []
    current = ""
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If the paragraph itself is too long, split by sentence
            if len(para) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                seg = ""
                for sent in sentences:
                    if len((seg + " " + sent).strip()) <= max_chars:
                        seg = (seg + " " + sent).strip()
                    else:
                        if seg:
                            chunks.append(seg)
                        seg = sent
                if seg:
                    current = seg
                else:
                    current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c.strip()) > 20]


def _sha256(text: str) -> str:
    """Returns the SHA-256 hex digest of the given string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# VersionedKBManager
# ---------------------------------------------------------------------------

class VersionedKBManager:
    """
    Manages versioned KB articles:
    - Parses .md files from a folder
    - Chunks and checksums each piece
    - Tracks versions in SQLite (versioned_kb_metadata.db)
    - Embeds only NEW or CHANGED chunks into ChromaDB ("versioned_kb" collection)
    - Exposes ingestion statistics for Update Efficiency metric
    """

    def __init__(
        self,
        articles_dir: str,
        db_path: str = "./versioned_chroma_store",
        sqlite_path: str = "./versioned_kb_metadata.db",
        collection_name: str = "versioned_kb",
        embed_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.articles_dir = Path(articles_dir)
        self.db_path = db_path
        self.sqlite_path = sqlite_path
        self.collection_name = collection_name

        # Embedding model
        self.embed_model = SentenceTransformer(embed_model_name)

        # ChromaDB setup — reuse same chroma_store with a distinct collection
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # SQLite setup
        self._init_sqlite()

        # Ingestion stats (populated during ingest_all)
        # Keys: version string → {"total": int, "updated": int}
        self.ingestion_stats: dict = {}

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _init_sqlite(self):
        """Creates the version-tracking table if it does not exist."""
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_versions (
                chunk_id      TEXT PRIMARY KEY,
                version       TEXT NOT NULL,
                version_date  TEXT NOT NULL,
                source_file   TEXT NOT NULL,
                chunk_index   INTEGER NOT NULL,
                checksum      TEXT NOT NULL,
                changed_flag  INTEGER NOT NULL DEFAULT 1,
                ingested_at   TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _get_prior_checksum(self, logical_key: str) -> Optional[str]:
        """
        Returns the checksum of the most recently ingested version of a
        logical chunk key (source_file + chunk_index), or None if never seen.
        """
        conn = sqlite3.connect(self.sqlite_path)
        row = conn.execute(
            """
            SELECT checksum FROM chunk_versions
            WHERE source_file = ? AND chunk_index = ?
            ORDER BY version_date DESC
            LIMIT 1
            """,
            logical_key,
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def _record_chunk(
        self,
        chunk_id: str,
        version: str,
        version_date: datetime.date,
        source_file: str,
        chunk_index: int,
        checksum: str,
        changed: bool,
    ):
        """Upserts a chunk record into SQLite."""
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO chunk_versions
              (chunk_id, version, version_date, source_file, chunk_index, checksum, changed_flag, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                version,
                version_date.isoformat(),
                source_file,
                chunk_index,
                checksum,
                1 if changed else 0,
                datetime.datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_all(self, force_reingest: bool = False) -> dict:
        """
        Reads all .md files from articles_dir sorted chronologically by version,
        chunks + checksums each file, and upserts new/changed chunks into ChromaDB.

        Returns self.ingestion_stats:
          { version: {"total": int, "updated": int, "skipped": int} }
        """
        md_files = sorted(
            self.articles_dir.glob("*.md"),
            key=lambda p: _version_date(_parse_version_from_filename(p.name)),
        )

        if not md_files:
            print(f"   [Warning] No .md files found in {self.articles_dir}")
            return {}

        print(f"\n[KB] Ingesting {len(md_files)} versioned articles into ChromaDB ...")

        for md_path in md_files:
            version = _parse_version_from_filename(md_path.name)
            v_date = _version_date(version)
            source_file = md_path.name

            raw_text = md_path.read_text(encoding="utf-8", errors="replace")
            chunks = _chunk_text(raw_text)

            stats = {"total": len(chunks), "updated": 0, "skipped": 0}

            ids_to_add = []
            docs_to_add = []
            embs_to_add = []
            metas_to_add = []

            for idx, chunk in enumerate(chunks):
                checksum = _sha256(chunk)
                logical_key = (source_file, idx)
                prior_checksum = self._get_prior_checksum(logical_key)
                chunk_id = f"{version}__chunk_{idx}"

                changed = (prior_checksum is None) or (prior_checksum != checksum)

                if changed or force_reingest:
                    # Embed this chunk
                    embedding = self.embed_model.encode([chunk])[0].tolist()
                    ids_to_add.append(chunk_id)
                    docs_to_add.append(chunk)
                    embs_to_add.append(embedding)
                    metas_to_add.append({
                        "version": version,
                        "version_date": v_date.isoformat(),
                        "source_file": source_file,
                        "chunk_index": idx,
                        "checksum": checksum,
                        "changed": 1 if changed else 0,
                    })
                    stats["updated"] += 1
                else:
                    stats["skipped"] += 1

                self._record_chunk(
                    chunk_id, version, v_date, source_file, idx, checksum, changed
                )

            # Batch upsert into ChromaDB
            if ids_to_add:
                self.collection.upsert(
                    ids=ids_to_add,
                    documents=docs_to_add,
                    embeddings=embs_to_add,
                    metadatas=metas_to_add,
                )

            self.ingestion_stats[version] = stats
            print(
                f"   [{version}]  {stats['total']} chunks total | "
                f"{stats['updated']} re-embedded | {stats['skipped']} unchanged (skipped)"
            )

        print(f"   [KB] Ingestion complete. Total collection size: {self.collection.count()} chunks.")
        return self.ingestion_stats

    # ------------------------------------------------------------------
    # Query helpers for metrics
    # ------------------------------------------------------------------

    def get_all_versions(self) -> list:
        """Returns sorted list of all ingested version strings."""
        return sorted(_VERSION_DATES.keys(), key=_version_date)

    def get_chunks_for_version(self, version: str) -> list:
        """Returns all ChromaDB chunk IDs for a given version."""
        conn = sqlite3.connect(self.sqlite_path)
        rows = conn.execute(
            "SELECT chunk_id, checksum FROM chunk_versions WHERE version = ?",
            (version,)
        ).fetchall()
        conn.close()
        return rows  # [(chunk_id, checksum), ...]

    def get_consecutive_version_pairs(self) -> list:
        """Returns a list of (version_a, version_b) tuples for consecutive versions."""
        versions = self.get_all_versions()
        return [(versions[i], versions[i + 1]) for i in range(len(versions) - 1)]


# ---------------------------------------------------------------------------
# VersionedAnswerGenerator
# ---------------------------------------------------------------------------

class VersionedAnswerGenerator:
    """
    Wraps AnswerGenerator with version-aware prompt injection.
    Prepends the target version context so the LLM knows which release
    it is answering about.
    """

    def __init__(self, llm: LLMWrapper):
        self.generator = AnswerGenerator(llm=llm)
        self.llm = llm

    def generate_answer(
        self,
        query: str,
        retrieved_docs: list,
        target_version: Optional[str] = None,
    ) -> str:
        """
        Generates an answer with optional version-awareness injected into the prompt.
        """
        if not retrieved_docs:
            return "I don't have enough context to answer this question."

        version_note = ""
        if target_version:
            version_note = (
                f"[Version Context: You are answering a question specifically about "
                f"Bootstrap {target_version}. Prioritise information from that release.]\n\n"
            )

        context_str = ""
        for i, doc in enumerate(retrieved_docs):
            doc_version = doc.get("metadata", {}).get("version", "unknown")
            context_str += (
                f"- Document [{i+1}] (Version {doc_version}): {doc['text']}\n"
            )

        prompt = (
            "System: You are a helpful technical assistant. Answer the user's question "
            "based strictly on the provided document context. If the context does not "
            "contain enough information, respond with 'I do not have enough information "
            "to answer this question.' Keep your answer concise and accurate.\n\n"
            f"{version_note}"
            f"Context:\n{context_str}\n"
            f"Question: {query}\n"
            "Answer:"
        )

        return self.llm.generate(prompt)
