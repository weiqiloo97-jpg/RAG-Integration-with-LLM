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
# ---------------------------------------------------------------------------
# Version ordering map  (used to compare versions chronologically)
# ---------------------------------------------------------------------------
_VERSION_DATES = {
    # Bootstrap releases
    "v5.2.3": datetime.date(2022, 11, 22),
    "v5.3.1": datetime.date(2023, 7, 26),
    "v5.3.2": datetime.date(2023, 9, 14),
    "v5.3.3": datetime.date(2024, 2, 20),
    "v5.3.4": datetime.date(2025, 6, 1),
    "v5.3.5": datetime.date(2025, 6, 8),
    # Apache Spark releases
    "v2.4.7": datetime.date(2020, 9, 12),
    "v3.3.4": datetime.date(2023, 12, 1),
    "v3.4.4": datetime.date(2024, 10, 20),
    "v3.5.3": datetime.date(2024, 9, 25),
    "v3.5.4": datetime.date(2024, 12, 18),
    "v3.5.5": datetime.date(2025, 3, 1),
}


def _version_date(version: str) -> datetime.date:
    """Returns a datetime.date for the given version string, or epoch if unknown."""
    return _VERSION_DATES.get(version, datetime.date(1970, 1, 1))


def _parse_version_from_filename(filename: str) -> str:
    """
    Extracts version string like 'v5.3.1' or 'v3.5.4' from filenames.
    """
    match = re.search(r"v?(\d+\.\d+\.\d+)", filename, re.IGNORECASE)
    if match:
        v = match.group(1)
        return f"v{v}" if not v.startswith("v") else v
    return "unknown"



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
    Manages versioned KB articles (Markdown & PDF):
    - Parses .md and .pdf files from a folder
    - Chunks, checksums, and attaches rich metadata:
      document_name, document_version, page_number, section_header, chunk_id, content_hash, timestamp
    - Tracks versions in SQLite (versioned_kb_metadata.db)
    - Embeds only NEW or CHANGED chunks into ChromaDB ("versioned_kb" collection)
    - Optional embedding cache to prevent redundant embedding calculations
    """

    def __init__(
        self,
        articles_dir: str,
        db_path: str = "./versioned_chroma_store",
        sqlite_path: str = "./versioned_kb_metadata.db",
        collection_name: str = "versioned_kb",
        embed_model_name: str = "all-MiniLM-L6-v2",
        use_embedding_cache: bool = False,
    ):
        self.articles_dir = Path(articles_dir)
        self.db_path = db_path
        self.sqlite_path = sqlite_path
        self.collection_name = collection_name
        self.use_embedding_cache = use_embedding_cache

        # Embedding model
        self.embed_model = SentenceTransformer(embed_model_name)

        # ChromaDB setup
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # SQLite setup
        self._init_sqlite()

        # Ingestion stats
        self.ingestion_stats: dict = {}

    def _init_sqlite(self):
        """Creates metadata and optional embedding cache tables in SQLite with schema migration."""
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_versions (
                chunk_id          TEXT PRIMARY KEY,
                document_name     TEXT NOT NULL DEFAULT '',
                version           TEXT NOT NULL,
                version_date      TEXT NOT NULL,
                page_number       INTEGER NOT NULL DEFAULT 1,
                section_header    TEXT NOT NULL DEFAULT '',
                source_file       TEXT NOT NULL,
                chunk_index       INTEGER NOT NULL,
                checksum          TEXT NOT NULL,
                changed_flag      INTEGER NOT NULL DEFAULT 1,
                ingested_at       TEXT NOT NULL
            )
        """)

        # Migration check: ensure missing columns are added if table existed prior to schema update
        cursor = conn.execute("PRAGMA table_info(chunk_versions)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        
        if "document_name" not in existing_cols:
            conn.execute("ALTER TABLE chunk_versions ADD COLUMN document_name TEXT NOT NULL DEFAULT ''")
        if "page_number" not in existing_cols:
            conn.execute("ALTER TABLE chunk_versions ADD COLUMN page_number INTEGER NOT NULL DEFAULT 1")
        if "section_header" not in existing_cols:
            conn.execute("ALTER TABLE chunk_versions ADD COLUMN section_header TEXT NOT NULL DEFAULT ''")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash  TEXT PRIMARY KEY,
                embedding_json TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()


    def _get_prior_checksum(self, source_file: str, chunk_index: int) -> Optional[str]:
        """Returns prior checksum for (source_file, chunk_index)."""
        conn = sqlite3.connect(self.sqlite_path)
        row = conn.execute(
            """
            SELECT checksum FROM chunk_versions
            WHERE source_file = ? AND chunk_index = ?
            ORDER BY ingested_at DESC
            LIMIT 1
            """,
            (source_file, chunk_index),
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def _get_cached_embedding(self, content_hash: str) -> Optional[list]:
        """Fetch cached embedding by content hash if optional cache is enabled."""
        if not self.use_embedding_cache:
            return None
        conn = sqlite3.connect(self.sqlite_path)
        row = conn.execute(
            "SELECT embedding_json FROM embedding_cache WHERE content_hash = ?",
            (content_hash,)
        ).fetchone()
        conn.close()
        if row:
            import json
            return json.loads(row[0])
        return None

    def _save_cached_embedding(self, content_hash: str, embedding: list):
        """Save embedding into cache if enabled."""
        if not self.use_embedding_cache:
            return
        import json
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (content_hash, embedding_json) VALUES (?, ?)",
            (content_hash, json.dumps(embedding)),
        )
        conn.commit()
        conn.close()

    def _record_chunk(self, chunk_data: dict, changed: bool, v_date: datetime.date):
        """Upserts a chunk record into SQLite."""
        conn = sqlite3.connect(self.sqlite_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO chunk_versions
              (chunk_id, document_name, version, version_date, page_number, section_header,
               source_file, chunk_index, checksum, changed_flag, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_data["chunk_id"],
                chunk_data.get("document_name", "Unknown Document"),
                chunk_data["document_version"],
                v_date.isoformat(),
                chunk_data.get("page_number", 1),
                chunk_data.get("section_header", ""),
                chunk_data["source_file"],
                chunk_data["chunk_index"],
                chunk_data["content_hash"],
                1 if changed else 0,
                chunk_data.get("timestamp", datetime.datetime.utcnow().isoformat()),
            ),
        )
        conn.commit()
        conn.close()

    def ingest_all(self, force_reingest: bool = False) -> dict:
        """
        Ingests all .md and .pdf files in articles_dir chronologically.
        """
        all_files = list(self.articles_dir.glob("*.md")) + list(self.articles_dir.glob("*.pdf"))
        
        all_files = sorted(
            all_files,
            key=lambda p: _version_date(_parse_version_from_filename(p.name)),
        )

        if not all_files:
            print(f"   [Warning] No .md or .pdf files found in {self.articles_dir}")
            return {}

        # Handle deleted files: remove chunks from ChromaDB & SQLite if source_file is no longer in articles_dir
        conn = sqlite3.connect(self.sqlite_path)
        db_sources = {
            row[0] for row in conn.execute("SELECT DISTINCT source_file FROM chunk_versions").fetchall()
        }
        present_sources = {p.name for p in all_files}
        deleted_sources = db_sources - present_sources

        for del_file in deleted_sources:
            del_rows = conn.execute(
                "SELECT chunk_id FROM chunk_versions WHERE source_file = ?", (del_file,)
            ).fetchall()
            del_ids = [r[0] for r in del_rows]
            if del_ids:
                try:
                    self.collection.delete(ids=del_ids)
                except Exception:
                    pass
                conn.execute("DELETE FROM chunk_versions WHERE source_file = ?", (del_file,))
                conn.commit()
                print(f"   [Deleted File] Removed {len(del_ids)} chunks for deleted file '{del_file}'")
        conn.close()

        print(f"\n[KB] Ingesting {len(all_files)} versioned documents (PDF & MD) into ChromaDB ...")

        for file_path in all_files:
            version = _parse_version_from_filename(file_path.name)
            v_date = _version_date(version)
            source_file = file_path.name

            # Process file according to extension
            if file_path.suffix.lower() == ".pdf":
                from ingestion.pdf_processor import process_pdf_file
                chunks_raw = process_pdf_file(str(file_path))
            else:
                doc_name = file_path.stem.replace("_", " ").replace("Release ", "").strip()
                raw_text = file_path.read_text(encoding="utf-8", errors="replace")
                md_chunks = _chunk_text(raw_text)
                now_iso = datetime.datetime.utcnow().isoformat()
                chunks_raw = []
                for idx, chunk in enumerate(md_chunks):
                    c_hash = _sha256(chunk)
                    chunks_raw.append({
                        "chunk_id": f"{source_file}__{version}__chunk_{idx}",
                        "text": chunk,
                        "document_name": doc_name,
                        "document_version": version,
                        "page_number": 1,
                        "section_header": f"Section {idx+1}",
                        "content_hash": c_hash,
                        "timestamp": now_iso,
                        "source_file": source_file,
                        "chunk_index": idx,
                    })

            stats = {
                "total": len(chunks_raw),
                "updated": 0,
                "skipped": 0,
                "embeddings_generated": 0,
                "db_operations": 0
            }

            ids_to_add = []
            docs_to_add = []
            embs_to_add = []
            metas_to_add = []

            for chunk_item in chunks_raw:
                c_hash = chunk_item["content_hash"]
                prior_checksum = self._get_prior_checksum(source_file, chunk_item["chunk_index"])
                changed = (prior_checksum is None) or (prior_checksum != c_hash)

                if changed or force_reingest:
                    cached_emb = self._get_cached_embedding(c_hash)
                    if cached_emb is not None:
                        embedding = cached_emb
                    else:
                        embedding = self.embed_model.encode([chunk_item["text"]])[0].tolist()
                        self._save_cached_embedding(c_hash, embedding)
                        stats["embeddings_generated"] += 1

                    ids_to_add.append(chunk_item["chunk_id"])
                    docs_to_add.append(chunk_item["text"])
                    embs_to_add.append(embedding)
                    metas_to_add.append({
                        "document_name": chunk_item["document_name"],
                        "version": chunk_item["document_version"],
                        "version_date": v_date.isoformat(),
                        "page_number": chunk_item["page_number"],
                        "section_header": chunk_item["section_header"],
                        "chunk_id": chunk_item["chunk_id"],
                        "content_hash": chunk_item["content_hash"],
                        "timestamp": chunk_item["timestamp"],
                        "source_file": source_file,
                        "chunk_index": chunk_item["chunk_index"],
                        "changed": 1 if changed else 0,
                    })
                    stats["updated"] += 1
                else:
                    stats["skipped"] += 1

                self._record_chunk(chunk_item, changed, v_date)

            if ids_to_add:
                self.collection.upsert(
                    ids=ids_to_add,
                    documents=docs_to_add,
                    embeddings=embs_to_add,
                    metadatas=metas_to_add,
                )
                stats["db_operations"] += len(ids_to_add)

            self.ingestion_stats[version] = stats
            print(
                f"   [{version}] ({file_path.name})  {stats['total']} chunks total | "
                f"{stats['updated']} re-embedded | {stats['skipped']} unchanged"
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
