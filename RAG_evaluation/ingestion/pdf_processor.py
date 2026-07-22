"""
pdf_processor.py
================
PDF Ingestion & Text Extraction Pipeline for Versioned RAG.

Transforms raw PDF files into structured markdown-like chunks with rich metadata:
- document_name
- document_version
- page_number
- section_header
- chunk_id
- content_hash
- timestamp
"""

import os
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

try:
    import pypdf
except ImportError:
    pypdf = None


def extract_version_from_filename(filename: str) -> str:
    """Extract version string like 'v3.5.4' or '3.5.4' from filename."""
    # Match patterns like 3.5.4, 2.4.7, v5.3.1
    match = re.search(r"v?(\d+\.\d+(?:\.\d+)?)", filename, re.IGNORECASE)
    if match:
        version = match.group(1)
        return f"v{version}" if not version.startswith("v") else version
    return "v1.0.0"


def extract_doc_name_from_filename(filename: str) -> str:
    """Extract clean document name from filename."""
    base = Path(filename).stem
    # Remove version numbers and trailing web title parts
    cleaned = re.sub(r"v?\d+\.\d+(?:\.\d+)?", "", base, flags=re.IGNORECASE)
    cleaned = cleaned.replace("_", " ").replace("-", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned if cleaned else base


def _guess_section_header(lines: List[str], fallback_page: int) -> str:
    """Identify potential section header from top lines of page chunk."""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Heading markers or short uppercase/title-case lines
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if len(stripped) < 60 and (stripped.isupper() or stripped.istitle() or stripped.endswith(":")):
            return stripped.rstrip(":")
    return f"Section Page {fallback_page}"


def process_pdf_file(pdf_path: str, max_chunk_chars: int = 800) -> List[Dict[str, Any]]:
    """
    Parses a PDF document into structured chunks with version metadata.

    Returns list of chunk dicts:
    [{
        "chunk_id": str,
        "text": str,
        "document_name": str,
        "document_version": str,
        "page_number": int,
        "section_header": str,
        "content_hash": str,
        "timestamp": str,
    }]
    """
    pdf_path = Path(pdf_path)
    filename = pdf_path.name
    doc_name = extract_doc_name_from_filename(filename)
    doc_version = extract_version_from_filename(filename)
    timestamp = datetime.utcnow().isoformat()

    chunks_data = []

    if pypdf is None:
        raise RuntimeError("pypdf is required for PDF processing. Run: pip install pypdf")

    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception as e:
        print(f"[-] Error reading PDF {pdf_path}: {e}")
        return []

    global_chunk_idx = 0

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        page_text = page.extract_text() or ""
        page_text = page_text.replace("\r\n", "\n").replace("\r", "\n").strip()

        if not page_text:
            continue

        lines = page_text.split("\n")
        section_header = _guess_section_header(lines, page_num)

        # Chunk page text into paragraphs/sections
        raw_paragraphs = re.split(r"\n{2,}", page_text)
        current_chunk = ""

        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue

            candidate = (current_chunk + "\n\n" + para).strip() if current_chunk else para
            if len(candidate) <= max_chunk_chars:
                current_chunk = candidate
            else:
                if current_chunk:
                    content_hash = hashlib.sha256(current_chunk.encode("utf-8")).hexdigest()
                    chunk_id = f"{doc_version}_p{page_num}_c{global_chunk_idx}"
                    chunks_data.append({
                        "chunk_id": chunk_id,
                        "text": current_chunk,
                        "document_name": doc_name,
                        "document_version": doc_version,
                        "page_number": page_num,
                        "section_header": section_header,
                        "content_hash": content_hash,
                        "timestamp": timestamp,
                        "source_file": filename,
                        "chunk_index": global_chunk_idx,
                    })
                    global_chunk_idx += 1
                current_chunk = para

        if current_chunk:
            content_hash = hashlib.sha256(current_chunk.encode("utf-8")).hexdigest()
            chunk_id = f"{doc_version}_p{page_num}_c{global_chunk_idx}"
            chunks_data.append({
                "chunk_id": chunk_id,
                "text": current_chunk,
                "document_name": doc_name,
                "document_version": doc_version,
                "page_number": page_num,
                "section_header": section_header,
                "content_hash": content_hash,
                "timestamp": timestamp,
                "source_file": filename,
                "chunk_index": global_chunk_idx,
            })
            global_chunk_idx += 1

    return chunks_data
