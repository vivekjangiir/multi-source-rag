"""
Unified ingestion entry point.
Detects source type and routes to the correct loader, then stores in vector DB.
"""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from src.vectorstore.store import add_documents
import config


def ingest(source: str, source_type: str = None) -> dict:
    """
    Ingest any source into the vector store.

    Args:
        source:       URL, file path, or raw text
        source_type:  Optional override. Auto-detected if not provided.
                      Values: youtube | pdf | docx | text | web_url | github

    Returns:
        {"status": "ok", "source_type": ..., "chunks_added": N}
    """
    detected_type = source_type or _detect_type(source)
    docs = _load(source, detected_type)

    if not docs:
        return {"status": "error", "message": "No content extracted from source."}

    ids = add_documents(docs)
    return {
        "status": "ok",
        "source_type": detected_type,
        "source": source,
        "chunks_added": len(ids),
    }


def _detect_type(source: str) -> str:
    """Auto-detect source type from the input string."""
    lower = source.lower()
    if "youtube.com" in lower or "youtu.be" in lower:
        return config.SourceType.YOUTUBE
    if lower.endswith(".pdf"):
        return config.SourceType.PDF
    if lower.endswith(".docx") or lower.endswith(".doc"):
        return config.SourceType.DOCX
    if lower.endswith(".txt") or lower.endswith(".md"):
        return config.SourceType.TEXT
    if lower.startswith("https://github.com"):
        return config.SourceType.GITHUB
    if lower.startswith("http://") or lower.startswith("https://"):
        return config.SourceType.WEB_URL
    return config.SourceType.TEXT


def _load(source: str, source_type: str) -> List[Document]:
    """Route to the correct loader."""
    if source_type == config.SourceType.YOUTUBE:
        from src.sources.youtube import load_youtube
        return load_youtube(source)

    elif source_type == config.SourceType.PDF:
        from src.sources.pdf import load_pdf
        return load_pdf(source)

    elif source_type == config.SourceType.DOCX:
        from src.sources.pdf import load_docx
        return load_docx(source)

    elif source_type == config.SourceType.TEXT:
        from src.sources.pdf import load_text
        return load_text(source)

    elif source_type == config.SourceType.WEB_URL:
        from src.sources.web_search import load_web_url
        return load_web_url(source)

    elif source_type == config.SourceType.GITHUB:
        from src.sources.github import load_github
        return load_github(source)

    else:
        raise ValueError(f"Unknown source type: {source_type}")
