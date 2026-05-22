"""
PDF, DOCX, and plain text file loaders.
Each chunk carries page number / section metadata for citations.
"""
from __future__ import annotations
import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.MAX_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and return chunked Documents with page-level citations.

    Metadata per chunk:
      - source_type: "pdf"
      - file_name: base filename
      - file_path: full path
      - page: page number (1-indexed)
      - citation: "[PDF] filename.pdf — Page N"
    """
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
    except ImportError:
        raise ImportError("Run: pip install pymupdf langchain-community")

    loader = PyMuPDFLoader(file_path)
    pages = loader.load()

    file_name = os.path.basename(file_path)
    splitter = _get_splitter()
    docs = []

    for page in pages:
        page_num = page.metadata.get("page", 0) + 1  # 0-indexed → 1-indexed
        chunks = splitter.split_text(page.page_content)
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source_type": "pdf",
                    "file_name": file_name,
                    "file_path": file_path,
                    "page": page_num,
                    "citation": f"[PDF] {file_name} — Page {page_num}",
                }
            ))

    return docs


def load_docx(file_path: str) -> List[Document]:
    """
    Load a Word document (.docx) and return chunked Documents.

    Metadata per chunk:
      - source_type: "docx"
      - file_name, file_path
      - citation: "[DOCX] filename.docx"
    """
    try:
        from langchain_community.document_loaders import Docx2txtLoader
    except ImportError:
        raise ImportError("Run: pip install docx2txt langchain-community")

    loader = Docx2txtLoader(file_path)
    pages = loader.load()
    file_name = os.path.basename(file_path)
    splitter = _get_splitter()
    docs = []

    for i, page in enumerate(pages):
        for chunk in splitter.split_text(page.page_content):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source_type": "docx",
                    "file_name": file_name,
                    "file_path": file_path,
                    "section": i + 1,
                    "citation": f"[DOCX] {file_name}",
                }
            ))

    return docs


def load_text(file_path: str) -> List[Document]:
    """
    Load a plain text file and return chunked Documents.
    """
    file_name = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    splitter = _get_splitter()
    docs = []
    for i, chunk in enumerate(splitter.split_text(text)):
        docs.append(Document(
            page_content=chunk,
            metadata={
                "source_type": "text",
                "file_name": file_name,
                "file_path": file_path,
                "chunk_index": i,
                "citation": f"[Text] {file_name}",
            }
        ))

    return docs


def load_file(file_path: str) -> List[Document]:
    """Auto-detect file type and load accordingly."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return load_docx(file_path)
    elif ext in (".txt", ".md"):
        return load_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .docx, .txt, .md")
