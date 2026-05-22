"""
LangGraph state schema for the RAG pipeline.
"""
from __future__ import annotations
from typing import List, Optional, TypedDict
from langchain_core.documents import Document


class RAGState(TypedDict):
    """State that flows through the RAG graph."""
    query: str                              # User's original question
    use_web_search: bool                    # Whether to run live web search
    vector_docs: List[Document]             # Docs retrieved from vector store
    web_docs: List[Document]                # Docs from live web search
    graded_docs: List[Document]             # Docs after relevance grading
    answer: str                             # Final generated answer
    citations: List[dict]                   # Structured citation list
    error: Optional[str]                    # Error message if something failed
