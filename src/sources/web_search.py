"""
Web search source — live results injected as context.
Also handles ingesting arbitrary web URLs into the vector store.

Search providers (switchable via WEB_SEARCH_PROVIDER):
  - duckduckgo  → free, no API key needed  (default)
  - tavily      → free tier: 1000 searches/month
  - serpapi     → paid
"""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


# ─────────────────────────────────────────────────────────────────────────────
# Live web search (returns Documents directly — not stored in vector DB)
# ─────────────────────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = None) -> List[Document]:
    """
    Run a live web search and return results as Documents.
    These are NOT stored in the vector store — they are injected directly
    as context for the current query.
    """
    max_results = max_results or config.WEB_SEARCH_MAX_RESULTS
    provider = config.WEB_SEARCH_PROVIDER.lower()

    if provider == "duckduckgo":
        return _search_duckduckgo(query, max_results)
    elif provider == "tavily":
        return _search_tavily(query, max_results)
    elif provider == "serpapi":
        return _search_serpapi(query, max_results)
    else:
        raise ValueError(
            f"Unknown WEB_SEARCH_PROVIDER: '{provider}'. "
            "Choose from: duckduckgo, tavily, serpapi"
        )


def _search_duckduckgo(query: str, max_results: int) -> List[Document]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError("Run: pip install duckduckgo-search")

    docs = []
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    for r in results:
        docs.append(Document(
            page_content=r.get("body", ""),
            metadata={
                "source_type": "web_search",
                "search_provider": "duckduckgo",
                "title": r.get("title", ""),
                "source_url": r.get("href", ""),
                "citation": f"[Web] {r.get('title', 'Untitled')} — {r.get('href', '')}",
            }
        ))
    return docs


def _search_tavily(query: str, max_results: int) -> List[Document]:
    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError("Run: pip install tavily-python")

    if not config.TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY not set.")

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    response = client.search(query=query, max_results=max_results, search_depth="basic")

    docs = []
    for r in response.get("results", []):
        docs.append(Document(
            page_content=r.get("content", ""),
            metadata={
                "source_type": "web_search",
                "search_provider": "tavily",
                "title": r.get("title", ""),
                "source_url": r.get("url", ""),
                "citation": f"[Web] {r.get('title', 'Untitled')} — {r.get('url', '')}",
            }
        ))
    return docs


def _search_serpapi(query: str, max_results: int) -> List[Document]:
    try:
        from serpapi import GoogleSearch
    except ImportError:
        raise ImportError("Run: pip install google-search-results")

    if not config.SERPAPI_API_KEY:
        raise ValueError("SERPAPI_API_KEY not set.")

    search = GoogleSearch({"q": query, "api_key": config.SERPAPI_API_KEY, "num": max_results})
    results = search.get_dict().get("organic_results", [])

    docs = []
    for r in results[:max_results]:
        docs.append(Document(
            page_content=r.get("snippet", ""),
            metadata={
                "source_type": "web_search",
                "search_provider": "serpapi",
                "title": r.get("title", ""),
                "source_url": r.get("link", ""),
                "citation": f"[Web] {r.get('title', 'Untitled')} — {r.get('link', '')}",
            }
        ))
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Ingest a web URL into the vector store
# ─────────────────────────────────────────────────────────────────────────────

def load_web_url(url: str) -> List[Document]:
    """
    Fetch a web page, extract its text, chunk it, and return Documents
    ready to be stored in the vector store.
    """
    try:
        from langchain_community.document_loaders import WebBaseLoader
    except ImportError:
        raise ImportError("Run: pip install langchain-community beautifulsoup4")

    loader = WebBaseLoader(url)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.MAX_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    docs = []
    for page in pages:
        title = page.metadata.get("title", url)
        for i, chunk in enumerate(splitter.split_text(page.page_content)):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source_type": "web_url",
                    "source_url": url,
                    "title": title,
                    "chunk_index": i,
                    "citation": f"[Web] {title} — {url}",
                }
            ))

    return docs
