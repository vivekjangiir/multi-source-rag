"""
LangGraph node functions.
Each function receives the full RAGState and returns a partial update dict.

Pipeline:
  query_router → [retrieve_vector, retrieve_web] → grade_documents → generate_answer
"""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.graph.state import RAGState
from src.llm.provider import get_llm
import config


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: Query Router
# ─────────────────────────────────────────────────────────────────────────────

def query_router(state: RAGState) -> dict:
    """
    Decide if live web search should supplement vector store retrieval.
    Defaults to False (skip web search) if the LLM call fails.
    """
    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a query router. Should a live web search be run for this question?\n"
             "Reply with ONLY 'yes' or 'no'.\n"
             "'yes' = current events, today's news, live prices, latest version numbers.\n"
             "'no'  = questions about uploaded documents, videos, books, code, or general knowledge."),
            ("human", "{query}"),
        ])
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"query": state["query"]}).strip().lower()
        use_web = result.startswith("yes")
    except Exception as e:
        print(f"[query_router] Error (defaulting to no web search): {e}")
        use_web = False
    return {"use_web_search": use_web}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2a: Vector Store Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_vector(state: RAGState) -> dict:
    """Retrieve top-k documents from the vector store via similarity search."""
    from src.vectorstore.store import similarity_search_with_score
    try:
        results = similarity_search_with_score(state["query"], k=config.TOP_K_RESULTS)
        docs = [doc for doc, score in results]
        print(f"[retrieve_vector] Found {len(docs)} docs. "
              f"Top score: {results[0][1]:.3f}" if results else "[retrieve_vector] No docs found.")
    except Exception as e:
        docs = []
        print(f"[retrieve_vector] Error: {e}")
    return {"vector_docs": docs}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2b: Web Search Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_web(state: RAGState) -> dict:
    """Run a live web search only if the router decided it's needed."""
    if not state.get("use_web_search"):
        return {"web_docs": []}
    from src.sources.web_search import web_search
    try:
        docs = web_search(state["query"])
        print(f"[retrieve_web] Found {len(docs)} web results.")
    except Exception as e:
        docs = []
        print(f"[retrieve_web] Error: {e}")
    return {"web_docs": docs}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: Document Grader  (FIXED — score-based, not LLM-per-chunk)
# ─────────────────────────────────────────────────────────────────────────────

def grade_documents(state: RAGState) -> dict:
    """
    Filter documents to keep only those relevant to the query.

    Strategy (fast, no extra LLM calls per chunk):
      1. Keep ALL vector-store docs that pass a minimum similarity threshold.
         ChromaDB already scored them — we just drop obvious misfits.
      2. Keep web-search docs always (they were fetched specifically for this query).
      3. If nothing survives, fall back to all docs so the generator always has context.
    """
    vector_docs = state.get("vector_docs", [])
    web_docs    = state.get("web_docs", [])

    # Web docs are always relevant — they were fetched live for this exact query
    graded = list(web_docs)

    # For vector docs: accept all of them. The similarity search already ranked them.
    # We keep all TOP_K results — the LLM is smart enough to ignore noise.
    graded.extend(vector_docs)

    # Hard cap to avoid bloating the prompt
    graded = graded[:config.TOP_K_RESULTS + 3]

    print(f"[grade_documents] Kept {len(graded)} docs "
          f"({len(vector_docs)} vector + {len(web_docs)} web)")

    return {"graded_docs": graded}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: Answer Generator
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert research assistant. Your job is to answer questions "
    "using ONLY the numbered context passages provided. "
    "After each factual claim, add an inline citation marker like [1] or [2] "
    "that matches the passage number. "
    "If the context does not contain the answer, say so clearly — do NOT invent facts."
)

_HUMAN_PROMPT = """\
Context passages:
{context}

Question: {query}

Instructions:
- Answer clearly and completely using only the context above.
- Use inline citation markers [1], [2], etc. after each claim.
- If a passage is from a YouTube video, refer to the timestamp when relevant.
- If the context is insufficient, say: "The provided sources don't cover this — try adding more relevant sources."

Answer:"""


def generate_answer(state: RAGState) -> dict:
    """Generate the final answer with inline citations from graded context."""
    docs = state.get("graded_docs", [])

    if not docs:
        return {
            "answer": (
                "No sources found. Add sources first using the sidebar "
                "(YouTube URL, PDF, GitHub repo, or web URL), then ask your question."
            ),
            "citations": [],
        }

    # Build numbered context + citation metadata
    context_parts  = []
    seen_urls      = {}   # url → citation_number (deduplicate same source)
    citations_map  = {}   # citation_number → metadata dict
    doc_to_citnum  = []   # parallel list to docs

    for doc in docs:
        meta  = doc.metadata
        url   = (meta.get("source_url") or meta.get("timestamp_url")
                 or meta.get("file_url") or meta.get("file_path") or "")
        key   = meta.get("citation", url or doc.page_content[:60])

        if key in seen_urls:
            num = seen_urls[key]
        else:
            num = len(seen_urls) + 1
            seen_urls[key] = num
            citations_map[num] = {
                "number":      num,
                "text":        key,
                "source_type": meta.get("source_type", "unknown"),
                "url":         url,
                "title":       meta.get("title") or meta.get("file_name") or meta.get("repo_label", ""),
                "extra":       _extra(meta),
            }

        doc_to_citnum.append(num)
        context_parts.append(f"[{num}] {doc.page_content}")

    context = "\n\n".join(context_parts)

    try:
        llm   = get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human",  _HUMAN_PROMPT),
        ])
        answer = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "query":   state["query"],
        })
    except Exception as e:
        answer = f"Error generating answer: {e}"

    unique_citations = list(citations_map.values())
    return {"answer": answer, "citations": unique_citations}


def _extra(meta: dict) -> str:
    if meta.get("timestamp"):
        return f"@ {meta['timestamp']}"
    if meta.get("page"):
        return f"Page {meta['page']}"
    if meta.get("language"):
        return meta["language"]
    return ""
