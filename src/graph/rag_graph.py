"""
LangGraph RAG pipeline.

Graph flow:
  ┌─────────────────────────────────────────────────┐
  │                  query_router                    │
  └──────────┬──────────────────────────────────────┘
             │ (parallel)
  ┌──────────▼──────────┐   ┌───────────────────────┐
  │   retrieve_vector   │   │     retrieve_web       │
  └──────────┬──────────┘   └───────────┬───────────┘
             │                          │
             └──────────┬───────────────┘
                        ▼
               grade_documents
                        ▼
               generate_answer
"""
from __future__ import annotations
from langgraph.graph import StateGraph, END
from src.graph.state import RAGState
from src.graph.nodes import (
    query_router,
    retrieve_vector,
    retrieve_web,
    grade_documents,
    generate_answer,
)


def build_rag_graph():
    """Compile and return the RAG StateGraph."""
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("query_router",    query_router)
    graph.add_node("retrieve_vector", retrieve_vector)
    graph.add_node("retrieve_web",    retrieve_web)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate_answer", generate_answer)

    # Entry point
    graph.set_entry_point("query_router")

    # Router → parallel retrieval
    graph.add_edge("query_router", "retrieve_vector")
    graph.add_edge("query_router", "retrieve_web")

    # Both retrievals → grader
    graph.add_edge("retrieve_vector", "grade_documents")
    graph.add_edge("retrieve_web",    "grade_documents")

    # Grader → generator
    graph.add_edge("grade_documents", "generate_answer")

    # End
    graph.add_edge("generate_answer", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public query function
# ─────────────────────────────────────────────────────────────────────────────

_compiled_graph = None


def query(user_query: str) -> dict:
    """
    Run the full RAG pipeline for a user query.

    Returns:
        {
          "answer": "...",
          "citations": [
            {"number": 1, "text": "[YouTube] ...", "url": "...", ...},
            ...
          ]
        }
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_rag_graph()

    initial_state: RAGState = {
        "query": user_query,
        "use_web_search": False,
        "vector_docs": [],
        "web_docs": [],
        "graded_docs": [],
        "answer": "",
        "citations": [],
        "error": None,
    }

    final_state = _compiled_graph.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "citations": final_state["citations"],
    }
