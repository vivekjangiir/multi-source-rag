"""
FastAPI backend for Multi-Source RAG App.

Endpoints:
  POST /ingest          — Add a source (URL, file path, YouTube, GitHub, etc.)
  POST /query           — Ask a question, get answer + citations
  GET  /sources         — List all ingested source types
  GET  /health          — Health check
"""
from __future__ import annotations
import os
import shutil
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import config

app = FastAPI(
    title=config.APP_NAME,
    description="Multi-source RAG with citations — LangChain + LangGraph",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    source: str
    source_type: Optional[str] = None   # auto-detected if omitted


class QueryRequest(BaseModel):
    query: str


class IngestResponse(BaseModel):
    status: str
    source_type: str
    source: str
    chunks_added: int


class Citation(BaseModel):
    number: int
    text: str
    source_type: str
    url: str
    title: str
    extra: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_ui():
    """Serve the chat UI at http://localhost:8000"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "frontend", "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_provider": config.LLM_PROVIDER,
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "vector_store": config.VECTOR_STORE,
        "web_search_provider": config.WEB_SEARCH_PROVIDER,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest_source(req: IngestRequest):
    """Ingest any source into the vector store."""
    import traceback
    try:
        from src.sources.ingest import ingest
        result = ingest(req.source, req.source_type)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except HTTPException:
        raise
    except ValueError as e:
        # ValueError = expected user-facing errors (bad URL, blocked, etc.)
        # Return 400 with the clean message, no stack trace needed
        msg = str(e)
        print(f"[ingest] ValueError — {msg[:120]}")
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        print(f"[ingest] ERROR — {detail}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=detail)


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Upload a file (PDF, DOCX, TXT) and ingest it."""
    import traceback
    try:
        upload_dir = os.path.join(os.path.dirname(__file__), "data", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        from src.sources.ingest import ingest
        result = ingest(file_path)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        print(f"[ingest/file] ERROR — {detail}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=detail)


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    """Ask a question. Returns the answer and a list of citations."""
    import traceback
    try:
        from src.graph.rag_graph import query
        result = query(req.query)
        return result
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        print(f"[query] ERROR — {detail}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=detail)


@app.get("/config")
def get_config():
    """Return current provider configuration (no secrets)."""
    return {
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": {
            "groq": config.GROQ_MODEL,
            "gemini": config.GEMINI_MODEL,
            "openai": config.OPENAI_MODEL,
            "anthropic": config.ANTHROPIC_MODEL,
            "ollama": config.OLLAMA_MODEL,
        }.get(config.LLM_PROVIDER, "unknown"),
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "vector_store": config.VECTOR_STORE,
        "web_search_provider": config.WEB_SEARCH_PROVIDER,
        "chunk_size": config.MAX_CHUNK_SIZE,
        "top_k": config.TOP_K_RESULTS,
    }


@app.get("/debug/retrieve")
def debug_retrieve(q: str = "what is this about"):
    """
    Run similarity search and return scores so you can see what gets retrieved.
    Usage: http://localhost:8000/debug/retrieve?q=summarize+the+video
    """
    try:
        from src.vectorstore.store import similarity_search_with_score
        results = similarity_search_with_score(q, k=8)
        return {
            "query": q,
            "results": [
                {
                    "score":       round(score, 4),
                    "source_type": doc.metadata.get("source_type", "?"),
                    "citation":    doc.metadata.get("citation", ""),
                    "preview":     doc.page_content[:120],
                }
                for doc, score in results
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/debug/store")
def debug_store():
    """
    Show what's currently stored in the vector store.
    Useful for verifying that ingestion worked.
    """
    try:
        from src.vectorstore.store import get_vector_store
        store = get_vector_store()

        # ChromaDB exposes .get() directly
        if hasattr(store, '_collection'):
            col = store._collection
            result = col.get(include=["metadatas", "documents"], limit=50)
            docs = result.get("documents", [])
            metas = result.get("metadatas", [])
            items = []
            for doc, meta in zip(docs, metas):
                items.append({
                    "source_type": meta.get("source_type", "?"),
                    "citation":    meta.get("citation", ""),
                    "preview":     doc[:120] + "..." if len(doc) > 120 else doc,
                })
            return {"total_chunks": len(items), "chunks": items}

        # Fallback: do a broad search
        results = store.similarity_search("what is this about", k=20)
        return {
            "total_chunks": len(results),
            "chunks": [
                {
                    "source_type": d.metadata.get("source_type", "?"),
                    "citation":    d.metadata.get("citation", ""),
                    "preview":     d.page_content[:120],
                }
                for d in results
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@app.delete("/store/clear")
def clear_store():
    """
    Wipe the entire vector store — deletes ChromaDB folder from disk.
    Server recreates it fresh on the next ingest.
    """
    import traceback
    try:
        # 1. Drop in-memory singleton so next access creates a fresh client
        try:
            from src.vectorstore.store import reset_store
            reset_store()
        except Exception:
            pass  # singleton may not exist yet — that's fine

        # 2. Delete entire data directory from disk
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print(f"[clear_store] Deleted {data_dir}")
            return {"status": "cleared", "deleted": data_dir}

        return {"status": "already empty"}

    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        print(f"[clear_store] ERROR — {detail}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=detail)


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=config.DEBUG)
