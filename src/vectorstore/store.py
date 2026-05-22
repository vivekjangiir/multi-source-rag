"""
Vector store abstraction.
Switch between ChromaDB (local/free), Qdrant (local/free), Pinecone (cloud)
by changing VECTOR_STORE in .env.
"""
from __future__ import annotations
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
import config
from src.llm.provider import get_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Singleton store instance
# ─────────────────────────────────────────────────────────────────────────────
_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Return (or create) the vector store based on VECTOR_STORE env var.

    Free defaults:
      - chroma  → persisted locally at ./data/chroma_db  (no setup needed)
      - qdrant  → run `docker run -p 6333:6333 qdrant/qdrant` first
    Paid/Cloud:
      - pinecone → requires API key and a created index
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance

    embeddings = get_embeddings()
    provider = config.VECTOR_STORE.lower()

    if provider == "chroma":
        from langchain_chroma import Chroma
        _store_instance = Chroma(
            collection_name="rag_documents",
            embedding_function=embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )

    elif provider == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY or None,
        )
        # Create collection if it doesn't exist
        collections = [c.name for c in client.get_collections().collections]
        if config.QDRANT_COLLECTION not in collections:
            # Get embedding dimension dynamically
            sample_emb = embeddings.embed_query("test")
            client.create_collection(
                collection_name=config.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=len(sample_emb), distance=Distance.COSINE),
            )
        _store_instance = QdrantVectorStore(
            client=client,
            collection_name=config.QDRANT_COLLECTION,
            embedding=embeddings,
        )

    elif provider == "pinecone":
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec
        if not config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not set.")
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        if config.PINECONE_INDEX not in pc.list_indexes().names():
            sample_emb = embeddings.embed_query("test")
            pc.create_index(
                name=config.PINECONE_INDEX,
                dimension=len(sample_emb),
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        _store_instance = PineconeVectorStore(
            index_name=config.PINECONE_INDEX,
            embedding=embeddings,
            pinecone_api_key=config.PINECONE_API_KEY,
        )

    else:
        raise ValueError(
            f"Unknown VECTOR_STORE: '{provider}'. "
            "Choose from: chroma, qdrant, pinecone"
        )

    return _store_instance


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def add_documents(docs: List[Document]) -> List[str]:
    """Embed and store documents. Returns list of stored IDs."""
    store = get_vector_store()
    return store.add_documents(docs)


def similarity_search(query: str, k: int = None, filter: dict = None) -> List[Document]:
    """Retrieve top-k most similar documents for a query."""
    store = get_vector_store()
    k = k or config.TOP_K_RESULTS
    kwargs = {"k": k}
    if filter:
        kwargs["filter"] = filter
    return store.similarity_search(query, **kwargs)


def similarity_search_with_score(query: str, k: int = None) -> List[tuple[Document, float]]:
    """Retrieve documents with relevance scores."""
    store = get_vector_store()
    k = k or config.TOP_K_RESULTS
    return store.similarity_search_with_relevance_scores(query, k=k)


def reset_store() -> None:
    """Clear the singleton so a new store is created on next access."""
    global _store_instance
    _store_instance = None
