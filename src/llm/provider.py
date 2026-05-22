"""
LLM & Embeddings provider abstraction.
Switch providers by changing LLM_PROVIDER / EMBEDDING_PROVIDER in .env.
"""
from __future__ import annotations
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
import config


# ─────────────────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Return a LangChain chat model based on LLM_PROVIDER env var.

    Free defaults:
      - groq    → llama3-8b-8192  (no cost, fast)
      - gemini  → gemini-1.5-flash (free tier, generous quota)
      - ollama  → local model, fully free
    Paid:
      - openai  → gpt-4o-mini / gpt-4o
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set. Get a free key at https://console.groq.com")
        return ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_MODEL,
            temperature=config.TEMPERATURE,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set. Get a free key at https://aistudio.google.com")
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=config.TEMPERATURE,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set.")
        return ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_MODEL,
            temperature=config.TEMPERATURE,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
            temperature=config.TEMPERATURE,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            "Choose from: groq, gemini, openai, ollama"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    Return a LangChain embeddings model based on EMBEDDING_PROVIDER env var.

    Free defaults:
      - huggingface → BAAI/bge-small-en-v1.5 (runs locally, no API key)
      - ollama      → nomic-embed-text (runs locally, no API key)
    Paid:
      - openai      → text-embedding-3-small
    """
    provider = config.EMBEDDING_PROVIDER.lower()

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=config.HF_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set.")
        return OpenAIEmbeddings(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_EMBEDDING_MODEL,
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_EMBEDDING_MODEL,
        )

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set.")
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=config.GOOGLE_API_KEY,
        )

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: '{provider}'. "
            "Choose from: huggingface, openai, ollama, gemini"
        )
