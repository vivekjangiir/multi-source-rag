"""
Central configuration for Multi-Source RAG App.
All settings are driven by environment variables with sensible free-tier defaults.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ─────────────────────────────────────────────────────────────────────
LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "groq")          # groq | gemini | openai | ollama | nvidia
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL         = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "llama3")

# ── Anthropic ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL    = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

# ── NVIDIA NIM ───────────────────────────────────────────────────────────────
NVIDIA_API_KEY     = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL    = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL       = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
NVIDIA_EMBEDDING_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER        = os.getenv("EMBEDDING_PROVIDER", "huggingface")   # huggingface | openai | ollama | gemini
HF_EMBEDDING_MODEL        = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
OPENAI_EMBEDDING_MODEL    = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OLLAMA_EMBEDDING_MODEL    = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ── Vector Store ─────────────────────────────────────────────────────────────
VECTOR_STORE          = os.getenv("VECTOR_STORE", "chroma")     # chroma | qdrant | pinecone
# Absolute path so it doesn't matter what directory the server is launched from
_PROJECT_ROOT      = os.path.dirname(os.path.abspath(__file__))
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    os.path.join(_PROJECT_ROOT, "data", "chroma_db")
)
QDRANT_URL            = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY        = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION     = os.getenv("QDRANT_COLLECTION", "rag_documents")
PINECONE_API_KEY      = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX        = os.getenv("PINECONE_INDEX", "rag-index")
PINECONE_ENVIRONMENT  = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")

# ── Web Search ───────────────────────────────────────────────────────────────
WEB_SEARCH_PROVIDER    = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo")   # duckduckgo | tavily | serpapi
TAVILY_API_KEY         = os.getenv("TAVILY_API_KEY", "")
SERPAPI_API_KEY        = os.getenv("SERPAPI_API_KEY", "")
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))

# ── GitHub ───────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# ── App ──────────────────────────────────────────────────────────────────────
APP_NAME       = os.getenv("APP_NAME", "Multi-Source RAG")
DEBUG          = os.getenv("DEBUG", "false").lower() == "true"
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS  = int(os.getenv("TOP_K_RESULTS", "5"))
TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.1"))

# ── Source types enum ────────────────────────────────────────────────────────
class SourceType:
    YOUTUBE      = "youtube"
    PDF          = "pdf"
    DOCX         = "docx"
    WEB_URL      = "web_url"
    GITHUB       = "github"
    TEXT         = "text"
    WEB_SEARCH   = "web_search"
    KNOWLEDGE_BASE = "knowledge_base"
