---
title: Multi Source RAG with Citations
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🔍 Multi-Source RAG with Citations

> Ingest **YouTube videos, PDFs, Word docs, web pages, GitHub repos, and live web search** — get answers with exact citations back to the source, timestamp, and page number.

Built with **LangChain + LangGraph**. Every provider (LLM, embeddings, vector store, web search) swaps via a single `.env` line. **Runs entirely free by default.**

🚀 **Live Demo:** [huggingface.co/spaces/vieveksharmaa/multi-source-rag](https://huggingface.co/spaces/vieveksharmaa/multi-source-rag)  
📦 **GitHub:** [github.com/vieveksharmaa/multi-source-rag](https://github.com/vieveksharmaa/multi-source-rag)

---

## Table of Contents

1. [What Is RAG?](#1-what-is-rag)
2. [Architecture](#2-architecture)
3. [Local Setup](#3-local-setup)
4. [LLM Providers](#4-llm-providers)
5. [Embedding Providers](#5-embedding-providers)
6. [Vector Stores](#6-vector-stores)
7. [Web Search Providers](#7-web-search-providers)
8. [Source Types](#8-source-types)
9. [Deployment](#9-deployment)
10. [Environment Variables Reference](#10-environment-variables-reference)
11. [API Reference](#11-api-reference)
12. [Troubleshooting](#12-troubleshooting)
13. [Tech Stack](#13-tech-stack)

---

## 1. What Is RAG?

**RAG (Retrieval-Augmented Generation)** grounds an LLM's answer in real documents rather than training data. Instead of hallucinating, the model reads the relevant passages first, then answers from them — with citations.

**Why multi-source?** Most RAG apps handle one source type. This app handles six simultaneously:

| Source | Why it matters |
|---|---|
| YouTube | Massive knowledge base; transcripts are free |
| PDF / DOCX | Research papers, reports, books, manuals |
| Web URLs | Articles, docs, any public page |
| GitHub repos | Understand codebases by asking questions |
| Live web search | Real-time info — news, current prices, latest releases |
| Plain text / Markdown | Notes, logs, any raw text |

---

## 2. Architecture

```
                       ┌─────────────────────┐
                       │     User Query       │
                       └──────────┬──────────┘
                                  │
                       ┌──────────▼──────────┐
                       │    Query Router      │  LLM decides: needs live web?
                       └────────┬────────────┘
                       (parallel fan-out)
            ┌──────────────────┴──────────────────┐
            │                                      │
 ┌──────────▼──────────┐              ┌────────────▼───────────┐
 │   Vector Retrieval   │              │     Web Search          │
 │ (ChromaDB / Qdrant / │              │ (DuckDuckGo / Tavily)   │
 │  Pinecone)           │              └────────────┬───────────┘
 └──────────┬──────────┘                           │
            └──────────────────┬──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Document Grader     │  Filters irrelevant chunks
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Answer Generator    │  Writes answer with [1][2] citations
                    └─────────────────────┘
```

**Ingestion pipeline:**

```
Source URL / File
     │
Source Loader    (YouTube transcript / PDF parser / web scraper / GitHub walker)
     │
Text Splitter    (RecursiveCharacterTextSplitter, 1000 chars, 200 overlap)
     │
Embeddings       (HuggingFace / OpenAI / Ollama / Gemini / NVIDIA)
     │
Vector Store     (ChromaDB / Qdrant / Pinecone)
```

---

## 3. Local Setup

### Prerequisites

- Python 3.11
- `git`
- A free [Groq API key](https://console.groq.com) (60 seconds to get)

### Step 1 — Clone

```bash
git clone https://github.com/vieveksharmaa/multi-source-rag.git
cd multi-source-rag
```

### Step 2 — Virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure

```bash
cp .env.example .env
```

Set at minimum:

```env
GROQ_API_KEY=your_key_here   # Free at https://console.groq.com
```

Everything else has a working free default.

### Step 5 — Run

```bash
python main.py
```

Open **http://localhost:8000**. The first ingest downloads the HuggingFace embedding model (~80 MB, cached after that).

### Step 6 — Test it

1. Paste any YouTube URL in the sidebar → click **Ingest**
2. Ask a question about the video
3. Answer appears with `[1] [YouTube] Title @ 2:34` citations

---

## 4. LLM Providers

Set `LLM_PROVIDER=` in `.env` to switch. Supported: **`groq` · `gemini` · `openai` · `anthropic` · `nvidia` · `ollama`**

---

### 4.1 Groq ✅ Default

Runs Llama/Gemma on custom LPU hardware — 10–20× faster than a GPU server. Free with generous rate limits.

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...           # https://console.groq.com
GROQ_MODEL=llama-3.1-8b-instant
# or: llama-3.3-70b-versatile
```

| Model | Speed | Quality | Rate limit |
|---|---|---|---|
| `llama-3.1-8b-instant` | ⚡⚡⚡ | ⭐⭐⭐ | 30k tok/min |
| `llama-3.3-70b-versatile` | ⚡⚡ | ⭐⭐⭐⭐⭐ | 6k tok/min |
| `gemma2-9b-it` | ⚡⚡⚡ | ⭐⭐⭐ | 15k tok/min |

---

### 4.2 Google Gemini

Free tier: 15 req/min, 1M tokens/day. Good for longer contexts.

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=AIza...          # https://aistudio.google.com
GEMINI_MODEL=gemini-1.5-flash
# or: gemini-1.5-pro
```

Uncomment in `requirements.txt`: `langchain-google-genai>=2.0.0`

---

### 4.3 OpenAI

Industry standard. GPT-4o is the highest quality option. Paid only.

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...           # https://platform.openai.com
OPENAI_MODEL=gpt-4o-mini
# or: gpt-4o
```

Uncomment in `requirements.txt`: `langchain-openai>=0.2.0`

---

### 4.4 Anthropic Claude ✅ Supported

Claude models via the Anthropic API. Excellent at following instructions and producing well-structured answers with citations.

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...    # https://console.anthropic.com
ANTHROPIC_MODEL=claude-3-5-haiku-20241022
# or: claude-3-5-sonnet-20241022   # higher quality
# or: claude-opus-4-6              # best quality
```

Uncomment in `requirements.txt`: `langchain-anthropic>=0.3.0`

**Model comparison:**

| Model | Speed | Quality | Best for |
|---|---|---|---|
| `claude-3-5-haiku-20241022` | ⚡⚡⚡ | ⭐⭐⭐⭐ | Fast, cost-effective |
| `claude-3-5-sonnet-20241022` | ⚡⚡ | ⭐⭐⭐⭐⭐ | Best balance |
| `claude-opus-4-6` | ⚡ | ⭐⭐⭐⭐⭐ | Maximum quality |

---

### 4.5 NVIDIA NIM (free, 100+ models)

One API endpoint with 100+ open-source models. Free tier, no credit card.

```env
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...        # https://build.nvidia.com
NVIDIA_MODEL=meta/llama-3.1-8b-instruct
```

Popular models: `meta/llama-3.3-70b-instruct` · `deepseek-ai/deepseek-r1` · `qwen/qwen2.5-72b-instruct`  
Full catalogue: [build.nvidia.com/explore/discover](https://build.nvidia.com/explore/discover)

---

### 4.6 Ollama (Local)

Run models entirely on your own machine. No API key, no cost, no data leaves your computer.

```bash
ollama pull llama3    # 4.7GB
```

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

Uncomment in `requirements.txt`: `langchain-ollama>=0.2.0`

---

## 5. Embedding Providers

Set `EMBEDDING_PROVIDER=` in `.env`. Supported: **`huggingface` · `openai` · `gemini` · `nvidia` · `ollama`**

> ⚠️ If you change embedding provider after ingesting, delete `./data/chroma_db/` and re-ingest. Vectors from different models are incompatible.

---

### 5.1 HuggingFace (Local) ✅ Default

Downloads and runs a small model locally. No API key. Free forever.

```env
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

| Model | Size | Quality |
|---|---|---|
| `all-MiniLM-L6-v2` | 80 MB | ⭐⭐⭐ |
| `BAAI/bge-small-en-v1.5` | 130 MB | ⭐⭐⭐⭐ |
| `BAAI/bge-large-en-v1.5` | 1.3 GB | ⭐⭐⭐⭐⭐ |

---

### 5.2 OpenAI Embeddings

Best commercial quality. ~$0.02/1M tokens.

```env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

---

### 5.3 Google Gemini Embeddings

Free tier. Pairs well with Gemini LLM.

```env
EMBEDDING_PROVIDER=gemini
GOOGLE_API_KEY=AIza...
```

---

### 5.4 NVIDIA NIM Embeddings

Purpose-built for retrieval Q&A. No local memory usage.

```env
EMBEDDING_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...
NVIDIA_EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
```

---

### 5.5 Ollama Embeddings (Local)

```bash
ollama pull nomic-embed-text
```

```env
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

---

## 6. Vector Stores

Set `VECTOR_STORE=` in `.env`. Supported: **`chroma` · `qdrant` · `pinecone`**

---

### 6.1 ChromaDB ✅ Default

Zero setup. Persists to disk. Fast for up to ~100K vectors.

```env
VECTOR_STORE=chroma
CHROMA_PERSIST_DIR=./data/chroma_db
```

---

### 6.2 Qdrant

High-performance. Runs locally via Docker or on Qdrant Cloud (free 1GB tier).

```bash
docker run -p 6333:6333 qdrant/qdrant
```

```env
VECTOR_STORE=qdrant
QDRANT_URL=http://localhost:6333
```

Uncomment in `requirements.txt`: `langchain-qdrant>=0.1.0` and `qdrant-client>=1.9.0`

---

### 6.3 Pinecone

Fully managed, serverless. Free tier: 1 index, 2GB.

```env
VECTOR_STORE=pinecone
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX=rag-index
```

Uncomment in `requirements.txt`: `langchain-pinecone>=0.2.0` and `pinecone-client>=4.0.0`

**Dimension guide** (must match embedding model):

| Embedding model | Dimension |
|---|---|
| `all-MiniLM-L6-v2` | 384 |
| `BAAI/bge-small-en-v1.5` | 384 |
| `text-embedding-3-small` | 1536 |
| `nvidia/nv-embedqa-e5-v5` | 1024 |

---

## 7. Web Search Providers

Set `WEB_SEARCH_PROVIDER=` in `.env`. Triggered automatically when a query needs live data.

| Provider | Cost | Setup |
|---|---|---|
| `duckduckgo` ✅ default | Free | None |
| `tavily` | 1000 searches/mo free | [tavily.com](https://tavily.com) |
| `serpapi` | Paid | [serpapi.com](https://serpapi.com) |

---

## 8. Source Types

| Source | How to ingest | Citation format |
|---|---|---|
| **YouTube** | Paste URL in sidebar | `[YouTube] "Title" @ 2:34 — youtube.com/...?t=154s` |
| **PDF** | Upload file or local path | `[PDF] "filename.pdf" — Page 5` |
| **DOCX** | Upload file | `[DOCX] "filename.docx"` |
| **Web URL** | Paste `https://` URL | `[Web] "Page Title" — example.com` |
| **GitHub repo** | Paste `https://github.com/org/repo` | `[GitHub] org/repo — src/auth.py` |
| **Live search** | Automatic (no ingestion needed) | `[Web Search] "Result Title"` |

**GitHub note:** Set `GITHUB_TOKEN=ghp_...` for private repos or to avoid rate limits.

**YouTube note:** Cloud platforms (HuggingFace, Render) may have YouTube block transcript requests. Run locally if this is an issue.

---

## 9. Deployment

### 9.1 Hugging Face Spaces ✅ Recommended (free)

16GB RAM, persistent storage, no sleep on public spaces.

1. Push to GitHub
2. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space) — SDK: **Docker**, Hardware: **CPU Basic**
3. Add HF as a remote and push:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/multi-source-rag
   git push hf main
   ```
4. Add secrets: Space → **Settings** → **Variables and Secrets** → add `GROQ_API_KEY` (or your chosen provider key)

Your app runs at `https://YOUR_HF_USERNAME-multi-source-rag.hf.space`

**Redeploy:**
```bash
git add . && git commit -m "update" && git push hf main
```

---

### 9.2 Railway

$5 free credit/month. Auto-deploys from GitHub. No sleep.

1. [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Settings → Variables → add your `.env` values
3. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

---

### 9.3 Render

Free tier available but only 512MB RAM (embedding model is tight) and ephemeral filesystem.

1. [render.com](https://render.com) → New Web Service → connect GitHub repo
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables in Dashboard → Environment

> Recommendation: Use HuggingFace Spaces instead — same cost ($0), 32× more RAM.

---

### 9.4 Fly.io

Free: 3 VMs + 3GB persistent volume. Docker-based.

```bash
fly auth login
fly launch
fly secrets set GROQ_API_KEY=gsk_...
fly volumes create rag_data --size 1
fly deploy
```

---

### 9.5 Self-Hosted VPS

Best for production. Hetzner starts at ~€4/month.

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git
git clone https://github.com/vieveksharmaa/multi-source-rag.git
cd multi-source-rag
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env
uvicorn main:app --host 0.0.0.0 --port 8000
```

Use systemd to run as a background service and Nginx + Certbot for HTTPS.

---

### Platform Comparison

| Platform | Cost | RAM | Persistent Storage | Sleep |
|---|---|---|---|---|
| **HuggingFace Spaces** | Free | 16 GB | ✅ /data | No |
| **Railway** | $5 credit/mo | 512 MB | ❌ | No |
| **Render** | Free | 512 MB | ❌ | After 15 min |
| **Fly.io** | Free (3 VMs) | 256 MB | ✅ 3 GB | Sometimes |
| **VPS (Hetzner)** | ~€4/mo | 2–4 GB | ✅ | No |

---

## 10. Environment Variables Reference

```env
# ── LLM ───────────────────────────────────────────────────────────────────
LLM_PROVIDER=groq              # groq | gemini | openai | anthropic | nvidia | ollama

GROQ_API_KEY=gsk_...           # https://console.groq.com (free)
GROQ_MODEL=llama-3.1-8b-instant

GOOGLE_API_KEY=AIza...         # https://aistudio.google.com (free tier)
GEMINI_MODEL=gemini-1.5-flash

OPENAI_API_KEY=sk-...          # https://platform.openai.com (paid)
OPENAI_MODEL=gpt-4o-mini

ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com
ANTHROPIC_MODEL=claude-3-5-haiku-20241022

NVIDIA_API_KEY=nvapi-...       # https://build.nvidia.com (free, no credit card)
NVIDIA_MODEL=meta/llama-3.1-8b-instruct

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# ── Embeddings ────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER=huggingface  # huggingface | openai | gemini | nvidia | ollama
HUGGINGFACE_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
NVIDIA_EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# ── Vector Store ──────────────────────────────────────────────────────────
VECTOR_STORE=chroma             # chroma | qdrant | pinecone
CHROMA_PERSIST_DIR=./data/chroma_db

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=rag_documents

PINECONE_API_KEY=pcsk_...
PINECONE_INDEX=rag-index
PINECONE_ENVIRONMENT=us-east-1-aws

# ── Web Search ────────────────────────────────────────────────────────────
WEB_SEARCH_PROVIDER=duckduckgo  # duckduckgo | tavily | serpapi
TAVILY_API_KEY=tvly-...
WEB_SEARCH_MAX_RESULTS=5

# ── GitHub ────────────────────────────────────────────────────────────────
GITHUB_TOKEN=ghp_...            # Optional — needed for private repos / high rate limits

# ── App ───────────────────────────────────────────────────────────────────
APP_NAME="Multi-Source RAG"
DEBUG=false
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
TEMPERATURE=0.1
```

---

## 11. API Reference

Interactive docs at `http://localhost:8000/docs`.

### `POST /ingest`
```json
{ "source": "https://www.youtube.com/watch?v=VIDEO_ID" }
```
Response:
```json
{ "status": "ok", "source_type": "youtube", "chunks_added": 23 }
```

### `POST /ingest/file`
Form data: `file` — PDF, DOCX, TXT, MD

### `POST /query`
```json
{ "query": "What is the main argument?" }
```
Response:
```json
{
  "answer": "The main argument is... [1][2]",
  "citations": [
    { "number": 1, "source_type": "youtube", "title": "...", "url": "...", "extra": "@ 5:20" }
  ]
}
```

### `GET /health`
Returns current provider config.

### `GET /config`
Returns LLM provider, model, embedding provider, vector store, chunk settings.

### `DELETE /store/clear`
Wipes all indexed data.

### `GET /debug/retrieve?q=your+query`
Shows raw retrieval results with similarity scores — useful for debugging.

### `GET /debug/store`
Shows all chunks currently in the vector store.

---

## 12. Troubleshooting

**First ingest is slow (30–60 sec)**
The HuggingFace model downloads on first use (~80–130 MB). Cached after that. Docker builds pre-download it so cloud deploys don't have this delay.

**"YouTube transcripts are blocked"**
YouTube blocks cloud provider IPs. Options: use a PDF/URL source instead, or run locally where your home IP isn't blocked.

**Answers about wrong topic / stale data**
Click **Clear All** in the sidebar, then re-ingest your sources.

**"Context does not contain information about..."**
Either ingestion returned 0 chunks (check the response) or the query is too vague. Use `GET /debug/retrieve?q=your+query` to see exactly what gets retrieved.

**ChromaDB error after changing embedding model**
```bash
rm -rf ./data/chroma_db/
```
Then re-ingest.

**Groq rate limit**
```env
GROQ_MODEL=gemma2-9b-it         # different quota pool
# or switch provider entirely
LLM_PROVIDER=anthropic
LLM_PROVIDER=gemini
```

**GitHub clone fails**
The loader tries `main` → `master` → default branch automatically. If it still fails, the repo may be private — set `GITHUB_TOKEN` in `.env`.

**Out of memory on cloud**
The default embedding model needs ~300 MB RAM. Switch to an API-based embedder: `EMBEDDING_PROVIDER=gemini` or `EMBEDDING_PROVIDER=nvidia`.

---

## 13. Tech Stack

| Component | Library |
|---|---|
| LLM orchestration | LangChain |
| Pipeline graph | LangGraph |
| LLM (default) | Groq + Llama 3.1 |
| LLM (also supported) | Anthropic Claude · Google Gemini · OpenAI GPT · NVIDIA NIM · Ollama |
| Embeddings (default) | HuggingFace sentence-transformers |
| Vector store (default) | ChromaDB |
| Web search (default) | DuckDuckGo |
| YouTube transcripts | youtube-transcript-api + yt-dlp + Invidious fallback |
| PDF parsing | PyMuPDF |
| DOCX parsing | docx2txt |
| Web scraping | BeautifulSoup4 + requests |
| Backend API | FastAPI + Uvicorn |
| Containerisation | Docker |
