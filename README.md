---
title: Multi Source RAG with Citations
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🔍 Multi-Source RAG with Citations

> A production-quality Retrieval-Augmented Generation app that ingests **YouTube videos, PDFs, Word docs, web pages, GitHub repos, and live web search** — and always shows **exactly which source each answer came from**, with timestamps and page numbers.

Built with **LangChain + LangGraph**. Every provider (LLM, embeddings, vector store, web search) is swappable via a single `.env` file. **Runs entirely free by default.**

---

## Table of Contents

1. [What Is RAG and Why Does This App Exist?](#1-what-is-rag-and-why-does-this-app-exist)
2. [Architecture](#2-architecture)
3. [Local Setup](#3-local-setup)
4. [LLM Providers](#4-llm-providers)
5. [Embedding Providers](#5-embedding-providers)
6. [Vector Stores](#6-vector-stores)
7. [Web Search Providers](#7-web-search-providers)
8. [Source Types](#8-source-types)
9. [Deployment: All Platforms](#9-deployment-all-platforms)
   - [Hugging Face Spaces](#91-hugging-face-spaces-recommended-free)
   - [Railway](#92-railway)
   - [Render](#93-render)
   - [Fly.io](#94-flyio)
   - [Self-hosted VPS](#95-self-hosted-vps-ubuntu)
10. [Environment Variables Reference](#10-environment-variables-reference)
11. [API Reference](#11-api-reference)
12. [Troubleshooting](#12-troubleshooting)
13. [Tech Stack](#13-tech-stack)

---

## 1. What Is RAG and Why Does This App Exist?

**RAG (Retrieval-Augmented Generation)** is the technique of grounding an LLM's answer in real documents rather than its training data alone. Instead of hallucinating, the model reads the relevant passages first, then answers from them.

This matters because:

- LLMs cut off at a training date — they don't know about your documents or recent events.
- LLMs hallucinate confidently. Grounding them in real sources reduces this.
- Citations let users verify every claim — critical for research, legal, and professional use.

**Why multi-source?** Most RAG apps handle one source type. This app handles six simultaneously:

| Source | Why it matters |
|---|---|
| YouTube | Massive knowledge base; transcripts are free |
| PDF / DOCX | Research papers, reports, books, manuals |
| Web URLs | Articles, documentation, any public page |
| GitHub repos | Understand codebases by asking questions |
| Live web search | Real-time information — news, current prices, latest releases |
| Plain text / Markdown | Notes, logs, any raw text |

The LangGraph pipeline routes each query through the right combination of these sources automatically.

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

**Ingestion pipeline** (separate from query):

```
Source URL / File
       │
  Source Loader         (YouTube transcript / PDF parser / web scraper / GitHub walker)
       │
  Text Splitter         (RecursiveCharacterTextSplitter, 1000 chars, 200 overlap)
       │
  Embeddings            (HuggingFace / OpenAI / Ollama / Gemini)
       │
  Vector Store          (ChromaDB / Qdrant / Pinecone)
```

---

## 3. Local Setup

### Prerequisites

- Python 3.11 (required — some dependencies don't support 3.12/3.13 yet)
- `git`
- A free [Groq API key](https://console.groq.com) (takes 60 seconds to get)

### Step 1 — Clone

```bash
git clone https://github.com/YOUR_USERNAME/multi-source-rag.git
cd multi-source-rag
```

### Step 2 — Create a virtual environment

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

Open `.env` and set at minimum:

```env
GROQ_API_KEY=your_key_here   # Free at https://console.groq.com
```

Everything else has a working free default — you don't need to change anything else to get started.

### Step 5 — Run

```bash
python main.py
```

Open **http://localhost:8000** in your browser.

The first time you ingest a source, the HuggingFace embedding model (~80 MB) downloads automatically. This takes 30–60 seconds once, then it's cached.

### Step 6 — Test it

1. Paste any YouTube URL into the sidebar → click **Ingest**
2. Wait for "Ingested successfully"
3. Type a question about the video → press **Send**
4. The answer appears with `[1] [YouTube] Title @ 2:34` citations

---

## 4. LLM Providers

The LLM is used for two things: routing queries (should we search the web?) and generating the final answer.

### How to switch

Set `LLM_PROVIDER=` in your `.env` file.

---

### 4.1 Groq ✅ Default (recommended)

**What:** Groq runs open-source models (Llama, Gemma) on custom LPU hardware — inference is 10–20× faster than a GPU server.

**Why use it:**
- Free tier with generous rate limits
- Fastest inference of any free provider (~500 tokens/sec)
- No local GPU needed
- Llama 3.1 is high quality

**Why not:** Rate limited (varies by model); no fine-tuning.

**Setup:**
1. Go to [console.groq.com](https://console.groq.com) → create account → API Keys → Create Key
2. Set in `.env`:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.1-8b-instant   # fast, good quality
# or: llama-3.3-70b-versatile     # slower, best quality
# or: gemma2-9b-it                # Google's Gemma 2
```

**Models comparison:**

| Model | Speed | Quality | Rate limit |
|---|---|---|---|
| `llama-3.1-8b-instant` | ⚡⚡⚡ | ⭐⭐⭐ | 30k tokens/min |
| `llama-3.3-70b-versatile` | ⚡⚡ | ⭐⭐⭐⭐⭐ | 6k tokens/min |
| `gemma2-9b-it` | ⚡⚡⚡ | ⭐⭐⭐ | 15k tokens/min |

---

### 4.2 Google Gemini

**What:** Google's multimodal model family. The free tier is more generous than OpenAI's.

**Why use it:**
- Free tier: 15 requests/min, 1M tokens/day
- Gemini 1.5 Flash is fast and capable
- Good for longer context windows

**Why not:** Google account required; responses can be over-cautious.

**Setup:**
1. Go to [aistudio.google.com](https://aistudio.google.com) → Get API Key
2. Set in `.env`:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=AIza...
GEMINI_MODEL=gemini-1.5-flash   # fast, free
# or: gemini-1.5-pro            # more capable, stricter rate limits
```

Also uncomment in `requirements.txt`:
```
langchain-google-genai>=2.0.0
```

---

### 4.3 OpenAI

**What:** GPT-4o, GPT-4o-mini, GPT-3.5-turbo. The industry standard.

**Why use it:**
- Best overall quality (GPT-4o)
- Most reliable API
- Rich ecosystem

**Why not:** Paid only. GPT-4o costs ~$5 per 1M input tokens.

**Setup:**
1. Go to [platform.openai.com](https://platform.openai.com) → API Keys
2. Set in `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini   # cheap, good
# or: gpt-4o               # best quality
```

Also uncomment in `requirements.txt`:
```
langchain-openai>=0.2.0
```

---

### 4.4 Ollama (Local)

**What:** Run open-source models entirely on your own machine. No API key, no cost, no data leaves your computer.

**Why use it:**
- 100% private — no data sent to any server
- No rate limits
- Works offline
- Free forever

**Why not:** Requires a capable GPU (or slow on CPU); not suitable for cloud deployment.

**Setup:**
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model:
   ```bash
   ollama pull llama3        # 4.7GB — recommended
   ollama pull mistral       # 4.1GB — fast
   ollama pull phi3          # 2.3GB — lightweight
   ```
3. Set in `.env`:
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3
   ```

Also uncomment in `requirements.txt`:
```
langchain-ollama>=0.2.0
```

---

## 5. Embedding Providers

Embeddings convert text into vectors (lists of numbers that capture meaning). The embedding model determines how well your vector search works — a better model finds more relevant chunks.

> ⚠️ **Important:** If you change embedding provider after ingesting documents, you must clear your vector store and re-ingest. Vectors from different models are mathematically incompatible.

### How to switch

Set `EMBEDDING_PROVIDER=` in your `.env` file.

---

### 5.1 HuggingFace (Local) ✅ Default (recommended for free use)

**What:** Downloads and runs a small embedding model locally using `sentence-transformers`. No API key, no cost.

**Why use it:**
- Completely free, no API key
- Runs on CPU (no GPU needed)
- `all-MiniLM-L6-v2` is surprisingly good (~84% on BEIR benchmark)
- Model is cached after first download

**Why not:** ~80 MB download on first run; slightly slower than API-based embeddings.

**Setup:**

```env
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**Model options:**

| Model | Size | Speed | Quality | Best for |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 80 MB | ⚡⚡⚡ | ⭐⭐⭐ | General use, cloud deploy |
| `all-mpnet-base-v2` | 420 MB | ⚡⚡ | ⭐⭐⭐⭐ | Better quality, local only |
| `BAAI/bge-small-en-v1.5` | 130 MB | ⚡⚡⚡ | ⭐⭐⭐⭐ | Best small model |
| `BAAI/bge-large-en-v1.5` | 1.3 GB | ⚡ | ⭐⭐⭐⭐⭐ | Best quality, GPU recommended |

---

### 5.2 OpenAI Embeddings

**What:** `text-embedding-3-small` or `text-embedding-3-large` via the OpenAI API.

**Why use it:**
- Best quality embeddings available commercially
- Very fast (API call, not local compute)
- `text-embedding-3-small` costs ~$0.02 per 1M tokens (very cheap)

**Why not:** Paid; data sent to OpenAI.

**Setup:**

```env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

---

### 5.3 Ollama Embeddings (Local)

**What:** Uses a locally-running Ollama model for embeddings. Good companion if you're already using Ollama for the LLM.

**Why use it:** Everything local, private, free.

**Setup:**

```bash
ollama pull nomic-embed-text   # best local embedding model
```

```env
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

---

### 5.4 Google Gemini Embeddings

**What:** `models/embedding-001` via the Gemini API. Free tier.

**Why use it:** Free, good quality, pairs well with Gemini LLM.

**Setup:**

```env
EMBEDDING_PROVIDER=gemini
GOOGLE_API_KEY=AIza...
```

---

## 6. Vector Stores

The vector store holds your embedded document chunks and enables fast similarity search. When you ask a question, it finds the most semantically similar chunks to pass to the LLM.

### How to switch

Set `VECTOR_STORE=` in your `.env` file.

> ⚠️ After switching vector stores, you must re-ingest all your documents into the new store.

---

### 6.1 ChromaDB ✅ Default (recommended to start)

**What:** An open-source, embedded vector database. Runs in the same process as the app — no separate server needed. Stores data to disk automatically.

**Why use it:**
- Zero setup — just works
- Fast enough for thousands of documents
- Persists to disk between restarts
- Open source, free forever

**Why not:** Not suitable for multi-process or distributed deployments; performance degrades beyond ~100K vectors.

**Setup:**

```env
VECTOR_STORE=chroma
CHROMA_PERSIST_DIR=./data/chroma_db
```

Data is stored at `./data/chroma_db/`. To reset: delete that folder and re-ingest.

---

### 6.2 Qdrant

**What:** A purpose-built, high-performance vector database with filtering, payload storage, and a REST API.

**Why use it:**
- Significantly faster than ChromaDB for large collections
- Excellent filtering (e.g., "only search YouTube sources")
- Can run locally via Docker or on Qdrant Cloud (free tier: 1GB)
- Production-ready

**Why not:** Requires a separate Docker container or cloud account.

**Local setup:**

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

```env
VECTOR_STORE=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_documents
```

**Qdrant Cloud setup (free 1GB tier):**

1. Create account at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a cluster → copy URL and API key
3. Set in `.env`:
   ```env
   QDRANT_URL=https://your-cluster.qdrant.io
   QDRANT_API_KEY=your_key
   ```

Also uncomment in `requirements.txt`:
```
langchain-qdrant>=0.1.0
qdrant-client>=1.9.0
```

---

### 6.3 Pinecone

**What:** A fully managed, serverless vector database in the cloud. No infrastructure to run.

**Why use it:**
- Zero infrastructure management
- Scales to billions of vectors
- Free tier: 1 index, 2GB storage
- Best for production apps with high query volume

**Why not:** Paid beyond free tier; data stored on Pinecone's servers; higher latency than local options.

**Setup:**

1. Create account at [pinecone.io](https://pinecone.io)
2. Create an index with dimension `384` (for `all-MiniLM-L6-v2`)
3. Set in `.env`:
   ```env
   VECTOR_STORE=pinecone
   PINECONE_API_KEY=pcsk_...
   PINECONE_INDEX=rag-index
   PINECONE_ENVIRONMENT=us-east-1-aws
   ```

Also uncomment in `requirements.txt`:
```
langchain-pinecone>=0.2.0
pinecone-client>=4.0.0
```

**Dimension guide** — must match your embedding model:

| Embedding model | Dimension |
|---|---|
| `all-MiniLM-L6-v2` | 384 |
| `all-mpnet-base-v2` | 768 |
| `BAAI/bge-small-en-v1.5` | 384 |
| `text-embedding-3-small` | 1536 |
| `nomic-embed-text` | 768 |

---

## 7. Web Search Providers

Web search is used when the query router decides the question needs live / current information (news, recent events, real-time data). It supplements the vector store results.

### How to switch

Set `WEB_SEARCH_PROVIDER=` in your `.env` file.

---

### 7.1 DuckDuckGo ✅ Default (recommended, no key needed)

**What:** DuckDuckGo's unofficial search API. No account, no key, completely free.

**Why use it:** Zero setup. Works immediately.

**Why not:** Rate-limited if you make many searches quickly; occasionally returns fewer results than Google.

**Setup:** Nothing to configure. It's the default.

```env
WEB_SEARCH_PROVIDER=duckduckgo
WEB_SEARCH_MAX_RESULTS=5
```

---

### 7.2 Tavily

**What:** An AI-focused search API built specifically for RAG systems. Returns clean, structured results optimized for LLM consumption.

**Why use it:**
- Returns cleaner results than DuckDuckGo (less HTML noise)
- Designed for RAG — better source quality
- Free tier: 1,000 searches/month
- More reliable than the DuckDuckGo unofficial API

**Why not:** Requires signup; limited free tier.

**Setup:**

1. Create account at [tavily.com](https://tavily.com) → get API key
2. Set in `.env`:
   ```env
   WEB_SEARCH_PROVIDER=tavily
   TAVILY_API_KEY=tvly-...
   ```

Also uncomment in `requirements.txt`:
```
tavily-python>=0.3.0
```

---

### 7.3 SerpAPI

**What:** Google Search results via API. Paid only.

**Why use it:** Most accurate and comprehensive search results (it's real Google).

**Why not:** No free tier; $50+/month for meaningful usage.

**Setup:**

```env
WEB_SEARCH_PROVIDER=serpapi
SERPAPI_API_KEY=your_key
```

---

## 8. Source Types

### YouTube Videos

Pulls the auto-generated or human transcript from any YouTube video and chunks it with timestamps.

**Works with:** Any video that has captions enabled (most videos do)

**Doesn't work with:** Private videos, age-restricted videos without login, videos with transcripts disabled

**Ingest:** Paste the full YouTube URL: `https://www.youtube.com/watch?v=VIDEO_ID`

**Citation format:** `[YouTube] "Video Title" @ 12:34 — https://youtu.be/VIDEO_ID?t=754`

**Note:** On cloud platforms (HuggingFace, Render, etc.) YouTube may block transcript requests from cloud IP ranges. If this happens, try a different video or use a PDF/URL instead.

---

### PDF Documents

Extracts text from any PDF using PyMuPDF. Handles scanned PDFs (with embedded text layers), multi-column layouts, and tables.

**Ingest:** Upload via the file upload button, or provide a local file path in the source field.

**Citation format:** `[PDF] "filename.pdf" — Page 5`

**Tip:** Academic papers work exceptionally well. Upload a PDF and ask "summarize the key findings."

---

### Word Documents (DOCX)

Extracts plain text from `.docx` files using `docx2txt`.

**Ingest:** Upload via the file upload button.

**Citation format:** `[DOCX] "filename.docx"`

---

### Web URLs

Fetches and parses any public webpage. Removes navigation, ads, and boilerplate — keeps the main content.

**Ingest:** Paste any `https://` URL into the source field.

**Citation format:** `[Web] "Page Title" — https://example.com/article`

**Works great for:** Documentation pages, blog posts, news articles, Wikipedia.

---

### GitHub Repositories

Walks the repository file tree and ingests source files with their file paths as citations.

**Ingest:** Paste `https://github.com/org/repo` into the source field.

**Citation format:** `[GitHub] "org/repo" — src/components/Auth.tsx`

**Tip:** Ask "How does authentication work?" or "Where is the database connection configured?" and get file-path citations.

**Rate limiting:** GitHub allows 60 requests/hour unauthenticated. For large repos, set `GITHUB_TOKEN` in your `.env`:
```env
GITHUB_TOKEN=ghp_...   # Personal access token from github.com/settings/tokens
```

---

### Live Web Search

Automatically triggered by the query router when it detects time-sensitive keywords ("latest", "current", "today", "news", "2024", "now"). No manual ingestion needed.

**Citation format:** `[Web Search] "Result Title" — https://source.com`

---

## 9. Deployment: All Platforms

### 9.1 Hugging Face Spaces ✅ Recommended (free)

**What:** HuggingFace Spaces is a free hosting platform for ML demos. Docker spaces give you a full container environment with up to 16GB RAM — plenty for embedding models.

**Why choose this:**
- Free forever (CPU Basic tier)
- 16GB RAM — no out-of-memory errors during embedding
- Persistent `/data` volume — ChromaDB survives restarts
- No sleep on public spaces
- "Deployed on HuggingFace" looks great on a portfolio
- Designed for AI/ML apps — knows about model downloads

**Steps:**

1. Push your code to GitHub (if not done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/multi-source-rag.git
   git push -u origin main
   ```

2. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space):
   - Name: `multi-source-rag`
   - SDK: **Docker**
   - Hardware: **CPU Basic** (free)
   - Visibility: **Public**

3. Add HuggingFace as a git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/multi-source-rag
   git push hf main --force
   ```

4. Add your secret API key:
   - Space → **Settings** → **Variables and Secrets** → **New secret**
   - Key: `GROQ_API_KEY`, Value: your key

5. Your app is live at:
   `https://YOUR_HF_USERNAME-multi-source-rag.hf.space`

**Build time:** ~3–5 minutes (downloads model at build time, not runtime).

**Redeploy after code changes:**
```bash
git add .
git commit -m "Update"
git push hf main
```

---

### 9.2 Railway

**What:** A simple PaaS that deploys directly from GitHub. Similar to Heroku but modern. $5 free credit/month (~500 hours).

**Why choose this:**
- Easiest migration from Render
- No sleep (unlike Render free tier)
- Auto-deploy on every git push
- Good logs and metrics dashboard

**Steps:**

1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo

2. Select your repo → Railway auto-detects Python and uses `requirements.txt`

3. Add environment variables:
   - Settings → Variables → Add all variables from your `.env`

4. Set start command:
   - Settings → Deploy → Start Command:
   ```
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

5. Your app deploys automatically.

**Note:** Railway's free tier has 500 hours/month. Keep usage within limits to stay free.

---

### 9.3 Render

**What:** Render is a cloud platform with a free web service tier. The free tier sleeps after 15 minutes of inactivity.

**Why choose this:**
- Simple GitHub integration
- Free tier available
- Good for low-traffic apps

**Why it's tricky for this app:**
- Free tier only has 512MB RAM — the HuggingFace embedding model can exceed this
- Ephemeral filesystem — ChromaDB resets on every restart/sleep
- Cold starts take 30+ seconds on free tier

**Steps:**

1. Ensure `runtime.txt` contains `python-3.11.9` (already done)

2. Go to [render.com](https://render.com) → New → Web Service

3. Connect your GitHub repo

4. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

5. Add environment variables (Dashboard → Environment):
   - `GROQ_API_KEY` = your key
   - All other vars from `.env`

6. Click **Create Web Service**

**Recommendation:** Use HuggingFace Spaces instead — same free tier cost ($0) but 32× more RAM.

---

### 9.4 Fly.io

**What:** Container hosting with a generous free tier. Docker-based, meaning you deploy the same `Dockerfile` used for HuggingFace.

**Why choose this:**
- Free tier: 3 shared-CPU VMs + 3GB persistent volume
- Your data persists (unlike Render)
- CLI-driven deployment
- Global edge deployment

**Steps:**

1. Install Fly CLI:
   ```bash
   # macOS
   brew install flyctl
   # Windows
   iwr https://fly.io/install.ps1 -useb | iex
   ```

2. Login and launch:
   ```bash
   fly auth login
   fly launch   # auto-detects Dockerfile, asks for app name and region
   ```

3. Set secrets:
   ```bash
   fly secrets set GROQ_API_KEY=gsk_...
   ```

4. Attach persistent storage:
   ```bash
   fly volumes create rag_data --size 1   # 1GB volume, free
   ```

5. Deploy:
   ```bash
   fly deploy
   ```

**Redeploy:**
```bash
fly deploy
```

---

### 9.5 Self-Hosted VPS (Ubuntu)

**What:** Run the app on your own server — DigitalOcean, Hetzner, Linode, AWS EC2, etc.

**Why choose this:**
- Full control over resources and data
- No sleep, no cold starts
- Can use larger embedding models (GPU instances)
- Cheapest per-compute at scale (Hetzner VPS starts at €3.29/month)

**Steps:**

1. SSH into your server and install dependencies:
   ```bash
   sudo apt update && sudo apt install -y python3.11 python3.11-venv git
   ```

2. Clone your repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/multi-source-rag.git
   cd multi-source-rag
   ```

3. Set up environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   nano .env   # set your API keys
   ```

4. Run as a background service with systemd:
   ```bash
   sudo nano /etc/systemd/system/rag-app.service
   ```

   ```ini
   [Unit]
   Description=Multi-Source RAG App
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/multi-source-rag
   EnvironmentFile=/home/ubuntu/multi-source-rag/.env
   ExecStart=/home/ubuntu/multi-source-rag/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   ```bash
   sudo systemctl enable rag-app
   sudo systemctl start rag-app
   ```

5. Set up Nginx as a reverse proxy (optional but recommended):
   ```bash
   sudo apt install -y nginx
   sudo nano /etc/nginx/sites-available/rag-app
   ```

   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           client_max_body_size 50M;
       }
   }
   ```

   ```bash
   sudo ln -s /etc/nginx/sites-available/rag-app /etc/nginx/sites-enabled/
   sudo nginx -t && sudo systemctl restart nginx
   ```

6. Add HTTPS with Let's Encrypt:
   ```bash
   sudo apt install -y certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

---

### Platform Comparison

| Platform | Cost | RAM | Persistent Storage | Sleep | Best For |
|---|---|---|---|---|---|
| **HuggingFace Spaces** | Free | 16 GB | ✅ Yes (/data) | No | ✅ Portfolio / demos |
| **Railway** | $5 credit/mo | 512 MB | ❌ Ephemeral | No | Quick deploy |
| **Render** | Free | 512 MB | ❌ Ephemeral | After 15 min | Low traffic |
| **Fly.io** | Free (3 VMs) | 256 MB | ✅ 3 GB volume | Sometimes | CLI power users |
| **VPS (Hetzner)** | ~€4/mo | 2–4 GB | ✅ Full disk | No | Production |

---

## 10. Environment Variables Reference

Copy `.env.example` to `.env` and set these values.

```env
# ── LLM ───────────────────────────────────────────────────────────────────
LLM_PROVIDER=groq              # groq | gemini | openai | ollama
GROQ_API_KEY=gsk_...           # https://console.groq.com
GROQ_MODEL=llama-3.1-8b-instant

GOOGLE_API_KEY=AIza...         # https://aistudio.google.com
GEMINI_MODEL=gemini-1.5-flash

OPENAI_API_KEY=sk-...          # https://platform.openai.com
OPENAI_MODEL=gpt-4o-mini

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# ── Embeddings ────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER=huggingface  # huggingface | openai | ollama | gemini
HUGGINGFACE_EMBEDDING_MODEL=all-MiniLM-L6-v2
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# ── Vector Store ──────────────────────────────────────────────────────────
VECTOR_STORE=chroma             # chroma | qdrant | pinecone
CHROMA_PERSIST_DIR=./data/chroma_db

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                 # Only for Qdrant Cloud
QDRANT_COLLECTION=rag_documents

PINECONE_API_KEY=pcsk_...
PINECONE_INDEX=rag-index
PINECONE_ENVIRONMENT=us-east-1-aws

# ── Web Search ────────────────────────────────────────────────────────────
WEB_SEARCH_PROVIDER=duckduckgo  # duckduckgo | tavily | serpapi
TAVILY_API_KEY=tvly-...
WEB_SEARCH_MAX_RESULTS=5

# ── GitHub ────────────────────────────────────────────────────────────────
GITHUB_TOKEN=ghp_...            # Optional — increases rate limit

# ── App Settings ──────────────────────────────────────────────────────────
APP_NAME="Multi-Source RAG"
DEBUG=false
MAX_CHUNK_SIZE=1000             # Characters per chunk
CHUNK_OVERLAP=200               # Overlap between chunks
TOP_K_RESULTS=5                 # Docs to retrieve per query
TEMPERATURE=0.1                 # LLM temperature (0 = deterministic)
```

---

## 11. API Reference

The FastAPI backend exposes these endpoints. When running locally, interactive docs are at `http://localhost:8000/docs`.

### `POST /ingest`

Ingest a URL-based source.

**Request:**
```json
{ "source": "https://www.youtube.com/watch?v=VIDEO_ID" }
{ "source": "https://github.com/langchain-ai/langchain" }
{ "source": "https://example.com/article" }
```

**Response:**
```json
{
  "status": "success",
  "message": "Ingested 23 chunks from YouTube: 'Video Title'",
  "chunks": 23
}
```

---

### `POST /ingest/file`

Upload a file (PDF, DOCX, TXT, MD).

**Form data:**
- `file` — the file to upload

**Response:**
```json
{
  "status": "success",
  "message": "Ingested 45 chunks from PDF: 'document.pdf'",
  "chunks": 45
}
```

---

### `POST /query`

Ask a question. Returns an answer with citations.

**Request:**
```json
{ "query": "What is the main argument in the paper?" }
```

**Response:**
```json
{
  "answer": "The main argument is that RAG reduces hallucination by [1] grounding responses in retrieved documents [2].",
  "citations": [
    {
      "number": 1,
      "text": "[YouTube] 'LangChain RAG Tutorial' @ 5:20",
      "url": "https://youtube.com/watch?v=...&t=320s",
      "source_type": "youtube"
    },
    {
      "number": 2,
      "text": "[PDF] 'rag_survey.pdf' — Page 3",
      "url": "",
      "source_type": "pdf"
    }
  ]
}
```

---

### `GET /health`

Returns current configuration and status.

```json
{
  "status": "ok",
  "llm_provider": "groq",
  "embedding_provider": "huggingface",
  "vector_store": "chroma"
}
```

---

### `GET /sources`

Lists all ingested source types.

```json
{ "sources": ["youtube", "pdf", "web"] }
```

---

### `DELETE /store/clear`

Deletes all ingested documents and resets the vector store.

```json
{ "status": "cleared" }
```

---

### `GET /debug/store`

Shows how many documents are in the vector store.

```json
{ "count": 142, "store": "chroma" }
```

---

### `GET /debug/retrieve?q=your+query`

Tests retrieval directly — bypasses the LLM and shows raw retrieved chunks.

```json
{
  "query": "your query",
  "results": [
    {
      "content": "chunk text...",
      "source_type": "youtube",
      "citation": "[YouTube] 'Title' @ 2:10",
      "score": 0.42
    }
  ]
}
```

---

## 12. Troubleshooting

**Q: First ingestion is slow (30–60 seconds)**
The HuggingFace embedding model downloads on first use (~80MB). It's cached after that. On cloud platforms with Docker, it's pre-downloaded at build time so this delay doesn't happen.

---

**Q: "Failed to fetch transcript" on YouTube**
YouTube blocks transcript requests from cloud provider IP ranges (AWS, GCP, HuggingFace infrastructure). Options:
- Try a different video (some are more permissive)
- Use a PDF or web URL instead for cloud deployments
- Run locally where your home IP isn't blocked

---

**Q: Answers are about the wrong topic / old data appears**
Your vector store has stale data from a previous session. Fix: click **"Clear All"** in the sidebar to wipe the store, then re-ingest your sources.

---

**Q: "The context does not contain information about..."**
Two possible causes:
1. The ingestion failed silently — check the ingestion response for chunk count. If it says 0 chunks, ingestion failed.
2. The query is too vague — try more specific keywords that appear in the source text.

Use `GET /debug/retrieve?q=your+query` to see exactly what the retriever finds.

---

**Q: ChromaDB error after changing embedding model**
Delete `./data/chroma_db/` and re-ingest. Vectors from different models can't be mixed.

```bash
rm -rf ./data/chroma_db/
```

---

**Q: Groq rate limit error**
Switch to a different model or provider temporarily:
```env
GROQ_MODEL=gemma2-9b-it        # Different quota pool
# or
LLM_PROVIDER=gemini            # Google's free tier
```

---

**Q: Out of memory on cloud platform**
The `all-MiniLM-L6-v2` model needs ~300MB RAM during inference. Render's free tier (512MB) can hit this. Switch to HuggingFace Spaces (16GB RAM) or use `EMBEDDING_PROVIDER=gemini` (API-based, no local memory).

---

**Q: Port already in use locally**
```bash
# Find what's using port 8000
lsof -i :8000        # macOS/Linux
netstat -ano | findstr :8000   # Windows

# Kill it or change the port:
uvicorn main:app --port 8001
```

---

## 13. Tech Stack

| Component | Library | Why |
|---|---|---|
| **LLM orchestration** | LangChain | Standardized interface across all LLM providers |
| **Pipeline graph** | LangGraph | Stateful, cyclical RAG pipeline with parallel retrieval |
| **LLM (default)** | Groq + Llama 3.1 | Fastest free inference; no GPU needed |
| **Embeddings (default)** | HuggingFace sentence-transformers | Free, local, no API key |
| **Vector store (default)** | ChromaDB | Zero-config local persistence |
| **Web search (default)** | DuckDuckGo | No API key, no rate limit signup |
| **YouTube transcripts** | youtube-transcript-api | No YouTube API key needed |
| **PDF parsing** | PyMuPDF (fitz) | Handles scanned/complex PDFs better than pdfplumber |
| **DOCX parsing** | docx2txt | Lightweight, reliable |
| **Web scraping** | BeautifulSoup4 + requests | Standard, well-maintained |
| **Backend API** | FastAPI | Async, fast, auto-generates OpenAPI docs |
| **ASGI server** | Uvicorn | Production-grade Python web server |
| **Containerisation** | Docker | Consistent environment across all deployment platforms |
