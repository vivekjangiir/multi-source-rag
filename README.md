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

A production-quality Retrieval-Augmented Generation app that pulls from **multiple source types simultaneously** — YouTube videos, PDFs, web pages, GitHub repos, books, and live web search — and always shows **exactly which source each answer came from**.

Built with **LangChain + LangGraph** for the retrieval pipeline and **Streamlit** for the UI.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-source ingestion** | YouTube, PDF, DOCX, TXT, web URLs, GitHub repos |
| **Live web search** | Auto-triggered for current events / real-time questions |
| **Citation system** | Every answer shows `[1] [YouTube] Title @ 2:34`, `[2] [PDF] doc.pdf — Page 5`, etc. |
| **LangGraph pipeline** | Route → Retrieve (parallel) → Grade → Generate |
| **Relevance grading** | Irrelevant docs are filtered before generation |
| **Switchable providers** | LLM, embeddings, vector store, web search — all swappable via `.env` |
| **Free by default** | Works end-to-end with zero API cost (Groq + HuggingFace + ChromaDB + DuckDuckGo) |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Router   │  ← LLM decides: does this need live web search?
└────────┬────────┘
         │ (parallel)
┌────────▼────────┐   ┌─────────────────┐
│ Vector Retrieval│   │  Web Search     │
│ (ChromaDB etc.) │   │ (DDG/Tavily)    │
└────────┬────────┘   └────────┬────────┘
         └──────────┬──────────┘
                    ▼
         ┌──────────────────┐
         │ Document Grader  │  ← Filters irrelevant docs
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ Answer Generator │  ← Synthesizes answer with inline [1][2] citations
         └──────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourusername/multi-source-rag
cd multi-source-rag
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — minimum needed for free setup:
#   GROQ_API_KEY=your_key   ← get free at https://console.groq.com
```

### 3. Run

```bash
# Option A: Streamlit UI (recommended)
streamlit run frontend/app.py

# Option B: FastAPI backend only
python main.py
# API docs at http://localhost:8000/docs
```

---

## ⚙️ Provider Options

### LLM (`LLM_PROVIDER`)

| Provider | Cost | Speed | Setup |
|---|---|---|---|
| `groq` ✅ **default** | Free (rate limited) | Very fast | Get key at [console.groq.com](https://console.groq.com) |
| `gemini` | Free tier | Fast | Get key at [aistudio.google.com](https://aistudio.google.com) |
| `ollama` | Free (local) | Depends on hardware | Install [ollama.ai](https://ollama.ai), run `ollama pull llama3` |
| `openai` | Paid | Fast | [platform.openai.com](https://platform.openai.com) |

**Switch:** Set `LLM_PROVIDER=gemini` in `.env`

---

### Embeddings (`EMBEDDING_PROVIDER`)

| Provider | Cost | Model | Notes |
|---|---|---|---|
| `huggingface` ✅ **default** | Free (local) | `BAAI/bge-small-en-v1.5` | Runs on CPU, no API key |
| `ollama` | Free (local) | `nomic-embed-text` | Requires Ollama running |
| `gemini` | Free tier | `models/embedding-001` | Needs Google API key |
| `openai` | Paid | `text-embedding-3-small` | Best quality, costs money |

**Switch:** Set `EMBEDDING_PROVIDER=ollama` in `.env`

> **Note:** If you change embedding provider after ingesting documents, you must re-ingest — vectors from different models are incompatible.

---

### Vector Store (`VECTOR_STORE`)

| Provider | Cost | Setup | Best For |
|---|---|---|---|
| `chroma` ✅ **default** | Free | Zero setup, persists locally | Local dev, small-medium datasets |
| `qdrant` | Free (local) / Cloud | `docker run -p 6333:6333 qdrant/qdrant` | Production, better filtering |
| `pinecone` | Free tier (1 index) | Account at [pinecone.io](https://pinecone.io) | Serverless cloud, large scale |

**Switch:** Set `VECTOR_STORE=qdrant` in `.env`

---

### Web Search (`WEB_SEARCH_PROVIDER`)

| Provider | Cost | Limit | Setup |
|---|---|---|---|
| `duckduckgo` ✅ **default** | Free | No limit (rate throttled) | No key needed |
| `tavily` | Free tier | 1,000 searches/month | Get key at [tavily.com](https://tavily.com) |
| `serpapi` | Paid | — | [serpapi.com](https://serpapi.com) |

**Switch:** Set `WEB_SEARCH_PROVIDER=tavily` in `.env`

---

## 📥 Supported Source Types

| Source | How to ingest | Citation format |
|---|---|---|
| **YouTube video** | Paste any `youtube.com` or `youtu.be` URL | `[YouTube] Title @ 2:34 — link` |
| **PDF** | Upload file or provide path | `[PDF] filename.pdf — Page 5` |
| **Word doc** | Upload `.docx` file | `[DOCX] filename.docx` |
| **Plain text / Markdown** | Upload `.txt` or `.md` | `[Text] filename.txt` |
| **Web page** | Paste any `https://` URL | `[Web] Page Title — URL` |
| **GitHub repo** | Paste `https://github.com/org/repo` | `[GitHub] org/repo — src/file.py` |
| **Live web search** | Auto-triggered by the query router | `[Web] Result Title — URL` |

---

## 🗂️ Project Structure

```
multi-source-rag/
├── config.py                   # All settings via env vars
├── main.py                     # FastAPI backend
├── requirements.txt
├── .env.example                # Copy to .env
│
├── src/
│   ├── llm/
│   │   └── provider.py         # LLM + embeddings abstraction
│   ├── vectorstore/
│   │   └── store.py            # Vector store abstraction
│   ├── sources/
│   │   ├── youtube.py          # YouTube transcript loader
│   │   ├── pdf.py              # PDF / DOCX / TXT loaders
│   │   ├── web_search.py       # Web search + URL loader
│   │   ├── github.py           # GitHub repo loader
│   │   └── ingest.py           # Unified ingestion entry point
│   └── graph/
│       ├── state.py            # LangGraph state schema
│       ├── nodes.py            # Graph node functions
│       └── rag_graph.py        # Graph assembly + query()
│
├── frontend/
│   └── app.py                  # Streamlit UI
│
└── data/
    ├── chroma_db/              # ChromaDB persistence (auto-created)
    ├── uploads/                # Uploaded files (auto-created)
    └── knowledge_base/         # Drop files here to pre-index
```

---

## 🔌 API Reference

Start the server: `python main.py`  
Swagger UI: `http://localhost:8000/docs`

### `POST /ingest`
```json
{ "source": "https://www.youtube.com/watch?v=dQw4w9WgXcQ" }
{ "source": "https://github.com/langchain-ai/langchain" }
{ "source": "https://example.com/article" }
```

### `POST /ingest/file`
Multipart file upload (PDF, DOCX, TXT).

### `POST /query`
```json
{ "query": "What is RAG and how does it work?" }
```
Response:
```json
{
  "answer": "RAG (Retrieval-Augmented Generation) is... [1] It works by... [2]",
  "citations": [
    { "number": 1, "text": "[YouTube] LangChain Tutorial @ 5:20", "url": "...", "source_type": "youtube" },
    { "number": 2, "text": "[PDF] rag_paper.pdf — Page 3", "url": "", "source_type": "pdf" }
  ]
}
```

### `GET /health`
Returns current provider configuration.

---

## 💡 Example Workflows

**Chat with a YouTube playlist:**
```
1. Paste each video URL into the sidebar → Ingest
2. Ask: "Summarize the key concepts across all videos"
3. Get timestamped citations back to each video moment
```

**Research a GitHub repo:**
```
1. Paste https://github.com/org/repo → Ingest
2. Ask: "How does authentication work in this codebase?"
3. Get answers with file-path citations
```

**Hybrid research (docs + live web):**
```
1. Upload your PDF research papers
2. Ask: "What are the latest developments in this field?"
3. The router detects "latest" → pulls live web results alongside your PDFs
```

---

## 🔧 Troubleshooting

**Slow first query?** HuggingFace embeddings download the model on first run (~90MB). Subsequent queries are fast.

**`ModuleNotFoundError`?** Run `pip install -r requirements.txt` — some packages are optional based on your chosen providers.

**ChromaDB error after changing embedding model?** Delete `./data/chroma_db/` and re-ingest your sources.

**Groq rate limit?** Switch to `LLM_PROVIDER=gemini` or use `GROQ_MODEL=llama3-70b-8192` for a different quota pool.

---

## 🧰 Tech Stack

- **[LangChain](https://langchain.com)** — document loaders, text splitters, chains
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — stateful RAG pipeline orchestration
- **[ChromaDB](https://www.trychroma.com)** — local vector store
- **[Groq](https://groq.com)** — fast free LLM inference
- **[Streamlit](https://streamlit.io)** — frontend UI
- **[FastAPI](https://fastapi.tiangolo.com)** — REST API backend
