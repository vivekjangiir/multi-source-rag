"""
RAG Pipeline Diagnostic Script
================================
Run: python debug_test.py

Paste a YouTube URL when prompted.
This will test each step independently and show exactly where things break.
"""
import os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # ensure correct working dir

from dotenv import load_dotenv
load_dotenv()

LINE = "─" * 60

def header(title):
    print(f"\n{LINE}\n  {title}\n{LINE}")

# ── Step 0: Config ────────────────────────────────────────────
header("STEP 0 — Config")
import config
print(f"  LLM provider   : {config.LLM_PROVIDER} / {config.GROQ_MODEL}")
print(f"  Embedding      : {config.EMBEDDING_PROVIDER} / {config.HF_EMBEDDING_MODEL}")
print(f"  Vector store   : {config.VECTOR_STORE}")
print(f"  Chroma path    : {os.path.abspath(config.CHROMA_PERSIST_DIR)}")
print(f"  Chroma exists  : {os.path.exists(config.CHROMA_PERSIST_DIR)}")

# ── Step 1: Wipe ChromaDB ─────────────────────────────────────
header("STEP 1 — Wipe ChromaDB (fresh start)")
db_path = os.path.abspath(config.CHROMA_PERSIST_DIR)
if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print(f"  ✓ Deleted {db_path}")
else:
    print(f"  (already empty)")

# ── Step 2: YouTube transcript ────────────────────────────────
header("STEP 2 — YouTube Transcript API")
url = input("\n  Paste a YouTube URL: ").strip()
if not url:
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

video_id = None
import re
for pat in [r"(?:v=|\/)([0-9A-Za-z_-]{11})", r"youtu\.be\/([0-9A-Za-z_-]{11})"]:
    m = re.search(pat, url)
    if m:
        video_id = m.group(1)
        break

print(f"  Video ID: {video_id}")

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)
    snippets = list(transcript)
    print(f"  ✓ Fetched {len(snippets)} transcript snippets")
    print(f"  First snippet: \"{snippets[0].text[:80]}\"  @ {snippets[0].start:.1f}s")
    print(f"  Last snippet:  \"{snippets[-1].text[:80]}\"  @ {snippets[-1].start:.1f}s")
    total_words = sum(len(s.text.split()) for s in snippets)
    print(f"  Total words: ~{total_words}")
except Exception as e:
    print(f"  ✗ TRANSCRIPT ERROR: {e}")
    sys.exit(1)

# ── Step 3: Chunk the transcript ──────────────────────────────
header("STEP 3 — Chunking")
from src.sources.youtube import load_youtube
try:
    docs = load_youtube(url)
    print(f"  ✓ Created {len(docs)} chunks")
    if docs:
        print(f"  First chunk ({len(docs[0].page_content)} chars):")
        print(f"    \"{docs[0].page_content[:150]}\"")
        print(f"  Metadata: {docs[0].metadata}")
except Exception as e:
    print(f"  ✗ CHUNKING ERROR: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Step 4: Embeddings ────────────────────────────────────────
header("STEP 4 — Embeddings")
try:
    from src.llm.provider import get_embeddings
    embeddings = get_embeddings()
    test_vec = embeddings.embed_query("test")
    print(f"  ✓ Embedding model loaded, dimension={len(test_vec)}")
except Exception as e:
    print(f"  ✗ EMBEDDING ERROR: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Step 5: Store in ChromaDB ─────────────────────────────────
header("STEP 5 — Store in ChromaDB")
try:
    from src.vectorstore.store import add_documents, get_vector_store
    ids = add_documents(docs)
    print(f"  ✓ Stored {len(ids)} chunks in ChromaDB")
    store = get_vector_store()
    if hasattr(store, '_collection'):
        total = store._collection.count()
        print(f"  ChromaDB total docs now: {total}")
except Exception as e:
    print(f"  ✗ STORE ERROR: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Step 6: Retrieval ─────────────────────────────────────────
header("STEP 6 — Retrieval (similarity search)")
test_queries = [
    "What is this video about?",
    "What are the main topics discussed?",
    "Summarize the key points",
]
try:
    from src.vectorstore.store import similarity_search_with_score
    for q in test_queries:
        results = similarity_search_with_score(q, k=3)
        print(f"\n  Query: \"{q}\"")
        for doc, score in results:
            src = doc.metadata.get("source_type", "?")
            ts  = doc.metadata.get("timestamp", "")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"    score={score:.3f} [{src}] {ts}  \"{preview}\"")
except Exception as e:
    print(f"  ✗ RETRIEVAL ERROR: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Step 7: LLM ───────────────────────────────────────────────
header("STEP 7 — LLM (single call test)")
try:
    from src.llm.provider import get_llm
    from langchain_core.messages import HumanMessage
    llm = get_llm()
    resp = llm.invoke([HumanMessage(content="Reply with exactly: OK")])
    print(f"  ✓ LLM responded: \"{resp.content.strip()}\"")
except Exception as e:
    print(f"  ✗ LLM ERROR: {e}")
    import traceback; traceback.print_exc()

# ── Step 8: Full query ────────────────────────────────────────
header("STEP 8 — Full RAG Query")
try:
    from src.graph.rag_graph import query, _compiled_graph
    import src.graph.rag_graph as rag
    rag._compiled_graph = None  # force rebuild

    result = query("What is this video about? Give a brief summary.")
    print(f"\n  ANSWER:\n{result['answer']}\n")
    print(f"  CITATIONS ({len(result['citations'])}):")
    for c in result['citations']:
        print(f"    [{c['number']}] {c['text']}")
except Exception as e:
    print(f"  ✗ QUERY ERROR: {e}")
    import traceback; traceback.print_exc()

header("DIAGNOSTIC COMPLETE")
print("  If Step 6 showed score < 0.3 for all results, the embedding")
print("  model may need warming up — try running again.")
print("  If Step 5 stored 0 chunks, the YouTube loader failed.")
print()
