"""
Streamlit frontend for Multi-Source RAG App.
Run with: streamlit run frontend/app.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import tempfile

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi Source RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.citation-box {
    background: #1e1e2e;
    border-left: 3px solid #7c3aed;
    padding: 10px 16px;
    margin: 6px 0;
    border-radius: 4px;
    font-size: 0.85rem;
}
.source-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
}
.badge-youtube   { background: #ef4444; color: white; }
.badge-pdf       { background: #f59e0b; color: white; }
.badge-docx      { background: #3b82f6; color: white; }
.badge-web_url   { background: #10b981; color: white; }
.badge-web_search{ background: #10b981; color: white; }
.badge-github    { background: #6b7280; color: white; }
.badge-text      { background: #8b5cf6; color: white; }
.answer-box {
    background: #0f0f1a;
    border: 1px solid #2d2d4e;
    border-radius: 8px;
    padding: 20px;
    margin: 10px 0;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configuration + source ingestion
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Multi-Source RAG")
    st.caption("LangChain + LangGraph")

    # Config display
    with st.expander("⚙️ Current Configuration", expanded=False):
        try:
            import config
            st.json({
                "LLM": f"{config.LLM_PROVIDER} / {getattr(config, config.LLM_PROVIDER.upper() + '_MODEL', '?')}",
                "Embeddings": config.EMBEDDING_PROVIDER,
                "Vector Store": config.VECTOR_STORE,
                "Web Search": config.WEB_SEARCH_PROVIDER,
            })
        except Exception as e:
            st.error(f"Config error: {e}")

    st.divider()
    st.subheader("➕ Add Sources")

    # URL / YouTube / GitHub ingestion
    with st.form("ingest_url_form"):
        url_input = st.text_input(
            "URL / YouTube / GitHub",
            placeholder="https://youtube.com/watch?v=... or https://github.com/org/repo"
        )
        submitted_url = st.form_submit_button("📥 Ingest URL", use_container_width=True)

    if submitted_url and url_input:
        with st.spinner("Ingesting..."):
            try:
                from src.sources.ingest import ingest
                result = ingest(url_input)
                st.success(f"✅ {result['chunks_added']} chunks added ({result['source_type']})")
                if "ingested_sources" not in st.session_state:
                    st.session_state.ingested_sources = []
                st.session_state.ingested_sources.append({
                    "type": result["source_type"],
                    "source": url_input,
                    "chunks": result["chunks_added"],
                })
            except Exception as e:
                st.error(f"❌ {str(e)}")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "docx", "txt", "md"],
        help="PDF, Word document, or plain text"
    )
    if uploaded_file:
        if st.button("📥 Ingest File", use_container_width=True):
            with st.spinner("Processing file..."):
                try:
                    ext = os.path.splitext(uploaded_file.name)[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    from src.sources.ingest import ingest
                    result = ingest(tmp_path)
                    os.unlink(tmp_path)

                    st.success(f"✅ {result['chunks_added']} chunks from '{uploaded_file.name}'")
                    if "ingested_sources" not in st.session_state:
                        st.session_state.ingested_sources = []
                    st.session_state.ingested_sources.append({
                        "type": result["source_type"],
                        "source": uploaded_file.name,
                        "chunks": result["chunks_added"],
                    })
                except Exception as e:
                    st.error(f"❌ {str(e)}")

    # Ingested sources list
    if st.session_state.get("ingested_sources"):
        st.divider()
        st.subheader("📚 Ingested Sources")
        for src in st.session_state.ingested_sources:
            badge_color = {
                "youtube": "#ef4444", "pdf": "#f59e0b", "docx": "#3b82f6",
                "web_url": "#10b981", "github": "#6b7280", "text": "#8b5cf6",
            }.get(src["type"], "#6b7280")
            st.markdown(
                f'<span style="background:{badge_color};color:white;padding:2px 8px;'
                f'border-radius:10px;font-size:0.7rem;">{src["type"].upper()}</span> '
                f'`{src["source"][:40]}...` — {src["chunks"]} chunks',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main — Chat interface
# ─────────────────────────────────────────────────────────────────────────────
st.title("💬 Ask Anything")
st.caption("Add sources in the sidebar, then ask questions below.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            _render_citations(msg["citations"]) if "citations" in dir() else None

# Query input
if prompt := st.chat_input("Ask a question about your sources..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                from src.graph.rag_graph import query
                result = query(prompt)
                answer = result["answer"]
                citations = result["citations"]
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"
                citations = []

        st.markdown(answer)

        # Render citations
        if citations:
            st.divider()
            st.markdown("**📎 Sources**")
            for c in citations:
                source_type = c.get("source_type", "unknown")
                badge_color = {
                    "youtube": "#ef4444", "pdf": "#f59e0b", "docx": "#3b82f6",
                    "web_url": "#10b981", "web_search": "#10b981",
                    "github": "#6b7280", "text": "#8b5cf6",
                }.get(source_type, "#6b7280")

                badge = (
                    f'<span style="background:{badge_color};color:white;padding:2px 8px;'
                    f'border-radius:10px;font-size:0.75rem;font-weight:600;">'
                    f'{source_type.upper()}</span>'
                )
                title = c.get("title", "")
                extra = c.get("extra", "")
                url = c.get("url", "")

                link = f' <a href="{url}" target="_blank">🔗</a>' if url else ""
                extra_text = f" <i>{extra}</i>" if extra else ""

                st.markdown(
                    f'<div class="citation-box">'
                    f'<b>[{c["number"]}]</b> {badge} {title}{extra_text}{link}'
                    f'</div>',
                    unsafe_allow_html=True
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
    })
