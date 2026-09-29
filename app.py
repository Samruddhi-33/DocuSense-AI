"""DocuSense AI - Streamlit UI.  Run:  streamlit run app.py"""
import json

import streamlit as st

from docusense import config, features
from docusense.library import Library
from docusense.llm import LLMClient
from docusense.rag import answer_question

st.set_page_config(page_title="DocuSense AI", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading document library...")
def get_library() -> Library:
    return Library.load_or_create()


lib = get_library()
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------ Sidebar -----------------------------------
with st.sidebar:
    st.title("📄 DocuSense AI")
    api_key = st.text_input(
        "Anthropic API key", type="password", value=config.ANTHROPIC_API_KEY,
        help="Or set ANTHROPIC_API_KEY in .env",
    )
    llm = LLMClient(api_key=api_key)
    st.caption("🟢 LLM ready" if llm.available else "🟠 No API key - Q&A shows raw passages only")

    st.divider()
    files = st.file_uploader("Upload documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if st.button("Index documents", type="primary", disabled=not files):
        for f in files:
            try:
                with st.spinner(f"Indexing {f.name}..."):
                    n = lib.add_document(f.name, f.getvalue())
                st.success(f"{f.name}: {n} chunks")
            except Exception as exc:  # noqa: BLE001
                st.error(f"{f.name}: {exc}")

    st.divider()
    st.subheader("Library")
    if not lib.documents:
        st.info("No documents yet.")
    for d in lib.documents:
        c1, c2 = st.columns([5, 1])
        c1.write(f"📑 {d}")
        if c2.button("🗑", key=f"del_{d}", help="Remove"):
            lib.remove_document(d)
            st.rerun()

docs = lib.documents
st.header("AI Document Intelligence")
if not docs:
    st.info("Upload and index at least one document from the sidebar to get started.")
    st.stop()

tab_ask, tab_sum, tab_ext, tab_cmp = st.tabs(["💬 Ask", "📝 Summarize", "🔍 Extract", "⚖️ Compare"])

# ------------------------------ Ask ---------------------------------------
with tab_ask:
    scope = st.multiselect("Search in (empty = all documents)", docs)
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Sources"):
                    for i, s in enumerate(m["sources"], 1):
                        st.markdown(f"**[{i}] {s['doc']} - p.{s['page']}** (score {s['score']:.2f})")
                        st.caption(s["text"])
    if q := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        history = [
            (st.session_state.messages[i]["content"], st.session_state.messages[i + 1]["content"])
            for i in range(0, len(st.session_state.messages) - 1, 2)
        ]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    ans = answer_question(lib, llm, q, docs=scope or None, history=history)
                    text, sources = ans.text, [
                        {"doc": c.doc, "page": c.page, "score": s, "text": c.text} for c, s in ans.sources
                    ]
                except Exception as exc:  # noqa: BLE001
                    text, sources = f"⚠️ Error: {exc}", []
            st.markdown(text)
        st.session_state.messages.append({"role": "assistant", "content": text, "sources": sources})
        st.rerun()
    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------ Summarize ---------------------------------
with tab_sum:
    doc = st.selectbox("Document", docs, key="sum_doc")
    style = st.radio("Style", list(features.SUMMARY_STYLES), horizontal=True)
    if st.button("Summarize"):
        if not llm.available:
            st.warning("Add an Anthropic API key in the sidebar.")
        else:
            with st.spinner("Summarizing..."):
                try:
                    st.markdown(features.summarize(lib, llm, doc, style))
                except Exception as exc:  # noqa: BLE001
                    st.error(exc)

# ------------------------------ Extract -----------------------------------
with tab_ext:
    doc = st.selectbox("Document", docs, key="ext_doc")
    extra = st.text_input("Extra fields (comma separated, optional)", placeholder="invoice_number, governing_law")
    if st.button("Extract key information"):
        if not llm.available:
            st.warning("Add an Anthropic API key in the sidebar.")
        else:
            with st.spinner("Extracting..."):
                try:
                    result = features.extract_info(lib, llm, doc, [x.strip() for x in extra.split(",") if x.strip()])
                    st.json(result)
                    st.download_button("Download JSON", json.dumps(result, indent=2), f"{doc}.json", "application/json")
                except Exception as exc:  # noqa: BLE001
                    st.error(exc)

# ------------------------------ Compare -----------------------------------
with tab_cmp:
    picked = st.multiselect("Choose 2 or more documents", docs, key="cmp_docs")
    focus = st.text_input("Focus (optional)", placeholder="e.g. payment terms, termination clauses")
    if st.button("Compare"):
        if len(picked) < 2:
            st.warning("Select at least two documents.")
        elif not llm.available:
            st.warning("Add an Anthropic API key in the sidebar.")
        else:
            with st.spinner("Comparing..."):
                try:
                    st.markdown(features.compare(lib, llm, picked, focus))
                except Exception as exc:  # noqa: BLE001
                    st.error(exc)
