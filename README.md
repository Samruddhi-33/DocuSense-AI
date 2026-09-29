# 📄 DocuSense AI - AI Document Intelligence

Upload PDF / Word documents, ask questions in plain English, summarize, extract
structured information, and compare multiple documents - all grounded in your files (RAG).

## Features
- **Upload** PDF, DOCX, TXT, MD (multiple files)
- **Ask** natural-language questions with page-level citations and follow-up memory
- **Summarize** (short / detailed / bullets) - map-reduce handles long documents
- **Extract** key info as JSON (parties, dates, amounts, deadlines, risks + custom fields)
- **Compare** 2+ documents (similarities, differences table, conflicts), optionally by topic
- Persistent index on disk (`data/`), delete documents anytime

## Architecture
```
Upload -> loader (pypdf / python-docx) -> chunker (sentence-aware, overlap)
       -> embeddings (sentence-transformers MiniLM) -> FAISS (cosine / IndexFlatIP)
Question -> embed -> top-k chunks -> prompt + context -> Claude -> answer + citations
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # add your ANTHROPIC_API_KEY
streamlit run app.py
```
The first run downloads the embedding model (~90 MB). Without an API key, Ask shows the
most relevant passages (retrieval only); other tabs need the key.

## Project layout
```
app.py                  Streamlit UI
docusense/
  config.py             settings from .env
  loader.py             PDF/DOCX/TXT -> pages
  chunker.py            text splitting
  vector_store.py       FAISS + embeddings + persistence
  library.py            document registry (add/remove/save/load)
  llm.py                Claude API wrapper
  rag.py                retrieval + grounded answering
  features.py           summarize / extract / compare
tests/test_core.py      unit tests  (run: pytest)
```

## Notes & ideas
- Scanned (image-only) PDFs need OCR first (e.g. `ocrmypdf`).
- To use another LLM, only `docusense/llm.py` needs changing.
- Ideas: hybrid BM25 + vector search, reranker, OCR support, user auth, multi-user indexes.
