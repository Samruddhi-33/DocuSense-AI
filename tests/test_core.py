import io

from docx import Document

from docusense.chunker import chunk_pages, split_text
from docusense.features import _parse_json
from docusense.loader import Page, load_document


def test_split_short_text():
    assert split_text("Hello world.", 1000, 200) == ["Hello world."]


def test_split_long_text_overlaps_and_terminates():
    text = " ".join(f"Sentence number {i}." for i in range(400))
    parts = split_text(text, 500, 100)
    assert len(parts) > 5
    assert all(len(p) <= 500 for p in parts)


def test_chunk_pages_metadata():
    chunks = chunk_pages("a.pdf", [Page(1, "One."), Page(2, "Two.")])
    assert [(c.page, c.chunk_id) for c in chunks] == [(1, 0), (2, 1)]


def test_load_docx_roundtrip():
    d = Document()
    d.add_paragraph("Agreement between Acme and Beta.")
    buf = io.BytesIO()
    d.save(buf)
    pages = load_document("x.docx", buf.getvalue())
    assert "Acme" in pages[0].text


def test_parse_json_with_fences():
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}
