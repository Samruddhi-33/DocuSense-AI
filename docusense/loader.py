"""Read PDF / Word / text files into pages of plain text."""
import io
from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader


@dataclass
class Page:
    number: int
    text: str


def _load_pdf(data: bytes) -> list[Page]:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise ValueError("PDF is password protected.") from exc
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(i, text))
    return pages


def _load_docx(data: bytes, section_chars: int = 3000) -> list[Page]:
    """Word files have no fixed pages, so we group content into ~3000-char sections."""
    doc = DocxDocument(io.BytesIO(data))
    lines = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append(" | ".join(cells))
    pages, buf, num = [], "", 1
    for line in lines:
        if buf and len(buf) + len(line) > section_chars:
            pages.append(Page(num, buf.strip()))
            num += 1
            buf = ""
        buf += line + "\n"
    if buf.strip():
        pages.append(Page(num, buf.strip()))
    return pages


def load_document(filename: str, data: bytes) -> list[Page]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        pages = _load_pdf(data)
    elif ext == ".docx":
        pages = _load_docx(data)
    elif ext in (".txt", ".md"):
        pages = [Page(1, data.decode("utf-8", errors="ignore").strip())]
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    pages = [p for p in pages if p.text]
    if not pages:
        raise ValueError(
            "No extractable text found. Scanned PDFs need OCR first (e.g. ocrmypdf)."
        )
    return pages
