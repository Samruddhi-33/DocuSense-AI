"""Split page text into overlapping, sentence-aware chunks."""
import re
from dataclasses import dataclass

from .loader import Page


@dataclass
class Chunk:
    doc: str
    page: int
    chunk_id: int
    text: str


def split_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "), window.rfind("\n"))
            if cut > size * 0.6:  # prefer breaking at a sentence end
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_pages(doc: str, pages: list[Page], size: int = 1000, overlap: int = 200) -> list[Chunk]:
    chunks, cid = [], 0
    for page in pages:
        for piece in split_text(page.text, size, overlap):
            chunks.append(Chunk(doc, page.number, cid, piece))
            cid += 1
    return chunks
