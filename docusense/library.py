"""Document library: keeps raw page text + the vector index, persisted to disk."""
import json
from dataclasses import asdict

from . import config
from .chunker import chunk_pages
from .loader import Page, load_document
from .vector_store import VectorStore


class Library:
    def __init__(self):
        self.store = VectorStore(config.EMBEDDING_MODEL)
        self.pages: dict[str, list[Page]] = {}

    @classmethod
    def load_or_create(cls) -> "Library":
        lib = cls()
        lib.store.load(config.DATA_DIR)
        pages_file = config.DATA_DIR / "pages.json"
        if pages_file.exists():
            raw = json.loads(pages_file.read_text(encoding="utf-8"))
            lib.pages = {k: [Page(**p) for p in v] for k, v in raw.items()}
        return lib

    def save(self) -> None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.store.save(config.DATA_DIR)
        (config.DATA_DIR / "pages.json").write_text(
            json.dumps({k: [asdict(p) for p in v] for k, v in self.pages.items()}),
            encoding="utf-8",
        )

    @property
    def documents(self) -> list[str]:
        return sorted(self.pages)

    def add_document(self, filename: str, data: bytes) -> int:
        """Parse, chunk, embed and index a file. Returns number of chunks."""
        pages = load_document(filename, data)
        if filename in self.pages:
            self.remove_document(filename)
        chunks = chunk_pages(filename, pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.store.add(chunks)
        self.pages[filename] = pages
        self.save()
        return len(chunks)

    def remove_document(self, filename: str) -> None:
        self.store.remove_document(filename)
        self.pages.pop(filename, None)
        self.save()

    def full_text(self, filename: str) -> str:
        return "\n\n".join(f"[Page {p.number}]\n{p.text}" for p in self.pages[filename])
