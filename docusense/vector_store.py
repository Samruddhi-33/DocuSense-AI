"""FAISS vector store with sentence-transformer embeddings."""
import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    # -- embeddings -------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return emb.astype("float32")

    # -- index operations -------------------------------------------------
    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        emb = self._embed([c.text for c in chunks])
        if self.index is None:
            self.index = faiss.IndexFlatIP(emb.shape[1])  # cosine sim (vectors normalized)
        self.index.add(emb)
        self.chunks.extend(chunks)

    def remove_document(self, doc: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c.doc != doc]
        if len(keep) == len(self.chunks):
            return
        if not keep:
            self.index, self.chunks = None, []
            return
        vecs = np.vstack([self.index.reconstruct(i) for i in keep]).astype("float32")
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.chunks = [self.chunks[i] for i in keep]

    def search(self, query: str, k: int = 5, docs: list[str] | None = None):
        """Return [(Chunk, score)] best first, optionally restricted to given docs."""
        if self.index is None or not self.chunks:
            return []
        fetch = len(self.chunks) if docs else min(k, len(self.chunks))
        scores, ids = self.index.search(self._embed([query]), fetch)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if docs and chunk.doc not in docs:
                continue
            results.append((chunk, float(score)))
            if len(results) == k:
                break
        return results

    # -- persistence ------------------------------------------------------
    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            for name in ("index.faiss", "chunks.json"):
                (folder / name).unlink(missing_ok=True)
            return
        faiss.write_index(self.index, str(folder / "index.faiss"))
        (folder / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks]), encoding="utf-8"
        )

    def load(self, folder: Path) -> None:
        idx_path, chunks_path = folder / "index.faiss", folder / "chunks.json"
        if idx_path.exists() and chunks_path.exists():
            self.index = faiss.read_index(str(idx_path))
            self.chunks = [Chunk(**c) for c in json.loads(chunks_path.read_text(encoding="utf-8"))]
