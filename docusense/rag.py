"""Retrieval-Augmented Generation: question -> retrieve -> grounded answer."""
from dataclasses import dataclass

from . import config
from .chunker import Chunk
from .library import Library
from .llm import LLMClient

SYSTEM = (
    "You are DocuSense, a careful document assistant. Answer ONLY from the provided "
    "context passages. Cite sources inline like [1], [2] using the passage numbers. "
    "If the answer is not in the context, say you could not find it in the documents. "
    "Be concise and precise; never invent facts, figures or dates."
)


@dataclass
class Answer:
    text: str
    sources: list[tuple[Chunk, float]]


def answer_question(
    lib: Library,
    llm: LLMClient,
    question: str,
    docs: list[str] | None = None,
    history: list[tuple[str, str]] | None = None,
    top_k: int = config.TOP_K,
) -> Answer:
    history = history or []
    # short follow-ups ("what about the penalty?") retrieve better with prior question added
    query = question if len(question.split()) > 6 or not history else f"{history[-1][0]} {question}"
    hits = lib.store.search(query, k=top_k, docs=docs)
    if not hits:
        return Answer("No documents are indexed yet. Upload some first.", [])

    if not llm.available:  # offline fallback: show best passages
        text = "_No API key set - showing the most relevant passages:_\n\n" + "\n\n".join(
            f"**[{i}] {c.doc} (p.{c.page})**: {c.text[:400]}..." for i, (c, _) in enumerate(hits, 1)
        )
        return Answer(text, hits)

    context = "\n\n".join(
        f"[{i}] (source: {c.doc}, page {c.page})\n{c.text}" for i, (c, _) in enumerate(hits, 1)
    )
    convo = "".join(f"User: {q}\nAssistant: {a}\n" for q, a in history[-3:])
    prompt = (
        f"Context passages:\n{context}\n\n"
        + (f"Previous conversation:\n{convo}\n" if convo else "")
        + f"Question: {question}\n\nAnswer with citations:"
    )
    return Answer(llm.generate(prompt, SYSTEM), hits)
