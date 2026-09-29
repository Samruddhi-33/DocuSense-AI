"""Higher-level document tasks: summarize, extract key info, compare."""
import json
import re

from .library import Library
from .llm import LLMClient

SEGMENT_CHARS = 15000
MAX_DIRECT_CHARS = 30000

SUMMARY_STYLES = {
    "Short (3-5 sentences)": "Write a concise summary in 3-5 sentences.",
    "Detailed": "Write a detailed structured summary with headings for each major topic.",
    "Bullet points": "Summarize as 6-10 bullet points covering the most important facts.",
}


def _segments(text: str, size: int = SEGMENT_CHARS) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def summarize(lib: Library, llm: LLMClient, doc: str, style: str = "Short (3-5 sentences)") -> str:
    """Map-reduce summary so long documents fit the context window."""
    instruction = SUMMARY_STYLES.get(style, SUMMARY_STYLES["Short (3-5 sentences)"])
    text = lib.full_text(doc)
    system = "You summarize documents faithfully without adding outside information."
    if len(text) <= MAX_DIRECT_CHARS:
        return llm.generate(f"{instruction}\n\nDocument '{doc}':\n{text}", system, 1500)
    partials = [
        llm.generate(f"Summarize the key points of this section:\n{seg}", system, 700)
        for seg in _segments(text)
    ]
    combined = "\n\n".join(f"Section {i+1}: {p}" for i, p in enumerate(partials))
    return llm.generate(
        f"{instruction}\n\nThese are summaries of consecutive sections of '{doc}'. "
        f"Merge them into one coherent summary:\n{combined}",
        system,
        1500,
    )


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def extract_info(lib: Library, llm: LLMClient, doc: str, extra_fields: list[str] | None = None) -> dict:
    fields = {
        "document_type": "e.g. contract, invoice, resume, report",
        "title": "document title or subject",
        "parties_or_people": "list of people/organizations named",
        "dates": "list of important dates with what each refers to",
        "amounts": "list of monetary values / quantities with context",
        "key_points": "list of the main points",
        "obligations_or_action_items": "list of duties, tasks or requirements",
        "deadlines": "list of deadlines",
        "risks_or_red_flags": "list of potential issues",
    }
    for f in extra_fields or []:
        fields[f] = "extract if present"
    spec = "\n".join(f'- "{k}": {v}' for k, v in fields.items())
    text = lib.full_text(doc)[:60000]
    prompt = (
        f"Extract information from the document below. Return ONLY a valid JSON object "
        f"with these keys (use null or [] when absent, never guess):\n{spec}\n\n"
        f"Document '{doc}':\n{text}"
    )
    raw = llm.generate(prompt, "You extract structured data from documents as strict JSON.", 2000)
    try:
        return _parse_json(raw)
    except json.JSONDecodeError:
        return {"raw_output": raw}


def compare(lib: Library, llm: LLMClient, docs: list[str], focus: str = "") -> str:
    """Compare two or more documents (optionally around a focus topic)."""
    if len(docs) < 2:
        raise ValueError("Select at least two documents.")
    blocks = []
    for d in docs:
        text = lib.full_text(d)
        if focus:  # focus on the relevant passages only
            hits = lib.store.search(focus, k=6, docs=[d])
            text = "\n\n".join(f"(p.{c.page}) {c.text}" for c, _ in hits) or text[:MAX_DIRECT_CHARS]
        elif len(text) > MAX_DIRECT_CHARS:
            text = summarize(lib, llm, d, "Detailed")
        blocks.append(f"=== DOCUMENT: {d} ===\n{text}")
    prompt = (
        ("Focus area: " + focus + "\n\n" if focus else "")
        + "Compare the documents below. Produce Markdown with these sections:\n"
        "1. **Overview** of each document (1-2 lines)\n"
        "2. **Key similarities**\n"
        "3. **Key differences** as a table (aspect | " + " | ".join(docs) + ")\n"
        "4. **Conflicts or inconsistencies**\n"
        "5. **Conclusion**\n"
        "Only use facts from the documents.\n\n" + "\n\n".join(blocks)
    )
    return llm.generate(prompt, "You are an expert analyst comparing documents objectively.", 2500)
