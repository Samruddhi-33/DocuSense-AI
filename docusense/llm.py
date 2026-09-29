"""Thin wrapper around the Anthropic API (Claude)."""
from . import config


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        self.model = model or config.LLM_MODEL
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1500) -> str:
        if not self.available:
            raise RuntimeError("No ANTHROPIC_API_KEY configured.")
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()
