"""Concrete LLM clients."""

from __future__ import annotations

import os
import re

import httpx

from versicherungs_rag.llm.base import LLMClient, LLMError, ModelSpec


class OllamaClient(LLMClient):
    """Open-source models served locally by Ollama (https://ollama.com)."""

    def __init__(self, spec: ModelSpec, default_base_url: str = "http://localhost:11434") -> None:
        super().__init__(spec)
        self.base_url = (spec.base_url or default_base_url).rstrip("/")
        self._http = httpx.Client(timeout=httpx.Timeout(300.0, connect=5.0))

    def _generate(self, system: str, user: str, max_tokens: int) -> tuple[str, int | None, int | None]:
        options = {"temperature": 0.0, "num_predict": max_tokens, "num_ctx": 4096, **self.spec.params}
        payload = {
            "model": self.spec.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": options,
        }
        try:
            response = self._http.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed for {self.spec.model}: {exc}") from exc
        data = response.json()
        return data["message"]["content"], data.get("prompt_eval_count"), data.get("eval_count")

    def is_available(self) -> tuple[bool, str]:
        try:
            tags = self._http.get(f"{self.base_url}/api/tags", timeout=3.0).json()
        except (httpx.HTTPError, ValueError):
            return False, f"Ollama not reachable at {self.base_url}"
        installed = {m["name"] for m in tags.get("models", [])}
        wanted = self.spec.model if ":" in self.spec.model else f"{self.spec.model}:latest"
        if wanted not in installed:
            return False, f"Model not pulled. Run: ollama pull {self.spec.model}"
        return True, "ok"


class AnthropicClient(LLMClient):
    """Claude models via the Anthropic Messages API."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        import anthropic

        self._api_key = os.environ.get(spec.api_key_env or "ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=self._api_key) if self._api_key else None

    def _generate(self, system: str, user: str, max_tokens: int) -> tuple[str, int | None, int | None]:
        if self._client is None:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        response = self._client.messages.create(
            model=self.spec.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **self.spec.params,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return text, response.usage.input_tokens, response.usage.output_tokens

    def is_available(self) -> tuple[bool, str]:
        env = self.spec.api_key_env or "ANTHROPIC_API_KEY"
        return (True, "ok") if self._api_key else (False, f"{env} is not set")


class OpenAICompatibleClient(LLMClient):
    """OpenAI, Azure OpenAI (v1 endpoint), vLLM, LM Studio, or Ollama's /v1 endpoint."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        import openai

        self._api_key = os.environ.get(spec.api_key_env or "OPENAI_API_KEY")
        self._client = openai.OpenAI(api_key=self._api_key, base_url=spec.base_url) if self._api_key else None

    def _generate(self, system: str, user: str, max_tokens: int) -> tuple[str, int | None, int | None]:
        if self._client is None:
            raise LLMError(f"{self.spec.api_key_env or 'OPENAI_API_KEY'} is not set")
        response = self._client.chat.completions.create(
            model=self.spec.model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **self.spec.params,
        )
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None,
        )

    def is_available(self) -> tuple[bool, str]:
        env = self.spec.api_key_env or "OPENAI_API_KEY"
        return (True, "ok") if self._api_key else (False, f"{env} is not set")


class ExtractiveBaseline(LLMClient):
    """No LLM at all: returns the context sentences with the highest word overlap.

    This is the floor every real model has to beat. It also shows what an LLM adds:
    the baseline can never say "this is not covered by the documents".
    """

    _CONTEXT_RE = re.compile(r"^\[(\d+)\][^\n]*\n(.*?)(?=^\[\d+\]|^Frage:)", re.MULTILINE | re.DOTALL)

    def _generate(self, system: str, user: str, max_tokens: int) -> tuple[str, int | None, int | None]:
        from versicherungs_rag.retrieval.bm25 import tokenize

        question = user.rsplit("Frage:", 1)[-1].split("\n", 1)[0]
        query_terms = set(tokenize(question))
        scored: list[tuple[float, int, str]] = []
        for number, body in self._CONTEXT_RE.findall(user):
            for sentence in re.split(r"(?<=[.;:])\s+|\n+", body):
                sentence = sentence.strip(" -|")
                terms = set(tokenize(sentence))
                if len(terms) < 3:
                    continue
                overlap = len(query_terms & terms) / max(len(query_terms), 1)
                scored.append((overlap, -int(number), sentence))
        best = sorted(scored, reverse=True)[:2]
        if not best:
            return "", None, None
        return " ".join(f"{sentence} [{-neg_number}]" for _, neg_number, sentence in best), None, None


def build_client(spec: ModelSpec, ollama_base_url: str = "http://localhost:11434") -> LLMClient:
    if spec.provider == "ollama":
        return OllamaClient(spec, ollama_base_url)
    if spec.provider == "anthropic":
        return AnthropicClient(spec)
    if spec.provider == "openai":
        return OpenAICompatibleClient(spec)
    if spec.provider == "extractive":
        return ExtractiveBaseline(spec)
    raise ValueError(f"Unknown provider '{spec.provider}' for model '{spec.name}'")
