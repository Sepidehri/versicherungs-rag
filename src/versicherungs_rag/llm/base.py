"""Provider-independent LLM interface.

Every provider (local Ollama, Anthropic, any OpenAI-compatible endpoint) implements
``LLMClient.generate``. The rest of the system never imports a provider SDK directly,
which is what makes models interchangeable in the pipeline and the evaluation.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class ModelSpec:
    """One entry of ``config/models.yaml``."""

    name: str
    provider: str
    model: str = ""
    base_url: str | None = None
    api_key_env: str | None = None
    params: dict = field(default_factory=dict)
    # USD per million tokens; leave empty when unknown so reports show "n/a", not a guess.
    price_input_per_mtok: float | None = None
    price_output_per_mtok: float | None = None
    description: str = ""

    def cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        if self.provider in {"ollama", "extractive"}:
            return 0.0
        if None in (self.price_input_per_mtok, self.price_output_per_mtok, input_tokens, output_tokens):
            return None
        return (input_tokens * self.price_input_per_mtok + output_tokens * self.price_output_per_mtok) / 1e6


class LLMError(RuntimeError):
    pass


class LLMClient(ABC):
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    def generate(self, system: str, user: str, max_tokens: int = 400) -> LLMResponse:
        start = time.perf_counter()
        text, input_tokens, output_tokens = self._generate(system, user, max_tokens)
        return LLMResponse(
            text=text.strip(),
            model=self.spec.name,
            latency_s=time.perf_counter() - start,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @abstractmethod
    def _generate(self, system: str, user: str, max_tokens: int) -> tuple[str, int | None, int | None]:
        """Return (text, input_tokens, output_tokens)."""

    def is_available(self) -> tuple[bool, str]:
        """Cheap readiness check used by the API and the evaluation runner."""
        return True, "ok"
