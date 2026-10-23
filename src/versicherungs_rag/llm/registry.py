"""Loads model definitions from ``config/models.yaml`` and caches client instances."""

from __future__ import annotations

from pathlib import Path

import yaml

from versicherungs_rag.llm.base import LLMClient, ModelSpec
from versicherungs_rag.llm.providers import build_client


class ModelRegistry:
    def __init__(self, specs: list[ModelSpec], ollama_base_url: str = "http://localhost:11434") -> None:
        self._specs = {spec.name: spec for spec in specs}
        self._clients: dict[str, LLMClient] = {}
        self._ollama_base_url = ollama_base_url

    @classmethod
    def from_yaml(cls, path: Path, ollama_base_url: str = "http://localhost:11434") -> ModelRegistry:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls([ModelSpec(**entry) for entry in raw.get("models", [])], ollama_base_url)

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    def spec(self, name: str) -> ModelSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(f"Unknown model '{name}'. Configured: {', '.join(self._specs)}") from None

    def client(self, name: str) -> LLMClient:
        if name not in self._clients:
            self._clients[name] = build_client(self.spec(name), self._ollama_base_url)
        return self._clients[name]
