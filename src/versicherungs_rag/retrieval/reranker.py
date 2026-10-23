"""Cross-encoder reranking.

A bi-encoder (the embedder) scores query and chunk independently; a cross-encoder reads
both together and is much more precise, but too slow to run on the whole corpus. So we
retrieve ~20 candidates cheaply and let the cross-encoder pick the final top-k.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Reranker(ABC):
    name: str

    @abstractmethod
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class FastEmbedReranker(Reranker):
    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._model.rerank(query, passages)]
