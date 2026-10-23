"""Wires the building blocks together from :class:`Settings`.

Model weights are loaded lazily and only once, because loading the embedder and the
reranker takes seconds while a query takes milliseconds.
"""

from __future__ import annotations

from functools import cached_property

from versicherungs_rag.config import RetrievalMode, Settings
from versicherungs_rag.ingestion import build_chunker, load_directory
from versicherungs_rag.llm import ModelRegistry
from versicherungs_rag.pipeline import RAGPipeline
from versicherungs_rag.retrieval import (
    Embedder,
    FastEmbedReranker,
    Reranker,
    Retriever,
    SearchIndex,
    build_embedder,
)


class Components:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @cached_property
    def embedder(self) -> Embedder:
        return build_embedder(self.settings.embedding_model)

    @cached_property
    def reranker(self) -> Reranker:
        return FastEmbedReranker(self.settings.reranker_model)

    @cached_property
    def index(self) -> SearchIndex:
        return SearchIndex.load(self.settings.index_dir, expected_embedder=self.settings.embedding_model)

    @cached_property
    def models(self) -> ModelRegistry:
        return ModelRegistry.from_yaml(self.settings.models_config, self.settings.ollama_base_url)

    def build_index(self, chunking: str | None = None) -> SearchIndex:
        s = self.settings
        chunker = build_chunker(chunking or s.chunking.value, s.chunk_max_chars, s.chunk_overlap_chars)
        index = SearchIndex.build(load_directory(s.documents_dir), chunker, self.embedder)
        index.save(s.index_dir)
        self.__dict__["index"] = index  # refresh the cached property
        return index

    def retriever(self, mode: RetrievalMode | None = None, index: SearchIndex | None = None) -> Retriever:
        mode = mode or self.settings.retrieval_mode
        return Retriever(
            index=index or self.index,
            embedder=self.embedder,
            mode=mode,
            reranker=self.reranker if mode == RetrievalMode.HYBRID_RERANK else None,
            candidates=self.settings.rerank_candidates,
        )

    def pipeline(self, mode: RetrievalMode | None = None) -> RAGPipeline:
        return RAGPipeline(self.retriever(mode), top_k=self.settings.top_k)
