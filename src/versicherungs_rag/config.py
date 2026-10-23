"""Runtime configuration, read from environment variables (prefix ``RAG_``) or a ``.env`` file."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RetrievalMode(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class ChunkingStrategy(StrEnum):
    SECTION = "section"
    FIXED = "fixed"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    models_config: Path = PROJECT_ROOT / "config" / "models.yaml"
    testset_path: Path = PROJECT_ROOT / "data" / "eval" / "testset.jsonl"
    results_dir: Path = PROJECT_ROOT / "results"

    # "hashing" selects the dependency-free HashingEmbedder (used in CI and tests).
    embedding_model: str = "jinaai/jina-embeddings-v2-base-de"
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual"

    chunking: ChunkingStrategy = ChunkingStrategy.SECTION
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 200

    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID_RERANK
    top_k: int = 4
    rerank_candidates: int = 10

    default_model: str = "llama3.2-3b"
    ollama_base_url: str = "http://localhost:11434"


@lru_cache
def get_settings() -> Settings:
    return Settings()
