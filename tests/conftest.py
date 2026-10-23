from pathlib import Path

import pytest

from versicherungs_rag.config import RetrievalMode
from versicherungs_rag.ingestion import SectionChunker, load_directory
from versicherungs_rag.llm.base import LLMClient, ModelSpec
from versicherungs_rag.retrieval import HashingEmbedder, Retriever, SearchIndex

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "documents"
TESTSET = ROOT / "data" / "eval" / "testset.jsonl"


@pytest.fixture(scope="session")
def documents():
    return load_directory(DOCS_DIR)


@pytest.fixture(scope="session")
def index(documents):
    return SearchIndex.build(documents, SectionChunker(), HashingEmbedder())


@pytest.fixture
def retriever(index):
    return Retriever(index, HashingEmbedder(), RetrievalMode.HYBRID)


class FakeLLM(LLMClient):
    """Returns a canned answer and records the prompt it received."""

    def __init__(self, answer: str, name: str = "fake") -> None:
        super().__init__(ModelSpec(name=name, provider="fake"))
        self.answer = answer
        self.last_user_prompt = ""

    def _generate(self, system, user, max_tokens):
        self.last_user_prompt = user
        return self.answer, 100, 20


@pytest.fixture
def fake_llm():
    return FakeLLM
