"""Core data types shared across ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Document:
    """A source document after loading, before chunking."""

    doc_id: str
    title: str
    text: str
    source: str
    # Character offset at which each page starts; empty for formats without pages.
    page_offsets: tuple[int, ...] = ()

    def page_at(self, offset: int) -> int | None:
        if not self.page_offsets:
            return None
        page = 1
        for i, start in enumerate(self.page_offsets, start=1):
            if start > offset:
                break
            page = i
        return page


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    # Human-readable heading, e.g. "Artikel 5 – Leitungswasserschäden".
    section: str = ""
    # Normalised ids ("Artikel 5") of every section the chunk overlaps; used for evaluation.
    section_ids: tuple[str, ...] = ()
    page: int | None = None

    @property
    def embedding_text(self) -> str:
        """Text used for embedding and keyword indexing.

        The document title and section heading are prepended so that chunks with
        similar wording from different products (e.g. "Selbstbehalt" in the
        household and the car policy) stay distinguishable.
        """
        header = " – ".join(part for part in (self.doc_title, self.section) if part)
        return f"{header}\n{self.text}" if header else self.text

    @property
    def citation_label(self) -> str:
        label = self.doc_title
        if self.section:
            label += f", {self.section}"
        if self.page is not None:
            label += f", S. {self.page}"
        return label

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        data = dict(data)
        data["section_ids"] = tuple(data.get("section_ids", ()))
        return cls(**data)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


@dataclass
class Citation:
    number: int
    chunk_id: str
    label: str


@dataclass
class RAGAnswer:
    question: str
    answer: str
    model: str
    refused: bool
    citations: list[Citation]
    contexts: list[RetrievedChunk]
    retrieval_latency_s: float
    generation_latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "model": self.model,
            "refused": self.refused,
            "citations": [asdict(c) for c in self.citations],
            "contexts": [
                {
                    "rank": rc.rank,
                    "score": round(rc.score, 4),
                    "chunk_id": rc.chunk.chunk_id,
                    "label": rc.chunk.citation_label,
                    "text": rc.chunk.text,
                }
                for rc in self.contexts
            ],
            "retrieval_latency_s": round(self.retrieval_latency_s, 3),
            "generation_latency_s": round(self.generation_latency_s, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
