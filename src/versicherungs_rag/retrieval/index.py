"""The persisted search index: chunks, their embeddings and a BM25 index.

For a corpus of a few thousand chunks a NumPy matrix with exact cosine search is fast
(well under a millisecond per query) and has no infrastructure to run. The
``VectorIndex`` interface is small enough to swap in Qdrant, pgvector or Azure AI Search
when the corpus grows.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from versicherungs_rag.ingestion.chunking import Chunker
from versicherungs_rag.retrieval.bm25 import BM25Index
from versicherungs_rag.retrieval.embeddings import Embedder
from versicherungs_rag.schema import Chunk, Document

_CHUNKS_FILE = "chunks.jsonl"
_VECTORS_FILE = "embeddings.npy"
_MANIFEST_FILE = "manifest.json"


class VectorIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if len(self.vectors) == 0:
            return []
        # Vectors are L2-normalised, so the dot product is the cosine similarity.
        scores = self.vectors @ query_vector
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]


class SearchIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str, chunker_name: str) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks and embeddings differ")
        self.chunks = chunks
        self.embedder_name = embedder_name
        self.chunker_name = chunker_name
        self.vector_index = VectorIndex(vectors)
        self.bm25 = BM25Index([c.embedding_text for c in chunks])

    @classmethod
    def build(cls, documents: list[Document], chunker: Chunker, embedder: Embedder) -> SearchIndex:
        chunks = chunker.chunk_all(documents)
        vectors = embedder.embed_documents([c.embedding_text for c in chunks])
        return cls(chunks, vectors, embedder.name, chunker.name)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / _CHUNKS_FILE).open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        np.save(directory / _VECTORS_FILE, self.vector_index.vectors)
        manifest = {
            "embedding_model": self.embedder_name,
            "chunker": self.chunker_name,
            "num_chunks": len(self.chunks),
            "documents": sorted({c.doc_id for c in self.chunks}),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (directory / _MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, expected_embedder: str | None = None) -> SearchIndex:
        manifest_path = directory / _MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"No index in {directory}. Run `vrag ingest` first.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected_embedder and manifest["embedding_model"] != expected_embedder:
            raise ValueError(
                f"Index was built with '{manifest['embedding_model']}' but '{expected_embedder}' "
                "is configured. Re-run `vrag ingest`."
            )
        with (directory / _CHUNKS_FILE).open(encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        vectors = np.load(directory / _VECTORS_FILE)
        return cls(chunks, vectors, manifest["embedding_model"], manifest["chunker"])
