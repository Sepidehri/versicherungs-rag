"""Retrieval strategies: BM25, dense, hybrid (reciprocal rank fusion) and hybrid + rerank."""

from __future__ import annotations

from versicherungs_rag.config import RetrievalMode
from versicherungs_rag.retrieval.embeddings import Embedder
from versicherungs_rag.retrieval.index import SearchIndex
from versicherungs_rag.retrieval.reranker import Reranker
from versicherungs_rag.schema import RetrievedChunk

RRF_K = 60


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Merge ranked lists by summing 1 / (k + rank).

    RRF only looks at ranks, so it needs no calibration between BM25 scores (unbounded)
    and cosine similarities (-1..1). k=60 is the value from the original paper
    (Cormack et al., 2009) and works well without tuning.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class Retriever:
    def __init__(
        self,
        index: SearchIndex,
        embedder: Embedder,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        reranker: Reranker | None = None,
        candidates: int = 20,
    ) -> None:
        if mode == RetrievalMode.HYBRID_RERANK and reranker is None:
            raise ValueError("hybrid_rerank mode needs a reranker")
        self.index = index
        self.embedder = embedder
        self.mode = mode
        self.reranker = reranker
        self.candidates = candidates

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        depth = max(self.candidates, top_k)
        if self.mode == RetrievalMode.BM25:
            hits = self.index.bm25.search(query, top_k)
        elif self.mode == RetrievalMode.DENSE:
            hits = self.index.vector_index.search(self.embedder.embed_query(query), top_k)
        else:
            dense = self.index.vector_index.search(self.embedder.embed_query(query), depth)
            sparse = self.index.bm25.search(query, depth)
            hits = reciprocal_rank_fusion([[i for i, _ in dense], [i for i, _ in sparse]])[:depth]
            if self.mode == RetrievalMode.HYBRID_RERANK:
                hits = self._rerank(query, [i for i, _ in hits])
            hits = hits[:top_k]

        return [
            RetrievedChunk(chunk=self.index.chunks[i], score=score, rank=rank)
            for rank, (i, score) in enumerate(hits, start=1)
        ]

    def _rerank(self, query: str, candidate_ids: list[int]) -> list[tuple[int, float]]:
        assert self.reranker is not None
        passages = [self.index.chunks[i].embedding_text for i in candidate_ids]
        scores = self.reranker.score(query, passages)
        return sorted(zip(candidate_ids, scores, strict=True), key=lambda item: item[1], reverse=True)
