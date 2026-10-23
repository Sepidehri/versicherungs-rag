import numpy as np

from versicherungs_rag.config import RetrievalMode
from versicherungs_rag.retrieval import HashingEmbedder, Retriever, SearchIndex, reciprocal_rank_fusion
from versicherungs_rag.retrieval.bm25 import tokenize


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    assert [idx for idx, _ in fused][:2] == [1, 3]


def test_tokenize_stems_and_joins_numbers():
    tokens = tokenize("Die Fahrräder sind bis EUR 3.000 versichert")
    assert "3000" in tokens
    assert "die" not in tokens
    assert tokenize("Fahrrad") == tokenize("Fahrrads")


def test_bm25_finds_exact_terms(index):
    retriever = Retriever(index, HashingEmbedder(), RetrievalMode.BM25)
    top = retriever.retrieve("Marderbiss", top_k=1)[0]
    assert top.chunk.doc_id == "kfz_kasko_avb"
    assert top.chunk.section_ids == ("Artikel 1",)


def test_hybrid_returns_ranked_unique_chunks(retriever):
    results = retriever.retrieve("Selbstbehalt Vollkasko", top_k=4)
    assert [r.rank for r in results] == [1, 2, 3, 4]
    assert len({r.chunk.chunk_id for r in results}) == 4
    assert results[0].chunk.doc_id == "kfz_kasko_avb"


def test_rerank_mode_uses_reranker_order(index):
    class ReverseReranker:
        name = "reverse"

        def score(self, query, passages):
            return [float(i) for i in range(len(passages))]

    retriever = Retriever(
        index, HashingEmbedder(), RetrievalMode.HYBRID_RERANK, reranker=ReverseReranker(), candidates=5
    )
    plain = Retriever(index, HashingEmbedder(), RetrievalMode.HYBRID, candidates=5).retrieve("Selbstbehalt", top_k=5)
    reranked = retriever.retrieve("Selbstbehalt", top_k=5)
    assert [r.chunk.chunk_id for r in reranked] == [r.chunk.chunk_id for r in reversed(plain)]


def test_index_roundtrip(tmp_path, index):
    index.save(tmp_path)
    loaded = SearchIndex.load(tmp_path, expected_embedder="hashing")
    assert loaded.chunks == index.chunks
    assert np.allclose(loaded.vector_index.vectors, index.vector_index.vectors)
