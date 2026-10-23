# Evaluationsbericht: Retrieval

## Konfiguration
- Fragen (beantwortbar): `36`
- Top-k: `4`
- Reranker: `jinaai/jina-reranker-v2-base-multilingual`

## Ergebnisse
| Konfiguration | Hit Rate @4 | MRR | Latenz p50 |
|---|---|---|---|
| section · hybrid_rerank · jina-embeddings-v2-base-de | **100 %** | 1.00 | 6656 ms |
| fixed · hybrid_rerank · jina-embeddings-v2-base-de | **100 %** | 0.99 | 7296 ms |
| fixed · hybrid · jina-embeddings-v2-base-de | **100 %** | 0.97 | 94 ms |
| section · hybrid · jina-embeddings-v2-base-de | **100 %** | 0.93 | 87 ms |
| section · dense · jina-embeddings-v2-base-de | **100 %** | 0.90 | 85 ms |
| fixed · dense · jina-embeddings-v2-base-de | **97 %** | 0.97 | 96 ms |
| fixed · bm25 | **97 %** | 0.91 | 1 ms |
| section · bm25 | **92 %** | 0.85 | 0 ms |

**Hit Rate**: Anteil der Fragen, bei denen alle relevanten Artikel unter den Top-4 sind. **MRR**: Mean Reciprocal Rank des ersten relevanten Treffers.
