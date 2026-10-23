"""The RAG pipeline: retrieve -> build prompt -> generate -> attach citations."""

from __future__ import annotations

import time

from versicherungs_rag.llm.base import LLMClient
from versicherungs_rag.prompts import SYSTEM_PROMPT, build_user_prompt, extract_citation_numbers, is_refusal
from versicherungs_rag.retrieval.retriever import Retriever
from versicherungs_rag.schema import Citation, RAGAnswer, RetrievedChunk


class RAGPipeline:
    def __init__(self, retriever: Retriever, top_k: int = 4) -> None:
        self.retriever = retriever
        self.top_k = top_k

    def retrieve(self, question: str) -> tuple[list[RetrievedChunk], float]:
        start = time.perf_counter()
        contexts = self.retriever.retrieve(question, self.top_k)
        return contexts, time.perf_counter() - start

    def answer(self, question: str, llm: LLMClient) -> RAGAnswer:
        contexts, retrieval_latency = self.retrieve(question)
        return self.generate(question, contexts, llm, retrieval_latency)

    def generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        llm: LLMClient,
        retrieval_latency_s: float = 0.0,
    ) -> RAGAnswer:
        """Generation step on its own, so the evaluation can reuse one retrieval for many models."""
        response = llm.generate(SYSTEM_PROMPT, build_user_prompt(question, contexts))
        citations = [
            Citation(number=n, chunk_id=contexts[n - 1].chunk.chunk_id, label=contexts[n - 1].chunk.citation_label)
            for n in extract_citation_numbers(response.text)
            if 1 <= n <= len(contexts)  # ignore hallucinated citation numbers
        ]
        return RAGAnswer(
            question=question,
            answer=response.text,
            model=llm.name,
            refused=is_refusal(response.text),
            citations=citations,
            contexts=contexts,
            retrieval_latency_s=retrieval_latency_s,
            generation_latency_s=response.latency_s,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
