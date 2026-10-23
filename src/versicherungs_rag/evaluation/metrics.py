"""Deterministic metrics. No LLM is needed to compute them, so results are reproducible."""

from __future__ import annotations

import math
import re

from versicherungs_rag.evaluation.dataset import EvalCase, GoldSource
from versicherungs_rag.schema import Chunk, RAGAnswer, RetrievedChunk


def normalize(text: str) -> str:
    text = text.lower().replace("€", " eur ").replace("\u00a0", " ")
    text = re.sub(r"(?<=\d)[.\u202f ](?=\d{3}\b)", "", text)  # 3.000 / 3 000 -> 3000
    text = re.sub(r"(?<=\d),00\b", "", text)  # 300,00 -> 300
    return re.sub(r"\s+", " ", text)


def _variant_in(answer: str, variant: str) -> bool:
    variant = normalize(variant)
    if re.fullmatch(r"[\d.,% ]+", variant):
        # Numbers must match as whole numbers: "50" must not match "500".
        return re.search(rf"(?<![\d.,]){re.escape(variant)}(?![\d]|[.,]\d)", answer) is not None
    return variant in answer


def fact_recall(answer: str, facts: tuple[tuple[str, ...], ...]) -> float:
    if not facts:
        return 1.0
    normalized = normalize(answer)
    found = sum(any(_variant_in(normalized, v) for v in variants) for variants in facts)
    return found / len(facts)


def is_gold(chunk: Chunk, gold_sources: tuple[GoldSource, ...]) -> bool:
    return any(chunk.doc_id == g.doc_id and g.section in chunk.section_ids for g in gold_sources)


def retrieval_hit(contexts: list[RetrievedChunk], case: EvalCase) -> bool:
    """True if every gold source appears in the retrieved context (multi-document questions need all)."""
    return all(any(is_gold(rc.chunk, (g,)) for rc in contexts) for g in case.gold_sources)


def reciprocal_rank(contexts: list[RetrievedChunk], case: EvalCase) -> float:
    for rc in contexts:
        if is_gold(rc.chunk, case.gold_sources):
            return 1.0 / rc.rank
    return 0.0


def citation_correct(answer: RAGAnswer, case: EvalCase) -> bool:
    """At least one citation points at a gold source."""
    by_id = {rc.chunk.chunk_id: rc.chunk for rc in answer.contexts}
    return any(is_gold(by_id[c.chunk_id], case.gold_sources) for c in answer.citations if c.chunk_id in by_id)


def is_correct(answer: RAGAnswer, case: EvalCase) -> bool:
    if not case.answerable:
        return answer.refused
    return not answer.refused and fact_recall(answer.answer, case.expected_facts) == 1.0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower, upper = math.floor(pos), math.ceil(pos)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
