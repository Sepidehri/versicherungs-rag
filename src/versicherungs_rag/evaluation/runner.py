"""Runs the test set against retrieval configurations and LLMs and aggregates the results."""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass

from versicherungs_rag.evaluation import metrics
from versicherungs_rag.evaluation.dataset import EvalCase
from versicherungs_rag.evaluation.judge import FaithfulnessJudge
from versicherungs_rag.llm.base import LLMClient, LLMError, ModelSpec
from versicherungs_rag.pipeline import RAGPipeline
from versicherungs_rag.retrieval.retriever import Retriever
from versicherungs_rag.schema import RetrievedChunk

Progress = Callable[[str], None]


# --------------------------------------------------------------------------- retrieval


@dataclass
class RetrievalSummary:
    name: str
    hit_rate: float
    mrr: float
    latency_ms_p50: float
    num_cases: int


def evaluate_retrieval(name: str, cases: list[EvalCase], retriever: Retriever, top_k: int) -> RetrievalSummary:
    answerable = [c for c in cases if c.answerable]
    hits, rrs, latencies = [], [], []
    for case in answerable:
        start = time.perf_counter()
        contexts = retriever.retrieve(case.question, top_k)
        latencies.append((time.perf_counter() - start) * 1000)
        hits.append(float(metrics.retrieval_hit(contexts, case)))
        rrs.append(metrics.reciprocal_rank(contexts, case))
    return RetrievalSummary(
        name=name,
        hit_rate=metrics.mean(hits) or 0.0,
        mrr=metrics.mean(rrs) or 0.0,
        latency_ms_p50=metrics.percentile(latencies, 0.5) or 0.0,
        num_cases=len(answerable),
    )


# --------------------------------------------------------------------------- generation


@dataclass
class CaseResult:
    case_id: str
    category: str
    model: str
    question: str
    answer: str
    refused: bool
    correct: bool
    fact_recall: float | None
    citation_correct: bool | None
    retrieval_hit: bool | None
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    faithfulness: int | None = None
    judge_reason: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_models(
    cases: list[EvalCase],
    pipeline: RAGPipeline,
    clients: list[LLMClient],
    judge: FaithfulnessJudge | None = None,
    progress: Progress = lambda _: None,
) -> list[CaseResult]:
    # Retrieve once per question and reuse the same context for every model, so the
    # comparison isolates the LLM and retrieval cost is not counted N times.
    retrieved: dict[str, tuple[list[RetrievedChunk], float]] = {
        case.id: pipeline.retrieve(case.question) for case in cases
    }

    results: list[CaseResult] = []
    for client in clients:
        _warm_up(client)
        for i, case in enumerate(cases, start=1):
            progress(f"{client.name}: {i}/{len(cases)} {case.id}")
            contexts, retrieval_latency = retrieved[case.id]
            results.append(_run_case(case, pipeline, client, contexts, retrieval_latency, judge))
    return results


def _warm_up(client: LLMClient) -> None:
    """The first call to a local model includes loading weights into RAM; keep it out of the latency numbers."""
    with contextlib.suppress(LLMError):
        client.generate("Antworte mit OK.", "Test", max_tokens=5)


def _run_case(
    case: EvalCase,
    pipeline: RAGPipeline,
    client: LLMClient,
    contexts: list[RetrievedChunk],
    retrieval_latency: float,
    judge: FaithfulnessJudge | None,
) -> CaseResult:
    spec: ModelSpec = client.spec
    try:
        answer = pipeline.generate(case.question, contexts, client, retrieval_latency)
    except LLMError as exc:
        return CaseResult(
            case.id,
            case.category,
            client.name,
            case.question,
            "",
            False,
            False,
            None,
            None,
            None,
            0.0,
            None,
            None,
            None,
            error=str(exc),
        )

    result = CaseResult(
        case_id=case.id,
        category=case.category,
        model=client.name,
        question=case.question,
        answer=answer.answer,
        refused=answer.refused,
        correct=metrics.is_correct(answer, case),
        fact_recall=metrics.fact_recall(answer.answer, case.expected_facts) if case.answerable else None,
        citation_correct=metrics.citation_correct(answer, case) if case.answerable else None,
        retrieval_hit=metrics.retrieval_hit(contexts, case) if case.answerable else None,
        latency_s=answer.generation_latency_s,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        cost_usd=spec.cost(answer.input_tokens, answer.output_tokens),
    )
    if judge is not None:
        verdict = judge.judge(answer)
        result.faithfulness, result.judge_reason = verdict.faithfulness, verdict.reason
    return result


# --------------------------------------------------------------------------- aggregation


@dataclass
class ModelSummary:
    model: str
    accuracy: float
    accuracy_answerable: float
    fact_recall: float
    citation_accuracy: float
    refusal_accuracy: float  # unanswerable questions correctly declined
    false_refusal_rate: float  # answerable questions wrongly declined
    faithfulness: float | None
    latency_p50_s: float
    latency_p95_s: float
    avg_output_tokens: float | None
    cost_per_1000_questions_usd: float | None
    errors: int
    num_cases: int
    by_category: dict[str, float]


def summarize(results: list[CaseResult]) -> list[ModelSummary]:
    grouped: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        grouped[r.model].append(r)
    return [_summarize_model(model, rows) for model, rows in grouped.items()]


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize_model(model: str, rows: list[CaseResult]) -> ModelSummary:
    ok = [r for r in rows if not r.error]
    answerable = [r for r in ok if r.fact_recall is not None]
    unanswerable = [r for r in ok if r.fact_recall is None]
    costs = [r.cost_usd for r in ok]
    output_tokens = [r.output_tokens for r in ok if r.output_tokens is not None]
    faithfulness = [r.faithfulness for r in ok if r.faithfulness is not None]
    latencies = [r.latency_s for r in ok]

    by_category: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_category[r.category].append(r.correct)

    return ModelSummary(
        model=model,
        accuracy=_rate([r.correct for r in rows]),
        accuracy_answerable=_rate([r.correct for r in answerable]),
        fact_recall=metrics.mean([r.fact_recall for r in answerable]) or 0.0,
        citation_accuracy=_rate([bool(r.citation_correct) for r in answerable]),
        refusal_accuracy=_rate([r.refused for r in unanswerable]),
        false_refusal_rate=_rate([r.refused for r in answerable]),
        faithfulness=metrics.mean(faithfulness),
        latency_p50_s=metrics.percentile(latencies, 0.5) or 0.0,
        latency_p95_s=metrics.percentile(latencies, 0.95) or 0.0,
        avg_output_tokens=metrics.mean(output_tokens),
        cost_per_1000_questions_usd=(
            None if not costs or None in costs else 1000 * sum(costs) / len(costs)  # type: ignore[arg-type]
        ),
        errors=len(rows) - len(ok),
        num_cases=len(rows),
        by_category={cat: _rate(vals) for cat, vals in sorted(by_category.items())},
    )
