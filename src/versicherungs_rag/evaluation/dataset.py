"""The evaluation test set (``data/eval/testset.jsonl``)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

UNANSWERABLE = "unbeantwortbar"


@dataclass(frozen=True)
class GoldSource:
    doc_id: str
    section: str  # normalised section id, e.g. "Artikel 5"


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    category: str
    reference_answer: str
    gold_sources: tuple[GoldSource, ...] = ()
    # Each fact is a list of accepted spellings; the answer must contain one of them.
    expected_facts: tuple[tuple[str, ...], ...] = field(default_factory=tuple)

    @property
    def answerable(self) -> bool:
        return self.category != UNANSWERABLE


def load_testset(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            case = EvalCase(
                id=raw["id"],
                question=raw["question"],
                category=raw["category"],
                reference_answer=raw.get("reference_answer", ""),
                gold_sources=tuple(GoldSource(**g) for g in raw.get("gold_sources", [])),
                expected_facts=tuple(tuple(f) for f in raw.get("expected_facts", [])),
            )
            if case.answerable and not (case.gold_sources and case.expected_facts):
                raise ValueError(
                    f"{path.name}:{line_no}: answerable case '{case.id}' needs gold_sources and expected_facts"
                )
            cases.append(case)
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate case ids in {path.name}")
    return cases
