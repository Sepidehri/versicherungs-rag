"""Writes evaluation results as JSON (machine-readable) and Markdown (for humans / the README)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from versicherungs_rag.evaluation.runner import CaseResult, ModelSummary, RetrievalSummary


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{100 * value:.0f} %"


def _num(value: float | None, fmt: str = "{:.1f}") -> str:
    return "–" if value is None else fmt.format(value)


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.2f}"


def model_table(summaries: list[ModelSummary]) -> str:
    rows = [
        "| Modell | Genauigkeit | Fakten-Recall | Zitat korrekt | Korrekt abgelehnt | Fälschlich abgelehnt "
        "| Faithfulness (1–5) | Latenz p50 / p95 | Kosten / 1000 Fragen |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted(summaries, key=lambda s: s.accuracy, reverse=True):
        rows.append(
            f"| {s.model} | **{_pct(s.accuracy)}** | {_pct(s.fact_recall)} | {_pct(s.citation_accuracy)} "
            f"| {_pct(s.refusal_accuracy)} | {_pct(s.false_refusal_rate)} | {_num(s.faithfulness)} "
            f"| {s.latency_p50_s:.1f} s / {s.latency_p95_s:.1f} s | {_usd(s.cost_per_1000_questions_usd)} |"
        )
    return "\n".join(rows)


def category_table(summaries: list[ModelSummary]) -> str:
    categories = sorted({c for s in summaries for c in s.by_category})
    rows = ["| Modell | " + " | ".join(categories) + " |", "|---" * (len(categories) + 1) + "|"]
    for s in summaries:
        rows.append(f"| {s.model} | " + " | ".join(_pct(s.by_category.get(c)) for c in categories) + " |")
    return "\n".join(rows)


def retrieval_table(summaries: list[RetrievalSummary], top_k: int) -> str:
    rows = [
        f"| Konfiguration | Hit Rate @{top_k} | MRR | Latenz p50 |",
        "|---|---|---|---|",
    ]
    for s in sorted(summaries, key=lambda s: (s.hit_rate, s.mrr), reverse=True):
        rows.append(f"| {s.name} | **{_pct(s.hit_rate)}** | {s.mrr:.2f} | {s.latency_ms_p50:.0f} ms |")
    return "\n".join(rows)


def failures(results: list[CaseResult], limit: int = 12) -> str:
    wrong = [r for r in results if not r.correct][:limit]
    if not wrong:
        return "_Keine Fehler._"
    lines = []
    for r in wrong:
        answer = (r.error or r.answer).replace("\n", " ")
        lines.append(f"- **{r.model}** · `{r.case_id}` ({r.category}) – {r.question}\n  > {answer[:300]}")
    return "\n".join(lines)


def write_generation_report(
    out_dir: Path,
    results: list[CaseResult],
    summaries: list[ModelSummary],
    config: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(
        json.dumps({"config": config, "models": [asdict(s) for s in summaries]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    config_lines = "\n".join(f"- {k}: `{v}`" for k, v in config.items())
    report = f"""# Evaluationsbericht: Modellvergleich

## Konfiguration
{config_lines}

## Ergebnisse
{model_table(summaries)}

**Genauigkeit**: beantwortbare Fragen enthalten alle erwarteten Fakten, unbeantwortbare Fragen werden abgelehnt.
**Faithfulness**: LLM-as-a-Judge, nur wenn mit `--judge` aktiviert.

## Genauigkeit nach Fragekategorie
{category_table(summaries)}

## Fehlerbeispiele
{failures(results)}
"""
    path = out_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    return path


def write_retrieval_report(out_dir: Path, summaries: list[RetrievalSummary], top_k: int, config: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_summary.json").write_text(
        json.dumps({"config": config, "results": [asdict(s) for s in summaries]}, indent=2), encoding="utf-8"
    )
    config_lines = "\n".join(f"- {k}: `{v}`" for k, v in config.items())
    path = out_dir / "retrieval_report.md"
    path.write_text(
        f"# Evaluationsbericht: Retrieval\n\n## Konfiguration\n{config_lines}\n\n## Ergebnisse\n"
        f"{retrieval_table(summaries, top_k)}\n\n"
        f"**Hit Rate**: Anteil der Fragen, bei denen alle relevanten Artikel unter den Top-{top_k} sind. "
        "**MRR**: Mean Reciprocal Rank des ersten relevanten Treffers.\n",
        encoding="utf-8",
    )
    return path
