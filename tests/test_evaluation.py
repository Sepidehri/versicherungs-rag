import pytest

from versicherungs_rag.evaluation import evaluate_models, load_testset, summarize
from versicherungs_rag.evaluation.dataset import EvalCase, GoldSource
from versicherungs_rag.evaluation.metrics import fact_recall, normalize, percentile
from versicherungs_rag.evaluation.report import write_generation_report
from versicherungs_rag.pipeline import RAGPipeline
from versicherungs_rag.prompts import NO_ANSWER

from .conftest import TESTSET


def test_testset_is_valid_and_references_existing_sections(index):
    cases = load_testset(TESTSET)
    assert len(cases) >= 40
    known = {(c.doc_id, sid) for c in index.chunks for sid in c.section_ids}
    for case in cases:
        for gold in case.gold_sources:
            assert (gold.doc_id, gold.section) in known, case.id


@pytest.mark.parametrize(
    ("answer", "facts", "expected"),
    [
        ("Der Selbstbehalt beträgt EUR 3.000.", (("3.000",),), 1.0),
        ("Bis 3 000 €", (("3000",),), 1.0),
        ("Es sind EUR 500 versichert.", (("50",),), 0.0),  # 50 must not match 500
        ("Mindestens 50 Euro, also 10 %.", (("50",), ("10 %", "10%")), 1.0),
        ("Bis 22:00 Uhr.", (("22",), ("nein",)), 0.5),
    ],
)
def test_fact_recall(answer, facts, expected):
    assert fact_recall(answer, facts) == expected


def test_normalize():
    assert normalize("EUR 1.500,00") == "eur 1500"


def test_percentile():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([], 0.5) is None


def test_end_to_end_evaluation(tmp_path, retriever, fake_llm):
    cases = [
        EvalCase(
            "a",
            "Wie hoch ist der Selbstbehalt in der Vollkasko?",
            "fakt",
            "350",
            (GoldSource("kfz_kasko_avb", "Artikel 5"),),
            (("350",),),
        ),
        EvalCase("u", "Wie hoch ist die Prämie?", "unbeantwortbar", ""),
    ]
    always_350 = fake_llm("Der Selbstbehalt beträgt EUR 350 [1].", name="m1")
    always_refuse = fake_llm(NO_ANSWER, name="m2")
    results = evaluate_models(cases, RAGPipeline(retriever, top_k=4), [always_350, always_refuse])
    summaries = {s.model: s for s in summarize(results)}

    assert summaries["m1"].accuracy_answerable == 1.0
    assert summaries["m1"].refusal_accuracy == 0.0
    assert summaries["m2"].refusal_accuracy == 1.0
    assert summaries["m2"].false_refusal_rate == 1.0
    assert summaries["m1"].citation_accuracy == 1.0

    report = write_generation_report(tmp_path, results, list(summaries.values()), {"test": True})
    assert "m1" in report.read_text(encoding="utf-8")
