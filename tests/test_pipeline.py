from versicherungs_rag.llm.base import ModelSpec
from versicherungs_rag.llm.providers import ExtractiveBaseline
from versicherungs_rag.pipeline import RAGPipeline
from versicherungs_rag.prompts import NO_ANSWER, extract_citation_numbers, is_refusal


def test_citations_are_parsed_and_mapped(retriever, fake_llm):
    llm = fake_llm("Der Selbstbehalt beträgt EUR 350 [1]. Siehe auch [2, 3] und [9].")
    answer = RAGPipeline(retriever, top_k=4).answer("Selbstbehalt Vollkasko?", llm)
    assert [c.number for c in answer.citations] == [1, 2, 3]  # [9] does not exist and is dropped
    assert answer.citations[0].chunk_id == answer.contexts[0].chunk.chunk_id
    assert not answer.refused
    assert "[1] AVB" in llm.last_user_prompt


def test_refusal_detection():
    assert is_refusal(NO_ANSWER)
    assert is_refusal("Die Auszüge enthalten dazu keine Angaben.")
    assert not is_refusal("Ja, der Marderbiss ist bis EUR 1.500 versichert [1].")


def test_extract_citation_numbers_deduplicates():
    assert extract_citation_numbers("a [2] b [1; 2] c [3,4]") == [2, 1, 3, 4]


def test_extractive_baseline_answers_from_context(retriever):
    baseline = ExtractiveBaseline(ModelSpec(name="baseline", provider="extractive"))
    answer = RAGPipeline(retriever, top_k=4).answer("Wie hoch ist der Selbstbehalt in der Vollkasko?", baseline)
    assert "350" in answer.answer
    assert answer.citations
