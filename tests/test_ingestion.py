from versicherungs_rag.ingestion import FixedSizeChunker, SectionChunker
from versicherungs_rag.ingestion.chunking import section_id
from versicherungs_rag.schema import Document


def test_loader_extracts_short_title(documents):
    titles = {d.doc_id: d.title for d in documents}
    assert titles["haushaltsversicherung_avb"] == "AVB Haushalt 2026"


def test_section_chunker_one_chunk_per_article(documents):
    household = next(d for d in documents if d.doc_id == "haushaltsversicherung_avb")
    chunks = SectionChunker().chunk(household)
    assert [c.section_ids for c in chunks] == [(f"Artikel {i}",) for i in range(1, 16)]
    # The heading is not duplicated in the body but is part of the embedding text.
    assert not chunks[0].text.startswith("##")
    assert chunks[0].embedding_text.startswith("AVB Haushalt 2026 – Artikel 1")


def test_section_chunker_splits_long_sections_with_overlap():
    paragraphs = [f"Absatz {i} " + "wort " * 60 for i in range(6)]
    doc = Document("d", "Doc", "## Artikel 1 – Lang\n\n" + "\n\n".join(paragraphs), "mem")
    chunks = SectionChunker(max_chars=700, overlap_chars=100).chunk(doc)
    assert len(chunks) > 1
    assert all(len(c.text) <= 700 + 100 + 2 for c in chunks)
    assert all(c.section_ids == ("Artikel 1",) for c in chunks)


def test_fixed_chunker_tracks_overlapping_sections(documents):
    chunks = FixedSizeChunker(max_chars=800, overlap_chars=150).chunk(documents[0])
    assert any(len(c.section_ids) > 1 for c in chunks)


def test_section_id_normalisation():
    assert section_id("Artikel 5 – Leitungswasser") == "Artikel 5"
    assert section_id("Art. 12 Obliegenheiten") == "Artikel 12"
    assert section_id("§ 3a Begriffe") == "§ 3a"


def test_plain_text_headings_are_detected():
    text = "Präambel\n\nArtikel 1 Gegenstand\nText eins.\n\nArtikel 2 Umfang\nText zwei."
    chunks = SectionChunker().chunk(Document("p", "PDF", text, "mem", page_offsets=(0,)))
    assert [c.section_ids for c in chunks] == [("Artikel 1",), ("Artikel 2",)]
    assert chunks[0].page == 1
