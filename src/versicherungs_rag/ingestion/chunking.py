"""Chunking strategies.

``SectionChunker`` follows the legal structure of insurance conditions (one chunk per
article, split further only when an article is long). ``FixedSizeChunker`` is the naive
sliding-window baseline used for comparison in the retrieval evaluation.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from versicherungs_rag.schema import Chunk, Document

# Markdown headings ("## Artikel 5 – ...") or plain-text article headings as found in PDFs.
_HEADING_RE = re.compile(
    r"^(?:#{2,6}\s*(?P<md>.+)|(?P<plain>(?:Artikel|Art\.|§)\s*\d+[a-z]?\b.{0,80}))$",
    flags=re.MULTILINE,
)
_SECTION_ID_RE = re.compile(r"^(Artikel|Art\.|§)\s*(\d+[a-z]?)")


@dataclass(frozen=True)
class _Section:
    heading: str
    start: int
    end: int


def section_id(heading: str) -> str:
    """Normalise a heading to a comparable id: 'Art. 5 Leitungswasser' -> 'Artikel 5'."""
    match = _SECTION_ID_RE.match(heading.strip())
    if not match:
        return heading.strip()
    prefix = "§" if match.group(1) == "§" else "Artikel"
    return f"{prefix} {match.group(2)}"


def find_sections(text: str) -> list[_Section]:
    matches = list(_HEADING_RE.finditer(text))
    sections: list[_Section] = []
    if not matches:
        return [_Section(heading="", start=0, end=len(text))]
    if matches[0].start() > 0:
        sections.append(_Section(heading="", start=0, end=matches[0].start()))
    for i, match in enumerate(matches):
        heading = (match.group("md") or match.group("plain")).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(_Section(heading=heading, start=match.start(), end=end))
    return sections


class Chunker(ABC):
    name: str

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]: ...

    def chunk_all(self, documents: list[Document]) -> list[Chunk]:
        return [chunk for doc in documents for chunk in self.chunk(doc)]


class SectionChunker(Chunker):
    name = "section"

    def __init__(self, max_chars: int = 1200, overlap_chars: int = 200) -> None:
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in find_sections(document.text):
            body = document.text[section.start : section.end]
            # Drop the heading line itself; it is re-added via Chunk.embedding_text.
            if section.heading:
                body = body.split("\n", 1)[1] if "\n" in body else ""
            body = body.strip()
            if not body or _is_preamble(section, body):
                continue
            sid = (section_id(section.heading),) if section.heading else ()
            for part in _split_paragraphs(body, self.max_chars, self.overlap_chars):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}::{len(chunks)}",
                        doc_id=document.doc_id,
                        doc_title=document.title,
                        text=part,
                        section=section.heading,
                        section_ids=sid,
                        page=document.page_at(section.start),
                    )
                )
        return chunks


class FixedSizeChunker(Chunker):
    name = "fixed"

    def __init__(self, max_chars: int = 800, overlap_chars: int = 150) -> None:
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        sections = find_sections(text)
        chunks: list[Chunk] = []
        start = 0
        step = self.max_chars - self.overlap_chars
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            piece = text[start:end].strip()
            if piece:
                overlapping = [s for s in sections if s.start < end and s.end > start and s.heading]
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}::{len(chunks)}",
                        doc_id=document.doc_id,
                        doc_title=document.title,
                        text=piece,
                        section="",
                        section_ids=tuple(section_id(s.heading) for s in overlapping),
                        page=document.page_at(start),
                    )
                )
            if end == len(text):
                break
            start += step
        return chunks


def build_chunker(strategy: str, max_chars: int, overlap_chars: int) -> Chunker:
    if strategy == "section":
        return SectionChunker(max_chars, overlap_chars)
    if strategy == "fixed":
        return FixedSizeChunker(max_chars, overlap_chars)
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def _is_preamble(section: _Section, body: str) -> bool:
    """The text before the first article (title + disclaimer) carries no answerable content."""
    return section.start == 0 and not _SECTION_ID_RE.match(section.heading or "") and len(body) < 400


def _split_paragraphs(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Pack paragraphs into chunks of at most ``max_chars``, carrying the tail of the
    previous chunk over as overlap so a sentence is never cut off from its context."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars or not current:
            current = candidate
            continue
        parts.append(current)
        overlap = current[-overlap_chars:]
        # Start the overlap at a word boundary.
        overlap = overlap[overlap.find(" ") + 1 :] if " " in overlap else overlap
        current = f"{overlap}\n\n{paragraph}" if overlap_chars else paragraph
    if current:
        parts.append(current)
    return parts
