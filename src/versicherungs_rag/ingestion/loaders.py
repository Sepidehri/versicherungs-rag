"""Load Markdown, plain-text and PDF files into :class:`Document` objects."""

from __future__ import annotations

import re
from pathlib import Path

from versicherungs_rag.schema import Document

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return _load_text(path)
    if suffix == ".pdf":
        return _load_pdf(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def load_directory(directory: Path) -> list[Document]:
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No supported documents (.md, .txt, .pdf) in {directory}")
    return [load_document(p) for p in paths]


def _load_text(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = _short_title(title_match.group(1)) if title_match else _title_from_filename(path)
    return Document(doc_id=path.stem, title=title, text=text, source=str(path))


def _load_pdf(path: Path) -> Document:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    offsets: list[int] = []
    position = 0
    for page in reader.pages:
        page_text = _clean_pdf_text(page.extract_text() or "")
        offsets.append(position)
        parts.append(page_text)
        position += len(page_text) + 2  # account for the "\n\n" separator
    meta_title = reader.metadata.title if reader.metadata else None
    title = _short_title(meta_title) if meta_title else _title_from_filename(path)
    return Document(
        doc_id=path.stem,
        title=title,
        text="\n\n".join(parts),
        source=str(path),
        page_offsets=tuple(offsets),
    )


def _clean_pdf_text(text: str) -> str:
    # Re-join words hyphenated across line breaks ("Versicherungs-\nnehmer").
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _short_title(title: str) -> str:
    """'Allgemeine Bedingungen für die Haushaltsversicherung (AVB Haushalt 2026)' -> 'AVB Haushalt 2026'."""
    title = title.strip()
    bracket = re.search(r"\(([^)]+)\)\s*$", title)
    return bracket.group(1) if bracket else title


def _title_from_filename(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").title()
