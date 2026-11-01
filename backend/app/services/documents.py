"""Text extraction and chunking for knowledge-base ingestion."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from app.models.knowledge import DocumentType


class DocumentExtractionError(ValueError):
    pass


@dataclass(slots=True)
class ExtractedDocument:
    text: str
    page_count: int | None = None
    document_type: DocumentType = DocumentType.OTHER


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
        elif tag in {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def detect_document_type(filename: str, mime_type: str | None = None) -> DocumentType:
    suffix = Path(filename).suffix.lower()
    return {
        ".pdf": DocumentType.PDF,
        ".doc": DocumentType.WORD,
        ".docx": DocumentType.WORD,
        ".md": DocumentType.MARKDOWN,
        ".markdown": DocumentType.MARKDOWN,
        ".txt": DocumentType.TEXT,
        ".csv": DocumentType.CSV,
        ".html": DocumentType.HTML,
        ".htm": DocumentType.HTML,
        ".json": DocumentType.JSON,
    }.get(suffix, DocumentType.OTHER)


def extract_document(path: Path, filename: str) -> ExtractedDocument:
    if not path.exists():
        raise DocumentExtractionError("Uploaded file was not found")
    document_type = detect_document_type(filename)
    if document_type == DocumentType.PDF:
        return _extract_pdf(path)
    if document_type == DocumentType.WORD:
        return _extract_docx(path)
    if document_type == DocumentType.HTML:
        return _extract_html(path)
    if document_type == DocumentType.CSV:
        return _extract_csv(path)
    if document_type == DocumentType.JSON:
        return _extract_json(path)
    if document_type in {DocumentType.TEXT, DocumentType.MARKDOWN}:
        return ExtractedDocument(
            text=_read_text(path), document_type=document_type
        )
    raise DocumentExtractionError(
        f"Unsupported document type '{Path(filename).suffix or 'unknown'}'. "
        "Supported: PDF, DOCX, TXT, MD, CSV, HTML, JSON."
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_pdf(path: Path) -> ExtractedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise DocumentExtractionError("pypdf is required for PDF ingestion") from exc
    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        raise DocumentExtractionError(f"Could not read PDF: {exc}") from exc
    return ExtractedDocument(
        text="\n\n".join(pages),
        page_count=len(pages),
        document_type=DocumentType.PDF,
    )


def _extract_docx(path: Path) -> ExtractedDocument:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise DocumentExtractionError("python-docx is required for DOCX ingestion") from exc
    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise DocumentExtractionError(f"Could not read DOCX: {exc}") from exc
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return ExtractedDocument(text="\n".join(parts), document_type=DocumentType.WORD)


def _extract_html(path: Path) -> ExtractedDocument:
    parser = _HTMLTextExtractor()
    parser.feed(_read_text(path))
    return ExtractedDocument(text=parser.text(), document_type=DocumentType.HTML)


def _extract_csv(path: Path) -> ExtractedDocument:
    text = _read_text(path)
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as exc:
        raise DocumentExtractionError(f"Could not read CSV: {exc}") from exc
    lines = [" | ".join(cell.strip() for cell in row) for row in rows if row]
    return ExtractedDocument(text="\n".join(lines), document_type=DocumentType.CSV)


def _extract_json(path: Path) -> ExtractedDocument:
    text = _read_text(path)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DocumentExtractionError(f"Could not read JSON: {exc}") from exc
    return ExtractedDocument(
        text=json.dumps(payload, indent=2, ensure_ascii=False),
        document_type=DocumentType.JSON,
    )


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks on natural boundaries.

    ``chunk_size`` and ``chunk_overlap`` are measured in characters, which is
    model-agnostic and keeps ingestion deterministic across embedding models.
    """

    text = normalize_text(text)
    if not text:
        return []
    chunk_size = max(100, chunk_size)
    overlap = max(0, min(chunk_overlap, chunk_size // 2))
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in sentences:
            if len(sentence) > chunk_size:
                if current:
                    units.append(current)
                    current = ""
                for offset in range(0, len(sentence), chunk_size - overlap):
                    units.append(sentence[offset : offset + chunk_size])
                continue
            candidate = f"{current} {sentence}".strip()
            if len(candidate) > chunk_size and current:
                units.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            units.append(current)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) > chunk_size and current:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{unit}".strip()
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())

    # Guarantee the configured maximum even when overlap causes growth.
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final.append(chunk)
            continue
        for offset in range(0, len(chunk), chunk_size - overlap):
            piece = chunk[offset : offset + chunk_size].strip()
            if piece:
                final.append(piece)
    return final


def keywords(query: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9_]{3,}", query.lower())
        if token not in _STOPWORDS
    ][:12]


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "what",
    "when",
    "where",
    "which",
    "who",
    "how",
    "why",
    "this",
    "that",
    "from",
    "are",
    "was",
    "were",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "about",
    "into",
    "your",
    "you",
}


__all__ = [
    "DocumentExtractionError",
    "ExtractedDocument",
    "chunk_text",
    "detect_document_type",
    "extract_document",
    "keywords",
    "normalize_text",
]
