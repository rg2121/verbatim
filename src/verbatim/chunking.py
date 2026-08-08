"""Page-aware PDF text extraction and chunking for retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterable
import unicodedata

import pymupdf

from verbatim.downloads import DownloadedPaper, file_sha256

CHUNKER_VERSION = "2"


def chunker_fingerprint(chunk_size: int, overlap: int) -> str:
    """Identify the algorithm *and* the parameters that produced a chunk.

    Changing either invalidates existing chunks, so both belong in the value
    that decides whether a PDF needs re-chunking.
    """
    return f"{CHUNKER_VERSION}:{chunk_size}:{overlap}"


@dataclass(frozen=True)
class TextSpan:
    """A chunk of page text together with its offsets in that normalized page."""

    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage with enough data to cite its original paper."""

    chunk_id: str
    arxiv_id: str
    source_url: str
    license_url: str | None
    pdf_sha256: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    chunker_version: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_pdf_chunks(
    downloads: list[DownloadedPaper],
    *,
    chunk_size: int = 1_200,
    overlap: int = 200,
    extract_pages: Callable[[Path], Iterable[str]] | None = None,
) -> list[Chunk]:
    """Extract and split every page referenced in a download manifest."""
    page_extractor = extract_pages or extract_pages_from_pdf
    fingerprint = chunker_fingerprint(chunk_size, overlap)
    chunks: list[Chunk] = []

    for download in downloads:
        pdf_path = Path(download.local_path)
        actual_checksum = file_sha256(pdf_path)
        if actual_checksum != download.sha256:
            raise ValueError(
                f"Checksum mismatch for {download.arxiv_id}: manifest={download.sha256}, file={actual_checksum}"
            )
        for page_number, page_text in enumerate(page_extractor(pdf_path), start=1):
            spans = chunk_page(page_text, chunk_size=chunk_size, overlap=overlap)
            for chunk_index, span in enumerate(spans):
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(download, page_number, chunk_index, fingerprint),
                        arxiv_id=download.arxiv_id,
                        source_url=download.source_url,
                        license_url=download.license_url,
                        pdf_sha256=download.sha256,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        char_start=span.char_start,
                        char_end=span.char_end,
                        text=span.text,
                        chunker_version=fingerprint,
                    )
                )
    return chunks


def extract_pages_from_pdf(pdf_path: Path) -> list[str]:
    """Return text for every page of a text-based PDF.

    Scanned PDFs need OCR, which will be a separate later stage.
    """
    with pymupdf.open(pdf_path) as document:
        return [page.get_text("text") for page in document]


def normalize_page_text(text: str) -> str:
    """Fold typographic ligatures and collapse whitespace.

    PDF extraction yields presentation glyphs such as U+FB02 "ﬂ", which break
    both keyword search and embedding similarity against ordinary "fl".
    """
    return " ".join(unicodedata.normalize("NFKC", text).split())


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Create normalized, overlapping chunks without splitting words when possible."""
    return [span.text for span in chunk_page(text, chunk_size=chunk_size, overlap=overlap)]


def chunk_page(text: str, *, chunk_size: int, overlap: int) -> list[TextSpan]:
    """Chunk one page, recording where each chunk sits in the normalized page.

    The offsets let a citation point at a character range rather than only at a
    page number.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    normalized = normalize_page_text(text)
    if not normalized:
        return []

    spans: list[TextSpan] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            word_boundary = normalized.rfind(" ", start + 1, end + 1)
            if word_boundary > start:
                end = word_boundary
        raw = normalized[start:end]
        stripped = raw.strip()
        if stripped:
            offset = start + (len(raw) - len(raw.lstrip()))
            spans.append(TextSpan(offset, offset + len(stripped), stripped))
        if end == len(normalized):
            break
        next_start = max(end - overlap, start + 1)
        if next_start > 0 and next_start < len(normalized) and normalized[next_start - 1] != " ":
            next_space = normalized.find(" ", next_start)
            next_start = len(normalized) if next_space == -1 else next_space + 1
        # A token longer than chunk_size has no interior space to advance to, so
        # the search above can land beyond `end`.  Clamping keeps the spans
        # contiguous; such a token is split mid-word rather than dropped.
        start = min(next_start, end)
    return spans


def _chunk_id(
    download: DownloadedPaper, page_number: int, chunk_index: int, fingerprint: str
) -> str:
    identifier = (
        f"{download.arxiv_id}:{download.sha256}:{page_number}:{chunk_index}:{fingerprint}"
    )
    return sha256(identifier.encode("utf-8")).hexdigest()[:20]
