"""Page-aware PDF text extraction and chunking for retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterable
import unicodedata

import pymupdf

from verbatim.downloads import DownloadedPaper, file_sha256


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
    text: str

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
    chunks: list[Chunk] = []

    for download in downloads:
        pdf_path = Path(download.local_path)
        actual_checksum = file_sha256(pdf_path)
        if actual_checksum != download.sha256:
            raise ValueError(
                f"Checksum mismatch for {download.arxiv_id}: manifest={download.sha256}, file={actual_checksum}"
            )
        for page_number, page_text in enumerate(page_extractor(pdf_path), start=1):
            for chunk_index, text in enumerate(chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)):
                chunk_id = _chunk_id(download, page_number, chunk_index)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        arxiv_id=download.arxiv_id,
                        source_url=download.source_url,
                        license_url=download.license_url,
                        pdf_sha256=download.sha256,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        text=text,
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
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    normalized = normalize_page_text(text)
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            word_boundary = normalized.rfind(" ", start + 1, end + 1)
            if word_boundary > start:
                end = word_boundary
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        next_start = max(end - overlap, start + 1)
        if next_start > 0 and next_start < len(normalized) and normalized[next_start - 1] != " ":
            next_space = normalized.find(" ", next_start)
            next_start = len(normalized) if next_space == -1 else next_space + 1
        start = next_start
    return chunks


def _chunk_id(download: DownloadedPaper, page_number: int, chunk_index: int) -> str:
    identifier = f"{download.arxiv_id}:{download.sha256}:{page_number}:{chunk_index}"
    return sha256(identifier.encode("utf-8")).hexdigest()[:20]
