"""Page-aware PDF text extraction and chunking for retrieval."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import multiprocessing
from pathlib import Path
from queue import Empty
import re
import time
import unicodedata

import pymupdf
import pymupdf4llm

from verbatim import config
from verbatim.downloads import DownloadedPaper, file_sha256

CHUNKER_VERSION = "8"

BODY = "body"
REFERENCES = "references"
TABLE = "table"

# Layout-aware extraction makes the bibliography heading its own section.
REFERENCES_TITLE = re.compile(
    r"^(?:\d+\s*\.?\s*)?(?:references|bibliography|works cited|literature cited)\s*$",
    re.I,
)
MARKDOWN_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")
# Emphasis, inline HTML, and link syntax; the words inside are kept.
MARKDOWN_MARKUP = re.compile(r"\*{1,3}|_+|`+|<[^>]{1,40}>|^\s*[-*+]\s+", re.M)
# Font-size heading detection sometimes swallows an author block into the
# title; a real section heading is short.
MAX_HEADING_LENGTH = 80
# Layout extraction is C code with no internal limit; one paper once ran for
# 46 minutes. Set by VERBATIM_EXTRACTION_TIMEOUT.
EXTRACTION_TIMEOUT_SECONDS = config.EXTRACTION_TIMEOUT
# How often a waiting extraction checks that its worker is still alive.
POLL_SECONDS = 0.5
# PDF text extraction can emit lone surrogates, which are valid Python str but
# cannot be encoded as UTF-8. They reach neither SQLite nor the JSONL export.
SURROGATES = re.compile(r"[\ud800-\udfff]")


def chunker_fingerprint(chunk_size: int, overlap: int) -> str:
    """The chunker version and parameters; a change to either means re-chunking."""
    return f"{CHUNKER_VERSION}:{chunk_size}:{overlap}"


@dataclass(frozen=True)
class TextSpan:
    """A chunk of page text together with its offsets in that normalized page."""

    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class Section:
    """One heading's worth of text and its offset in the page. Tables are their own sections."""

    path: tuple[str, ...]
    text: str
    char_start: int
    kind: str = BODY


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage with enough data to cite its original paper."""

    chunk_id: str
    arxiv_id: str
    source_url: str
    license_url: str | None
    # Checksum of the source this passage was read from, PDF or HTML.
    source_sha256: str
    # A page for a PDF; a section ordinal for HTML, which has no pages.
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    chunker_version: str
    kind: str = BODY
    section_path: str = ""
    # "pdf" or "html", which says how to read page_number.
    source_kind: str = "pdf"

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
        try:
            pages = list(page_extractor(pdf_path))
        except ExtractionFailed as error:
            # Skip the paper rather than fail the batch. It is still marked chunked,
            # so it is not retried forever.
            print(f"  ! {download.arxiv_id}: {error}", flush=True)
            continue
        paper_chunks: list[Chunk] = []
        heading_stack: list[tuple[int, str]] = []
        for page_number, page_text in enumerate(pages, start=1):
            sections, heading_stack = parse_markdown_sections(page_text, heading_stack)
            chunk_index = 0
            for section in sections:
                for span in chunk_page(section.text, chunk_size=chunk_size, overlap=overlap):
                    paper_chunks.append(
                        Chunk(
                            chunk_id=_chunk_id(download, page_number, chunk_index, fingerprint),
                            arxiv_id=download.arxiv_id,
                            source_url=download.source_url,
                            license_url=download.license_url,
                            source_sha256=download.sha256,
                            page_number=page_number,
                            chunk_index=chunk_index,
                            # Offsets stay relative to the page, not the section,
                            # so a citation still addresses the page a reader sees.
                            char_start=section.char_start + span.char_start,
                            char_end=section.char_start + span.char_end,
                            text=span.text,
                            chunker_version=fingerprint,
                            section_path=" > ".join(section.path),
                        )
                    )
                    chunk_index += 1
        chunks.extend(mark_references(paper_chunks))
    return chunks


def mark_references(paper_chunks: list[Chunk]) -> list[Chunk]:
    """Label the chunks inside a paper's bibliography.

    Only chunks whose own section heading is "References" or a synonym are
    marked, so appendices after the bibliography stay searchable.
    """
    return [
        replace(chunk, kind=REFERENCES) if is_bibliography(chunk.section_path) else chunk
        for chunk in paper_chunks
    ]


def is_bibliography(section_path: str) -> bool:
    """Whether a section path names a bibliography at any level."""
    if not section_path:
        return False
    return any(REFERENCES_TITLE.match(part.strip()) for part in section_path.split(" > "))


class ExtractionFailed(RuntimeError):
    """Layout extraction produced nothing for this paper."""


class ExtractionTimeout(ExtractionFailed):
    """Layout extraction did not finish within the time allowed."""


def _extract_into(queue: "multiprocessing.Queue[list[str]]", path: str) -> None:
    """Run layout extraction in a child process, so it can be abandoned."""
    pages = pymupdf4llm.to_markdown(path, page_chunks=True, show_progress=False)
    queue.put([page.get("text") or "" for page in pages])


def extract_pages_from_pdf(
    pdf_path: Path, *, timeout: float = EXTRACTION_TIMEOUT_SECONDS
) -> list[str]:
    """Return Markdown for every page of a text-based PDF.

    Layout extraction runs in a child process so a PDF that never finishes can
    be killed at `timeout`. Scanned PDFs yield nothing; there is no OCR.
    """
    context = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = context.Queue()
    worker = context.Process(target=_extract_into, args=(queue, str(pdf_path)), daemon=True)
    worker.start()
    deadline = time.monotonic() + timeout
    try:
        # Poll, so a child that crashes is reported at once rather than at the timeout.
        while True:
            try:
                return queue.get(timeout=max(min(POLL_SECONDS, deadline - time.monotonic()), 0))
            except Empty:
                pass
            if not worker.is_alive():
                try:
                    # A child that answered and then exited has flushed its
                    # answer; take it rather than calling that a crash.
                    return queue.get(timeout=POLL_SECONDS)
                except Empty:
                    raise ExtractionFailed(
                        f"Layout extraction of {pdf_path.name} exited with code "
                        f"{worker.exitcode} and produced nothing"
                    ) from None
            if time.monotonic() >= deadline:
                raise ExtractionTimeout(
                    f"Layout extraction of {pdf_path.name} exceeded {timeout:.0f}s"
                )
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join(5)


def extract_plain_pages_from_pdf(pdf_path: Path) -> list[str]:
    """Raw per-page text, without layout reconstruction."""
    with pymupdf.open(pdf_path) as document:
        return [page.get_text("text") for page in document]


def parse_markdown_sections(
    page_markdown: str, heading_stack: list[tuple[int, str]]
) -> tuple[list[Section], list[tuple[int, str]]]:
    """Split one page into sections, carrying the heading stack across pages.

    A chunk never straddles two sections, so each knows where in the paper it is.
    """
    stack = list(heading_stack)
    sections: list[Section] = []
    body: list[str] = []
    offset = 0

    def flush() -> None:
        nonlocal body, offset
        text = clean_markdown_text("\n".join(body))
        if text:
            sections.append(Section(path=tuple(title for _, title in stack), text=text, char_start=offset))
            offset += len(text) + 1  # +1 for the separator between sections
        body = []

    for line in page_markdown.splitlines():
        heading = MARKDOWN_HEADING.match(line)
        if heading is None:
            body.append(line)
            continue
        flush()
        level = len(heading.group(1))
        title = clean_markdown_text(heading.group(2))[:MAX_HEADING_LENGTH].strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        if title:
            stack.append((level, title))
    flush()
    return sections, stack


def clean_markdown_text(text: str) -> str:
    """Drop Markdown markup and normalize, so stored text matches what is quoted."""
    return normalize_page_text(MARKDOWN_MARKUP.sub("", text))


def normalize_page_text(text: str) -> str:
    """Fold ligatures such as "ﬂ" to "fl", drop lone surrogates, collapse whitespace."""
    without_surrogates = SURROGATES.sub("", text)
    return " ".join(unicodedata.normalize("NFKC", without_surrogates).split())


def chunk_page(text: str, *, chunk_size: int, overlap: int) -> list[TextSpan]:
    """Chunk one page, keeping each chunk's character offsets in the normalized page."""
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
        # A token longer than chunk_size is split mid-word rather than dropped.
        start = min(next_start, end)
    return spans


def _chunk_id(
    download: DownloadedPaper, page_number: int, chunk_index: int, fingerprint: str
) -> str:
    identifier = (
        f"{download.arxiv_id}:{download.sha256}:{page_number}:{chunk_index}:{fingerprint}"
    )
    return sha256(identifier.encode("utf-8")).hexdigest()[:20]
