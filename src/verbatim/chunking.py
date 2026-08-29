"""Page-aware PDF text extraction and chunking for retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
import re
from typing import Callable, Iterable
import unicodedata

import pymupdf
import pymupdf4llm

from verbatim.downloads import DownloadedPaper, file_sha256

CHUNKER_VERSION = "8"

BODY = "body"
REFERENCES = "references"
TABLE = "table"

# Layout-aware extraction turns the bibliography heading into a section, so a
# whole-title match is both simpler and far more reliable than the earlier
# approach of scanning prose for the word "References".
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
# PDF text extraction can emit lone surrogates, which are valid Python str but
# cannot be encoded as UTF-8. They reach neither SQLite nor the JSONL export.
SURROGATES = re.compile(r"[\ud800-\udfff]")


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
class Section:
    """One heading's worth of text, and where it sits in the page.

    A table is carried as its own section rather than as prose inside one, so
    that its rows survive chunking intact and a reader can be told that what
    they are looking at is a table.
    """

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
    # The checksum of whatever this passage was read from - a PDF or an arXiv
    # HTML rendering. Named for the role, not one of the two forms.
    source_sha256: str
    # A page for a PDF, a section ordinal for HTML, which has none.
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    chunker_version: str
    kind: str = BODY
    section_path: str = ""
    # "pdf" or "html". page_number means a page for a PDF and a section
    # ordinal for HTML, which has no pages, so a citation must know which.
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
        paper_chunks: list[Chunk] = []
        heading_stack: list[tuple[int, str]] = []
        for page_number, page_text in enumerate(page_extractor(pdf_path), start=1):
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
    """Label the chunks that sit inside a paper's bibliography.

    A chunk belongs to the bibliography when one of its own section headings
    *is* "References" (or a synonym), so a section called "Reference-Assisted
    Human Audit" is not swept up.

    Only the bibliography itself is marked. Marking everything that follows it
    was wrong twice over: papers often place appendices and supplementary
    material after the references, which hid roughly 72,000 chunks of real
    content from search, and the positional guard that came with it missed
    bibliographies sitting mid-document in papers with long appendices.
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


def extract_pages_from_pdf(pdf_path: Path) -> list[str]:
    """Return Markdown for every page of a text-based PDF.

    pymupdf4llm reconstructs reading order and marks headings by font size,
    which plain text extraction cannot express.  It costs roughly 50x the time
    of raw extraction, which is affordable because chunking is incremental.

    Scanned PDFs need OCR, which will be a separate later stage.
    """
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=False)
    return [page.get("text") or "" for page in pages]


def extract_plain_pages_from_pdf(pdf_path: Path) -> list[str]:
    """Raw per-page text, without layout reconstruction.

    Kept as the fast path for callers that do not need structure.
    """
    with pymupdf.open(pdf_path) as document:
        return [page.get_text("text") for page in document]


def parse_markdown_sections(
    page_markdown: str, heading_stack: list[tuple[int, str]]
) -> tuple[list[Section], list[tuple[int, str]]]:
    """Split one page into sections, carrying the heading stack across pages.

    A section runs from one heading to the next, so a chunk never straddles a
    section boundary and every chunk knows which part of the paper it came
    from.  The stack is returned because a section usually continues onto the
    following page.
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
        title = strip_markdown(heading.group(2))[:MAX_HEADING_LENGTH].strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        if title:
            stack.append((level, title))
    flush()
    return sections, stack


def clean_markdown_text(text: str) -> str:
    """Drop Markdown markup, then normalize, keeping only the words.

    Body text is stripped as well as headings: a stored passage containing
    "**flux**" would fail citation verification against a model that quoted
    "flux", and the markup is noise to both keyword and vector search.
    """
    return normalize_page_text(MARKDOWN_MARKUP.sub("", text))


def strip_markdown(text: str) -> str:
    """Remove Markdown emphasis and inline markup from a single line."""
    return clean_markdown_text(text)


def normalize_page_text(text: str) -> str:
    """Fold typographic ligatures and collapse whitespace.

    PDF extraction yields presentation glyphs such as U+FB02 "ﬂ", which break
    both keyword search and embedding similarity against ordinary "fl", and
    occasionally lone surrogates, which cannot be encoded as UTF-8 at all. One
    such character used to abort a whole chunking batch when the insert failed,
    discarding every paper processed alongside it.
    """
    without_surrogates = SURROGATES.sub("", text)
    return " ".join(unicodedata.normalize("NFKC", without_surrogates).split())


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
