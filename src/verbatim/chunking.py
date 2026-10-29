"""Page-aware PDF text extraction and chunking for retrieval."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
import functools
from hashlib import sha256
import math
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
    """The chunker version, parameters and way of cutting; a change to any means re-chunking."""
    fingerprint = f"{CHUNKER_VERSION}:{chunk_size}:{overlap}"
    return fingerprint if config.CHUNKER == "windows" else f"{fingerprint}:{config.CHUNKER}"


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
    cut = section_chunker(chunk_size, overlap)
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
                for span in cut(section.text):
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
    _check_settings(chunk_size, overlap)

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


def _check_settings(chunk_size: int, overlap: int) -> None:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")


CHUNKERS = ("windows", "sentences", "semantic")
# How much of a sentence semantic cutting embeds: enough to tell its subject. A
# run-on "sentence", such as an author list, pads its whole batch; capped, 10
# papers cut a third faster, with 446 of 455 sections cut the same.
SUBJECT_CHARS = 400

# A sentence ends at . ! or ?, perhaps inside a closing bracket or quote, before
# a space and a capital, a digit or an opening bracket.
SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?= [\"'(\[]?[A-Z0-9])")
# Words a full stop follows without ending the sentence, as in "et al. 2017".
ABBREVIATIONS = frozenset({
    "al", "fig", "figs", "eq", "eqs", "sec", "secs", "ref", "refs", "tab", "vs",
    "cf", "approx", "resp", "vol", "pp", "ch", "chap", "thm", "def", "prop",
    "lem", "alg", "app", "appx", "dr", "mr", "ms", "prof", "st", "inc", "ltd",
})


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Where each sentence of normalized text starts and ends.

    A rule, not a model: it misses some ends ("in the U.S. The"), which only
    makes a passage end a sentence later.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_END.finditer(text):
        word = text[text.rfind(" ", 0, match.start()) + 1 : match.start()].lstrip("(\"'[")
        # "e.g." and "i.e." carry an inner full stop; "A. Smith" is an initial.
        if word.lower() in ABBREVIATIONS or "." in word or (len(word) == 1 and word.isupper()):
            continue
        spans.append((start, match.end()))
        start = match.end() + 1  # normalized text has one space between sentences
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def chunk_sentences(text: str, *, chunk_size: int, overlap: int) -> list[TextSpan]:
    """Chunk one page into whole sentences, as many as fit; the overlap repeats whole sentences."""
    _check_settings(chunk_size, overlap)
    normalized = normalize_page_text(text)
    return _pack_sentences(
        normalized, sentence_spans(normalized), chunk_size=chunk_size, overlap=overlap,
        choose=lambda first, fits: fits[-1],
    )


def chunk_semantic(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    embed: Callable[[list[str]], Sequence[Sequence[float]]],
) -> list[TextSpan]:
    """Chunk one page at changes of subject.

    Each passage is at least half of `chunk_size`, and ends at the sentence
    boundary in that room where the sentences either side are least alike.
    """
    _check_settings(chunk_size, overlap)
    normalized = normalize_page_text(text)
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        # One passage either way, so nothing is embedded.
        return [TextSpan(0, len(normalized), normalized)]
    sentences = sentence_spans(normalized)
    alike = _neighbour_similarity(
        embed([normalized[start:end][:SUBJECT_CHARS] for start, end in sentences])
    )

    def choose(first: int, fits: list[int]) -> int:
        if fits[-1] == len(sentences) - 1:
            return fits[-1]  # the rest fits in one passage
        start = sentences[first][0]
        roomy = [last for last in fits if sentences[last][1] - start >= chunk_size // 2] or fits[-1:]
        # On a tie the later boundary wins, as a window would.
        return min(roomy, key=lambda last: (alike[last], -last))

    return _pack_sentences(
        normalized, sentences, chunk_size=chunk_size, overlap=overlap, choose=choose
    )


def _pack_sentences(
    normalized: str,
    sentences: list[tuple[int, int]],
    *,
    chunk_size: int,
    overlap: int,
    choose: Callable[[int, list[int]], int],
) -> list[TextSpan]:
    """Group sentences into passages; `choose` picks each passage's last sentence from those that fit."""
    spans: list[TextSpan] = []
    first = 0
    while first < len(sentences):
        start, first_end = sentences[first]
        if first_end - start > chunk_size:
            # A sentence longer than a passage is cut at words, as windows cut it.
            spans.extend(
                TextSpan(start + span.char_start, start + span.char_end, span.text)
                for span in chunk_page(normalized[start:first_end], chunk_size=chunk_size, overlap=overlap)
            )
            first += 1
            continue
        fits = [first]
        while fits[-1] + 1 < len(sentences) and sentences[fits[-1] + 1][1] - start <= chunk_size:
            fits.append(fits[-1] + 1)
        last = choose(first, fits)
        end = sentences[last][1]
        spans.append(TextSpan(start, end, normalized[start:end]))
        if last + 1 == len(sentences):
            break
        # The next passage opens with the trailing sentences that fit in the overlap.
        following = last + 1
        while following - 1 > first and end - sentences[following - 1][0] <= overlap:
            following -= 1
        first = following
    return spans


def _neighbour_similarity(vectors: Sequence[Sequence[float]]) -> list[float]:
    """How alike the text is either side of each sentence boundary.

    Two sentences a side, averaged, so one short sentence does not decide a cut.
    """
    def mean(rows: Sequence[Sequence[float]]) -> list[float]:
        total = [sum(column) for column in zip(*rows)]
        length = math.sqrt(sum(value * value for value in total)) or 1.0
        return [value / length for value in total]

    return [
        sum(a * b for a, b in zip(mean(vectors[max(i - 1, 0) : i + 1]), mean(vectors[i + 1 : i + 3])))
        for i in range(len(vectors) - 1)
    ]


def section_chunker(
    chunk_size: int,
    overlap: int,
    *,
    embed: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
) -> Callable[[str], list[TextSpan]]:
    """How prose is cut, by VERBATIM_CHUNKER: windows, whole sentences, or at changes of subject."""
    strategy = config.CHUNKER
    if strategy == "windows":
        return functools.partial(chunk_page, chunk_size=chunk_size, overlap=overlap)
    if strategy == "sentences":
        return functools.partial(chunk_sentences, chunk_size=chunk_size, overlap=overlap)
    if strategy == "semantic":
        return functools.partial(
            chunk_semantic, chunk_size=chunk_size, overlap=overlap,
            embed=embed or _sentence_embedder(),
        )
    raise ValueError(f"VERBATIM_CHUNKER must be {', '.join(CHUNKERS)}, got {strategy!r}")


@functools.cache
def _sentence_embedder() -> Callable[[list[str]], Sequence[Sequence[float]]]:
    """The passage embedder, reused to compare sentences, loaded only for semantic cuts."""
    from verbatim.embedding import FastEmbedEmbedder  # embedding imports this module

    return FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL).embed_passages


def _chunk_id(
    download: DownloadedPaper, page_number: int, chunk_index: int, fingerprint: str
) -> str:
    identifier = (
        f"{download.arxiv_id}:{download.sha256}:{page_number}:{chunk_index}:{fingerprint}"
    )
    return sha256(identifier.encode("utf-8")).hexdigest()[:20]
