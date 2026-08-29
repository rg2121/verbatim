"""Reading a paper from arXiv's HTML rather than its PDF.

arXiv renders every TeX submission to HTML, and that rendering is made from the
author's source rather than from a printed page. Measured on this corpus, 96%
of papers have one. It is better input than a PDF in three ways:

- **Maths survives.** Every <math> element carries the original LaTeX in its
  `alttext`, so an equation stays an equation instead of becoming the character
  soup a PDF extractor produces.
- **Structure is real.** Headings come from the author's sectioning commands,
  not from guessing at font sizes.
- **No extraction damage.** None of the ligatures, lone surrogates or
  reading-order problems that PDF text carries.

What it loses is page numbers: HTML has none, so an HTML-sourced passage is
located by section instead. `Chunk.page_number` therefore means "page" for a
PDF and "section ordinal" for HTML, which is why chunks record which they came
from.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Callable
from html.parser import HTMLParser
from urllib.error import HTTPError
import re

from hashlib import sha256
from pathlib import Path

from verbatim.chunking import (
    BODY,
    TABLE,
    Chunk,
    Section,
    chunk_page,
    chunker_fingerprint,
    mark_references,
    normalize_page_text,
)
from verbatim.downloads import file_sha256
from verbatim.fetching import fetch

ARXIV_HTML_URL = "https://arxiv.org/html"

HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Chrome and navigation arXiv wraps around the paper.
SKIP_CONTENT = {"script", "style", "nav", "header", "footer", "button", "select",
                "option", "label", "form", "dialog", "template", "noscript"}
# arXiv injects a "Report GitHub Issue" dialog whose prose reads like content.
# Matched on class or id, since it carries no distinguishing tag.
SKIP_MARKED = ("modal", "ltx_page_navbar", "ltx_align_center ltx_flex")
# These have no end tag, so a skip started on one would never be closed.
VOID_ELEMENTS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
                 "link", "meta", "param", "source", "track", "wbr"}
# Inline elements are joined with spaces, which leaves "[ 6 ] ." where the
# paper reads "[6].". Applied to prose only; maths is kept exactly as written.
SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:!?%)\]}])")
SPACE_AFTER_OPENING = re.compile(r"([(\[{])\s+")
MATHS_SPAN = re.compile(r"\$[^$]*\$")
# Long enough to be prose, short enough to exclude a whole swallowed document.
MAX_HEADING_LENGTH = 120
# LaTeXML lays display equations out as tables too, and those are already read
# through each <math> element's alttext. Only a real tabular is a table.
TABULAR_CLASS = "ltx_tabular"
# A cell holding more than this is prose in a layout table, not tabular data.
MAX_CELL_LENGTH = 200
# A span wider than any real table, so a malformed attribute cannot make the
# parser build an enormous row.
MAX_SPAN = 32
WHITESPACE = re.compile(r"\s+")


@dataclass
class _Block:
    level: int
    title: str


class ArxivHTMLParser(HTMLParser):
    """Collect headings and text, keeping maths as the LaTeX it came from."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[Section] = []
        self._stack: list[_Block] = []
        self._body: list[str] = []
        self._heading: list[str] | None = None
        self._heading_level = 0
        self._skip_tag: str | None = None
        self._skip_nesting = 0
        self._math_depth = 0
        self._offset = 0
        self._table: list[list[str]] | None = None
        self._table_depth = 0
        self._row: list[tuple[list[str], int, int]] | None = None
        self._cell: list[str] | None = None
        self._cell_spans = (1, 1)
        # column index -> (text, rows still to fill), for cells spanning down
        self._pending_spans: dict[int, tuple[str, int]] = {}

    # -- structure -------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_tag is not None:
            # Nothing inside a skipped region is read, not even its maths. A
            # nested element of the same name is counted so that its end tag
            # does not close the region early.
            if tag == self._skip_tag and tag not in VOID_ELEMENTS:
                self._skip_nesting += 1
            return
        attributes = dict(attrs)
        marker = f"{attributes.get('class') or ''} {attributes.get('id') or ''}"
        if tag in SKIP_CONTENT or any(m in marker for m in SKIP_MARKED):
            if tag in VOID_ELEMENTS:
                return  # no content to skip, and no end tag to wait for
            self._skip_tag = tag
            self._skip_nesting = 1
            return
        if tag == "table" and TABULAR_CLASS in (attributes.get("class") or ""):
            self._table, self._table_depth = [], 1
            return
        if self._table is not None:
            if tag == "table":
                self._table_depth += 1
            elif tag == "tr":
                self._row = []
            elif tag in ("td", "th"):
                self._cell = []
                self._cell_spans = (
                    _span(attributes.get("rowspan")), _span(attributes.get("colspan"))
                )
        if tag == "math":
            # Take the LaTeX and ignore the rendered glyphs inside, which would
            # otherwise arrive as unreadable fragments of the same equation.
            latex = (attributes.get("alttext") or "").strip()
            if latex:
                self._emit(f"${latex}$")
            self._math_depth += 1
            return
        if self._math_depth:
            return
        if tag in HEADINGS:
            self._flush()
            self._heading = []
            self._heading_level = int(tag[1])

    def handle_endtag(self, tag: str) -> None:
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_nesting -= 1
                if self._skip_nesting <= 0:
                    self._skip_tag, self._skip_nesting = None, 0
            return
        if tag == "math":
            self._math_depth = max(0, self._math_depth - 1)
            return
        if self._table is not None:
            if tag in ("td", "th") and self._cell is not None:
                cell = normalize_page_text("".join(self._cell))[:MAX_CELL_LENGTH]
                rowspan, colspan = self._cell_spans
                self._cell, self._cell_spans = None, (1, 1)
                if self._row is not None:
                    self._row.append(([cell], rowspan, colspan))
                return
            if tag == "tr" and self._row is not None:
                row = self._assemble_row(self._row)
                self._row = None
                if any(cell for cell in row):
                    self._table.append(row)
                return
            if tag == "table":
                self._table_depth -= 1
                if self._table_depth <= 0:
                    self._finish_table()
                return
        if tag in HEADINGS and self._heading is not None:
            title = WHITESPACE.sub(" ", "".join(self._heading)).strip()[:MAX_HEADING_LENGTH]
            self._heading = None
            if title:
                while self._stack and self._stack[-1].level >= self._heading_level:
                    self._stack.pop()
                self._stack.append(_Block(self._heading_level, title))

    def handle_data(self, data: str) -> None:
        if self._skip_tag is not None or self._math_depth:
            return
        if self._heading is not None:
            self._heading.append(data)
        else:
            self._emit(data)

    # -- text ------------------------------------------------------------
    def _emit(self, text: str) -> None:
        if self._heading is not None:
            self._heading.append(text)
        elif self._cell is not None:
            # Text inside a table cell belongs to that cell, including the
            # LaTeX of any maths in it.
            self._cell.append(text)
        elif self._table is not None:
            # Between cells - alignment padding and stray whitespace. Dropping
            # it keeps the serialised table to its actual contents.
            return
        elif self._skip_tag is None:
            self._body.append(text)

    def _assemble_row(self, cells: list[tuple[list[str], int, int]]) -> list[str]:
        """Place a row's cells in their real columns, honouring spans.

        Nineteen per cent of the tables in this corpus use rowspan and 29% use
        colspan. Ignoring them does not merely lose formatting: the cells after
        a span shift left, so a continuation line lands in the first column and
        the table states something the paper does not.
        """
        row: list[str] = []
        queue = list(cells)
        column = 0
        while queue or column in self._pending_spans:
            carried = self._pending_spans.get(column)
            if carried is not None:
                text, remaining = carried
                row.append(text)
                if remaining <= 1:
                    del self._pending_spans[column]
                else:
                    self._pending_spans[column] = (text, remaining - 1)
                column += 1
                continue
            if not queue:
                break
            parts, rowspan, colspan = queue.pop(0)
            text = "".join(parts)
            row.append(text)
            # A spanned cell is written once and its remaining columns left
            # empty, so a term is not counted several times by search.
            row.extend([""] * (colspan - 1))
            if rowspan > 1:
                for offset in range(colspan):
                    self._pending_spans[column + offset] = (
                        text if offset == 0 else "", rowspan - 1
                    )
            column += colspan
        return row

    def _finish_table(self) -> None:
        """Close the current table and record it as a section of its own."""
        rows, self._table = self._table or [], None
        self._table_depth, self._row, self._cell = 0, None, None
        self._pending_spans = {}
        text = render_table(rows)
        if not text:
            return
        self._flush()  # the prose before the table is its own section
        self.sections.append(
            Section(
                path=tuple(block.title for block in self._stack),
                text=text,
                char_start=self._offset,
                kind=TABLE,
            )
        )
        self._offset += len(text) + 1

    def _flush(self) -> None:
        # Cells are normalised as they are read, so a completed table is not
        # passed through normalisation again: that would collapse the newlines
        # its rows are made of.
        text = tidy_spacing(normalize_page_text(" ".join(self._body)))
        self._body = []
        if not text:
            return
        self.sections.append(
            Section(
                path=tuple(block.title for block in self._stack),
                text=text,
                char_start=self._offset,
            )
        )
        self._offset += len(text) + 1

    def close(self) -> None:  # noqa: D102 - inherited
        super().close()
        self._flush()


def _span(value: str | None) -> int:
    """Read a rowspan or colspan attribute, refusing anything unusable."""
    try:
        return max(1, min(int(str(value)), MAX_SPAN))
    except (TypeError, ValueError):
        return 1


def render_table(rows: list[list[str]]) -> str:
    """Serialise a table as a Markdown pipe table.

    The rows are what a PDF extractor destroys: flattened, "Person Corp Time
    Position Musk SpaceX 2002 CEO" gives a reader no way to see that Musk,
    SpaceX, 2002 and CEO are one row. Pipes are used because a model reads them
    as a table without being told, and because they survive being quoted back
    for citation verification.
    """
    filled = [row for row in rows if any(cell.strip() for cell in row)]
    if not filled:
        return ""
    width = max(len(row) for row in filled)
    if width < 2:
        # A single column is a list, and reads better as one.
        return "\n".join(row[0] for row in filled if row)
    padded = [row + [""] * (width - len(row)) for row in filled]
    lines = ["| " + " | ".join(padded[0]) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
    return "\n".join(lines)


def tidy_spacing(text: str) -> str:
    """Close up the spaces left by joining inline elements, sparing maths.

    Every inline element is joined with a space, so a citation the paper prints
    as "[6]." arrives as "[ 6 ] .". Maths spans are copied through untouched:
    the LaTeX is the reason for reading HTML at all, and it is not ours to
    reformat.
    """
    pieces: list[str] = []
    end = 0
    for span in MATHS_SPAN.finditer(text):
        pieces.append(_tidy_prose(text[end : span.start()]))
        pieces.append(span.group(0))
        end = span.end()
    pieces.append(_tidy_prose(text[end:]))
    return "".join(pieces)


def _tidy_prose(text: str) -> str:
    return SPACE_AFTER_OPENING.sub(r"\1", SPACE_BEFORE_PUNCTUATION.sub(r"\1", text))


def parse_html_sections(html: str) -> list[Section]:
    """Turn one arXiv HTML page into the sections the chunker already expects."""
    parser = ArxivHTMLParser()
    parser.feed(html)
    parser.close()
    return parser.sections


@dataclass(frozen=True)
class HtmlSource:
    """An auditable record of one paper's HTML rendering, held locally.

    Mirrors DownloadedPaper: a chunk may only claim to come from bytes we have
    and can re-verify, whichever form those bytes take.
    """

    arxiv_id: str
    url: str
    local_path: str
    sha256: str
    byte_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# arXiv's LaTeXML conversion sometimes fails partway and the partial result is
# published anyway. Measured over 170 papers, the text of a healthy rendering
# runs from 0.66 to 4.55 times its PDF's (median 1.09 - LaTeX markup adds
# characters), while a truncated one measured 0.08. Nothing legitimate came
# near the threshold from either side.
MIN_COMPLETENESS = 0.5


def completeness(html_text_length: int, pdf_text_length: int) -> float:
    """How much of the PDF's text the rendering accounts for.

    A rendering covering almost none of its paper is worse than the PDF it
    would replace, so this decides whether it is used at all.
    """
    if pdf_text_length <= 0:
        return 1.0  # nothing to compare against; trust the rendering
    return html_text_length / pdf_text_length


def html_url_for(arxiv_id: str) -> str:
    return f"{ARXIV_HTML_URL}/{arxiv_id}"


def fetch_html(arxiv_id: str, *, get: Callable[[str], bytes] | None = None) -> bytes | None:
    """Fetch one paper's HTML, or None when arXiv has no rendering for it.

    A missing rendering is normal - a PDF-only submission has no TeX source to
    convert - so it is reported rather than raised, and the caller falls back
    to the PDF.
    """
    fetcher = get or fetch
    try:
        return fetcher(html_url_for(arxiv_id))
    except HTTPError as error:
        if error.code == 404:
            return None
        raise


def extract_html_chunks(
    sources: list[HtmlSource],
    *,
    chunk_size: int = 1_200,
    overlap: int = 200,
    read_html: Callable[[Path], str] | None = None,
) -> list[Chunk]:
    """Chunk papers from their HTML rendering, verifying the bytes first.

    Same discipline as the PDF path: the file on disk is re-hashed before any
    text is emitted, so a passage can never claim to come from bytes that have
    since changed.

    Sections are numbered in place of pages. `page_number` therefore holds a
    section ordinal here, and `source_kind` records which meaning applies.
    """
    reader = read_html or _read_text
    fingerprint = chunker_fingerprint(chunk_size, overlap)
    chunks: list[Chunk] = []

    for source in sources:
        path = Path(source.local_path)
        actual = file_sha256(path)
        if actual != source.sha256:
            raise ValueError(
                f"Checksum mismatch for {source.arxiv_id}: "
                f"recorded={source.sha256}, file={actual}"
            )
        paper_chunks: list[Chunk] = []
        for ordinal, section in enumerate(parse_html_sections(reader(path)), start=1):
            for index, span in enumerate(
                chunk_page(section.text, chunk_size=chunk_size, overlap=overlap)
            ):
                paper_chunks.append(
                    Chunk(
                        chunk_id=_html_chunk_id(source, ordinal, index, fingerprint),
                        arxiv_id=source.arxiv_id,
                        source_url=f"https://arxiv.org/abs/{source.arxiv_id}",
                        license_url=None,
                        source_sha256=source.sha256,
                        page_number=ordinal,
                        chunk_index=index,
                        char_start=section.char_start + span.char_start,
                        char_end=section.char_start + span.char_end,
                        text=span.text,
                        chunker_version=fingerprint,
                        section_path=" > ".join(section.path),
                        kind=section.kind,
                        source_kind="html",
                    )
                )
        chunks.extend(mark_references(paper_chunks))
    return chunks


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _html_chunk_id(source: HtmlSource, ordinal: int, index: int, fingerprint: str) -> str:
    # "html" is part of the identity so an HTML chunk can never collide with the
    # PDF chunk of the same paper, section and position.
    identifier = f"html:{source.arxiv_id}:{source.sha256}:{ordinal}:{index}:{fingerprint}"
    return sha256(identifier.encode("utf-8")).hexdigest()[:20]
