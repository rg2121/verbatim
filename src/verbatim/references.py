"""Reading a paper's bibliography out of its arXiv HTML rendering.

A reference list is the one part of a paper that names other papers, and
LaTeXML marks it up properly: every entry is an <li class="ltx_bibitem">,
where a PDF offers only a run of text that has to be guessed apart.

What this is for is not a citation graph. Measured over 854 renderings, 10,707
references carry a resolvable arXiv identifier and just 70 of them point at a
paper this corpus holds - 69 edges, because the corpus is a slice of the newest
papers in each category and new papers cite old ones. The useful direction is
outward: the references say which papers this corpus keeps leaning on and does
not contain, and that is a reading list for growing it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
import re

BIBITEM_CLASS = "ltx_bibitem"
# arXiv:2401.12345v2, arxiv.org/abs/hep-th/9711200, and the spacing variants
# that appear in real bibliographies.
ARXIV_ID = re.compile(
    # "arXiv:2401.12345v2", "arxiv.org/abs/hep-th/9711200", "arXiv 2401.12345"
    r"arxiv(?:\.org)?[.:/\s]*(?:abs/|pdf/)?"
    r"((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?)",
    re.IGNORECASE,
)
DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
VERSION_SUFFIX = re.compile(r"v\d+$")
# LaTeXML prints the entry's own marker inside it; the ordinal already records
# it, so carrying "[1]" into the stored text would only duplicate it.
LEADING_MARKER = re.compile(r"^\[\d+\]\s*")


@dataclass(frozen=True)
class Reference:
    """One entry in a paper's bibliography, and what it could be resolved to.

    `raw_text` is kept whether or not an identifier was found: an unresolved
    reference is still evidence of what the paper builds on, and resolution can
    improve later without re-reading the HTML.
    """

    citing_arxiv_id: str
    ordinal: int
    raw_text: str
    cited_arxiv_id: str | None = None
    cited_doi: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BibliographyParser(HTMLParser):
    """Collect the text of every bibliography entry, in order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.entries: list[str] = []
        self._current: list[str] | None = None
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._current is not None:
            # A nested list inside an entry must not let its end tag close the
            # entry early, so same-name nesting is counted.
            if tag == "li":
                self._depth += 1
            return
        if tag == "li" and BIBITEM_CLASS in (dict(attrs).get("class") or ""):
            self._current, self._depth = [], 1

    def handle_endtag(self, tag: str) -> None:
        if self._current is None or tag != "li":
            return
        self._depth -= 1
        if self._depth > 0:
            return
        text = LEADING_MARKER.sub("", " ".join("".join(self._current).split()))
        self._current = None
        if text:
            self.entries.append(text)

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._current.append(data)


def parse_bibliography(html: str, citing_arxiv_id: str) -> list[Reference]:
    """Read one paper's reference list, resolving identifiers where they exist."""
    parser = BibliographyParser()
    parser.feed(html)
    parser.close()
    references = []
    for ordinal, text in enumerate(parser.entries, start=1):
        references.append(
            Reference(
                citing_arxiv_id=citing_arxiv_id,
                ordinal=ordinal,
                raw_text=text,
                cited_arxiv_id=extract_arxiv_id(text),
                cited_doi=extract_doi(text),
            )
        )
    return references


def extract_arxiv_id(text: str) -> str | None:
    """The arXiv identifier a reference names, without its version.

    Deterministic extraction only. A reference is resolved because it states an
    identifier, never because a title looked similar - the same standard the
    rest of the project holds citations to.
    """
    match = ARXIV_ID.search(text)
    if match is None:
        return None
    return VERSION_SUFFIX.sub("", match.group(1))


def extract_doi(text: str) -> str | None:
    match = DOI.search(text)
    return match.group(0).rstrip(".,;") if match is not None else None


def references_from_file(path: Path, citing_arxiv_id: str) -> list[Reference]:
    """Parse the bibliography of one locally held HTML rendering."""
    return parse_bibliography(
        path.read_text(encoding="utf-8", errors="replace"), citing_arxiv_id
    )
