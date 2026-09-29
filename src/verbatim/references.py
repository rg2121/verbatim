"""Reading a paper's bibliography from its arXiv HTML rendering.

LaTeXML marks every entry up as <li class="ltx_bibitem">. The entries feed
`grow` (papers cited but not held) and the citation graph in `verbatim.graph`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
import re

BIBITEM_CLASS = "ltx_bibitem"
# "arXiv:2401.12345v2", "arxiv.org/abs/hep-th/9711200", "arXiv 2401.12345", and
# other spacing variants found in real bibliographies.
ARXIV_ID = re.compile(
    r"arxiv(?:\.org)?[.:/\s]*(?:abs/|pdf/)?"
    r"((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?)",
    re.IGNORECASE,
)
DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
VERSION_SUFFIX = re.compile(r"v\d+$")
# LaTeXML prints the entry's own "[1]"; the ordinal already records it.
LEADING_MARKER = re.compile(r"^\[\d+\]\s*")


@dataclass(frozen=True)
class Reference:
    """One bibliography entry and what it resolves to.

    `raw_text` is kept even when nothing resolves, so resolution can improve later.
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
            # Count nested <li> so a nested list does not end the entry early.
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
    """The arXiv identifier a reference states, without its version; never guessed from a title."""
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
