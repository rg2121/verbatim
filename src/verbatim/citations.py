"""Turning a retrieved passage into a citation a reader can check."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import re

from verbatim.arxiv import Paper
from verbatim.chunking import Chunk
from verbatim.corpus import split_arxiv_version

MAX_NAMED_AUTHORS = 3
EXPORT_FORMATS = ("markdown", "bibtex", "csl-json")
BIBTEX_SPECIAL = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


@dataclass(frozen=True)
class Citation:
    """Everything needed to find the cited words again and check them.

    The version is kept apart from the base identifier because revisions of a
    paper can differ.
    """

    arxiv_id: str
    base_arxiv_id: str
    version: int
    title: str
    authors: list[str]
    page_number: int
    char_start: int
    char_end: int
    quoted_text: str
    chunk_id: str
    source_sha256: str
    source_url: str
    license_url: str | None
    retrieved_at: str
    source_kind: str = "pdf"
    # The author's own heading for this passage, e.g. "III The SuSAM* approach".
    section_path: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def citation_from_chunk(chunk: Chunk, paper: Paper, quoted_text: str) -> Citation:
    """Build a citation for the words a claim actually rests on."""
    base_id, version = split_arxiv_version(chunk.arxiv_id)
    return Citation(
        arxiv_id=chunk.arxiv_id,
        base_arxiv_id=base_id,
        version=version,
        title=paper.title,
        authors=list(paper.authors),
        page_number=chunk.page_number,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        quoted_text=quoted_text,
        chunk_id=chunk.chunk_id,
        source_sha256=chunk.source_sha256,
        source_kind=chunk.source_kind,
        section_path=chunk.section_path,
        source_url=chunk.source_url,
        # The paper's record is the authority: a chunk's own column is empty
        # for HTML passages and for any licence looked up after chunking.
        license_url=paper.license_url or chunk.license_url,
        retrieved_at=datetime.now(UTC).isoformat(),
    )


def format_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown author"
    if len(authors) > MAX_NAMED_AUTHORS:
        return f"{_surname(authors[0])} et al."
    return ", ".join(_surname(author) for author in authors)


def render_citation(marker: int, citation: Citation) -> str:
    """Render one reference line; the version and URL name the exact revision."""
    return (
        f"[{marker}] {format_authors(citation.authors)}, "
        f'"{citation.title}", '
        f"arXiv:{citation.arxiv_id}, {locator(citation)}. "
        f"{citation.source_url}"
    )


def locator(citation: Citation) -> str:
    """Where in the paper this is, in the terms its source actually has."""
    return render_locator(
        citation.source_kind, citation.page_number, citation.section_path
    )


def render_locator(source_kind: str, page_number: int, section_path: str) -> str:
    """Where a passage is: a page for a PDF, the author's section heading for HTML.

    HTML has no pages, so printing one would invent it; the section counter is
    only a fallback when there is no heading.
    """
    if source_kind != "html":
        return f"p. {page_number}"
    heading = section_path.split(" > ")[-1].strip()
    return f"section {heading}" if heading else f"section {page_number}"


def render_bibliography(citations: list[Citation]) -> str:
    return "\n".join(
        render_citation(marker, citation) for marker, citation in enumerate(citations, start=1)
    )


def export_references(
    citations: list[Citation], papers: dict[str, Paper], fmt: str = "markdown"
) -> str:
    """A reference list rendered by code, so that no model retypes one.

    Markdown numbers each passage, as the answer's markers do. BibTeX and CSL-JSON
    hold one record per paper version, with every cited locator in its note.
    """
    if fmt == "markdown":
        return "\n".join(
            f"{render_citation(marker, citation)} "
            f"Licence: {citation.license_url or 'not reported'}."
            for marker, citation in enumerate(citations, start=1)
        )
    groups: dict[str, list[Citation]] = {}
    for citation in citations:
        groups.setdefault(citation.arxiv_id, []).append(citation)
    if fmt == "bibtex":
        return "\n\n".join(
            _bibtex_entry(group, papers[arxiv_id]) for arxiv_id, group in groups.items()
        )
    if fmt == "csl-json":
        return json.dumps(
            [_csl_item(group, papers[arxiv_id]) for arxiv_id, group in groups.items()],
            indent=2,
            ensure_ascii=False,
        )
    raise ValueError(f"format must be one of {', '.join(EXPORT_FORMATS)}, got {fmt!r}")


def _record_key(arxiv_id: str) -> str:
    """A citation key, which an old-style id's slash would break."""
    return f"arxiv:{arxiv_id}".replace("/", "-")


def _record_note(group: list[Citation]) -> str:
    where = "; ".join(dict.fromkeys(locator(citation) for citation in group))
    return f"{where}. Licence: {group[0].license_url or 'not reported'}."


def _bibtex_text(text: str) -> str:
    """Escaped, since one unbalanced brace in a title breaks the whole .bib file."""
    return re.sub(r"[\\{}&%$#_~^]", lambda found: BIBTEX_SPECIAL[found.group()], " ".join(text.split()))


def _bibtex_entry(group: list[Citation], paper: Paper) -> str:
    cited = group[0]
    fields = {
        "title": _bibtex_text(paper.title),
        "author": _bibtex_text(" and ".join(paper.authors)),
        "year": paper.published[:4],
        "eprint": cited.arxiv_id,
        "archivePrefix": "arXiv",
        "primaryClass": paper.categories[0] if paper.categories else "",
        "url": cited.source_url,
        "note": _bibtex_text(_record_note(group)),
    }
    lines = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields.items() if value)
    return f"@misc{{{_record_key(cited.arxiv_id)},\n{lines}\n}}"


def _csl_author(author: str) -> dict[str, str]:
    given = " ".join(author.split()[:-1])
    return {"family": _surname(author), **({"given": given} if given else {})}


def _csl_item(group: list[Citation], paper: Paper) -> dict[str, object]:
    cited = group[0]
    published = re.match(r"(\d{4})-(\d{2})-(\d{2})", paper.published)
    return {
        "id": _record_key(cited.arxiv_id),
        "type": "article",
        "title": paper.title,
        "author": [_csl_author(author) for author in paper.authors],
        **({"issued": {"date-parts": [[int(part) for part in published.groups()]]}} if published else {}),
        "URL": cited.source_url,
        "number": f"arXiv:{cited.arxiv_id}",
        "note": _record_note(group),
    }


def _surname(author: str) -> str:
    return author.strip().split()[-1] if author.strip() else author
