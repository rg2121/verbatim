"""Turning a retrieved passage into a citation a reader can check."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from verbatim.arxiv import Paper
from verbatim.chunking import Chunk
from verbatim.corpus import split_arxiv_version

MAX_NAMED_AUTHORS = 3


@dataclass(frozen=True)
class Citation:
    """Everything needed to find the cited words again, and to prove they were there.

    The version is carried separately from the base identifier because v1 and
    v2 of a paper can disagree; citing "arXiv:2608.05440" without a version
    points at whichever revision the reader happens to fetch.
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
    pdf_sha256: str
    source_url: str
    license_url: str | None
    retrieved_at: str

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
        pdf_sha256=chunk.pdf_sha256,
        source_url=chunk.source_url,
        license_url=chunk.license_url,
        retrieved_at=datetime.now(UTC).isoformat(),
    )


def format_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown author"
    if len(authors) > MAX_NAMED_AUTHORS:
        return f"{_surname(authors[0])} et al."
    return ", ".join(_surname(author) for author in authors)


def render_citation(marker: int, citation: Citation) -> str:
    """Render one reference line.

    The version is always present, and the URL resolves to that exact version.
    """
    return (
        f"[{marker}] {format_authors(citation.authors)}, "
        f'"{citation.title}", '
        f"arXiv:{citation.arxiv_id}, p. {citation.page_number}. "
        f"{citation.source_url}"
    )


def render_bibliography(citations: list[Citation]) -> str:
    return "\n".join(
        render_citation(marker, citation) for marker, citation in enumerate(citations, start=1)
    )


def _surname(author: str) -> str:
    return author.strip().split()[-1] if author.strip() else author
