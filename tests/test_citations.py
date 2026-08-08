from pathlib import Path

from verbatim.citations import (
    Citation,
    citation_from_chunk,
    format_authors,
    render_bibliography,
    render_citation,
)
from verbatim.chunking import Chunk

from helpers import chunked_corpus, paper


def chunk(arxiv_id: str = "2608.05440v1") -> Chunk:
    return Chunk(
        chunk_id="abc123",
        arxiv_id=arxiv_id,
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        license_url=None,
        pdf_sha256="f" * 64,
        page_number=6,
        chunk_index=0,
        char_start=40,
        char_end=90,
        text="An increasing current in a small wire loop generates flux.",
        chunker_version="2:1200:200",
    )


def test_citation_carries_the_version_separately_from_the_base_identifier() -> None:
    citation = citation_from_chunk(chunk(), paper("2608.05440v1"), "generates flux")

    assert citation.base_arxiv_id == "2608.05440"
    assert citation.version == 1
    assert citation.quoted_text == "generates flux"
    assert citation.pdf_sha256 == "f" * 64


def test_citation_supports_legacy_identifiers() -> None:
    citation = citation_from_chunk(
        chunk("math/0309136v2"), paper("math/0309136v2"), "Ricci flow"
    )

    assert citation.base_arxiv_id == "math/0309136"
    assert citation.version == 2


def test_rendered_citation_always_pins_the_version() -> None:
    rendered = render_citation(1, citation_from_chunk(chunk(), paper("2608.05440v1"), "flux"))

    assert "arXiv:2608.05440v1" in rendered
    assert "arXiv:2608.05440," not in rendered  # never the unversioned form
    assert "p. 6" in rendered
    assert "https://arxiv.org/abs/2608.05440v1" in rendered


def test_author_lists_collapse_once_they_get_long() -> None:
    assert format_authors(["Ada Lovelace"]) == "Lovelace"
    assert format_authors(["Ada Lovelace", "Alan Turing"]) == "Lovelace, Turing"
    assert format_authors(["A One", "B Two", "C Three", "D Four"]) == "One et al."
    assert format_authors([]) == "Unknown author"


def test_bibliography_numbers_entries_from_one() -> None:
    citations = [
        citation_from_chunk(chunk("2608.05440v1"), paper("2608.05440v1"), "flux"),
        citation_from_chunk(chunk("2512.16941v3"), paper("2512.16941v3"), "string"),
    ]

    rendered = render_bibliography(citations).splitlines()

    assert rendered[0].startswith("[1] ")
    assert rendered[1].startswith("[2] ")
    assert "arXiv:2512.16941v3" in rendered[1]


def test_citation_can_be_built_from_a_stored_chunk(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux threads the ring."]}) as corpus:
        stored = corpus.all_chunks()[0]
        found = corpus.get_paper(stored.arxiv_id)
        assert found is not None
        citation = citation_from_chunk(stored, found, "Magnetic flux")

    assert isinstance(citation, Citation)
    assert citation.chunk_id == stored.chunk_id
    assert citation.quoted_text in stored.text
