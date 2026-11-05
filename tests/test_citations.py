import json
from pathlib import Path

import pytest

from verbatim.citations import (
    Citation,
    citation_from_chunk,
    export_references,
    format_authors,
    render_bibliography,
    render_citation,
)
from verbatim.chunking import Chunk

from helpers import chunked_corpus, paper, titled_paper


def chunk(arxiv_id: str = "2608.05440v1") -> Chunk:
    return Chunk(
        chunk_id="abc123",
        arxiv_id=arxiv_id,
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        license_url=None,
        source_sha256="f" * 64,
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
    assert citation.source_sha256 == "f" * 64


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
        stored = list(corpus.iter_chunks())[0]
        found = corpus.get_paper(stored.arxiv_id)
        assert found is not None
        citation = citation_from_chunk(stored, found, "Magnetic flux")

    assert isinstance(citation, Citation)
    assert citation.chunk_id == stored.chunk_id
    assert citation.quoted_text in stored.text


def _citation(**overrides):
    """A Citation with only the fields a locator test cares about set."""
    from verbatim.citations import Citation

    fields = dict(
        arxiv_id="2608.05440v1", base_arxiv_id="2608.05440", version=1,
        title="A Paper", authors=["Ada Lovelace"], page_number=1,
        char_start=0, char_end=10, quoted_text="some words", chunk_id="abc",
        source_sha256="f" * 64, source_url="https://arxiv.org/abs/2608.05440v1",
        license_url=None, retrieved_at="2026-08-29T00:00:00+00:00",
    )
    fields.update(overrides)
    return Citation(**fields)

def test_an_html_citation_names_the_section_the_paper_gave_it() -> None:
    """Our ordinal is not the paper's numbering, and papers use Roman numerals."""
    from verbatim.citations import locator

    cited = _citation(
        source_kind="html", page_number=4,
        section_path="Global superscaling analysis > III The SuSAM* approach",
    )

    assert locator(cited) == "section III The SuSAM* approach"


def test_an_html_citation_falls_back_to_the_ordinal_without_a_heading() -> None:
    from verbatim.citations import locator

    assert locator(_citation(source_kind="html", page_number=4)) == "section 4"


def test_a_pdf_citation_still_cites_a_page() -> None:
    from verbatim.citations import locator

    assert locator(_citation(page_number=7, section_path="2 Method")) == "p. 7"


LICENCE = "http://creativecommons.org/licenses/by/4.0/"
AWKWARD = "Fish & Chips: {a} 50% of #1 at $2_x ~ ^y \\ z"


def _exported(fmt: str, *citations: Citation, title: str = "A Paper") -> str:
    papers = {cited.arxiv_id: titled_paper(cited.arxiv_id, title) for cited in citations}
    return export_references(list(citations), papers, fmt)


def test_markdown_references_name_the_licence() -> None:
    lines = _exported(
        "markdown", _citation(license_url=LICENCE), _citation(arxiv_id="2608.05441v2", chunk_id="def")
    ).splitlines()

    assert lines[0].startswith('[1] Lovelace, "A Paper", arXiv:2608.05440v1, p. 1.')
    assert lines[0].endswith(f"Licence: {LICENCE}.")
    assert lines[1].startswith("[2] ")
    assert lines[1].endswith("Licence: not reported.")


def test_bibtex_cites_the_exact_version_with_the_locator_and_licence() -> None:
    entry = _exported(
        "bibtex",
        _citation(source_kind="html", section_path="3 Model > 3.3 YaRN", license_url=LICENCE),
    )

    assert entry.startswith("@misc{arxiv:2608.05440v1,")
    assert "title = {A Paper}" in entry
    assert "author = {Ada Lovelace}" in entry
    assert "year = {2026}" in entry
    assert "eprint = {2608.05440v1}" in entry
    assert "archivePrefix = {arXiv}" in entry
    assert "primaryClass = {cs.LG}" in entry
    assert "url = {https://arxiv.org/abs/2608.05440v1}" in entry
    assert "section 3.3 YaRN" in entry
    assert "Licence" in entry


def test_bibtex_has_one_entry_per_paper_with_every_locator() -> None:
    entry = _exported(
        "bibtex", _citation(page_number=2, chunk_id="a"), _citation(page_number=9, chunk_id="b"),
    )

    assert entry.count("@misc{") == 1
    assert "p. 2; p. 9" in entry


def test_bibtex_survives_a_title_with_its_special_characters() -> None:
    """One unbalanced brace in a title breaks the whole .bib file."""
    entry = _exported("bibtex", _citation(), title=AWKWARD)
    title = next(line for line in entry.splitlines() if "title =" in line)

    assert title.count(r"\{") == title.count(r"\}") == 1
    for special in ("&", "%", "#", "$", "_"):
        assert f"\\{special}" in title
    assert entry.count("{") - entry.count(r"\{") == entry.count("}") - entry.count(r"\}")


def test_an_old_style_id_makes_a_usable_bibtex_key() -> None:
    entry = _exported("bibtex", _citation(arxiv_id="cs/0301001v1"))

    assert entry.startswith("@misc{arxiv:cs-0301001v1,")


def test_csl_json_parses_and_names_its_authors() -> None:
    (item,) = json.loads(_exported("csl-json", _citation(license_url=LICENCE), title=AWKWARD))

    assert item["id"] == "arxiv:2608.05440v1"
    assert item["type"] == "article"
    assert item["title"] == AWKWARD
    assert item["author"] == [{"family": "Lovelace", "given": "Ada"}]
    assert item["issued"] == {"date-parts": [[2026, 8, 1]]}
    assert item["URL"] == "https://arxiv.org/abs/2608.05440v1"
    assert item["number"] == "arXiv:2608.05440v1"
    assert "p. 1" in item["note"] and LICENCE in item["note"]


def test_an_unknown_format_is_refused() -> None:
    with pytest.raises(ValueError, match="format must be"):
        _exported("word", _citation())
