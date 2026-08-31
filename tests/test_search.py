from pathlib import Path

import pytest

from verbatim.arxiv import Paper
from verbatim.chunking import extract_pdf_chunks
from verbatim.corpus import CorpusStore, build_fts_query
from verbatim.downloads import DownloadedPaper, file_sha256


def paper(arxiv_id: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        summary="A test abstract.",
        authors=["Ada Lovelace"],
        categories=["physics.gen-ph"],
        published="2026-08-01T10:00:00Z",
        updated="2026-08-08T10:00:00Z",
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        license_url=None,
    )


def indexed_corpus(tmp_path: Path, pages_by_paper: dict[str, list[str]]) -> CorpusStore:
    corpus = CorpusStore(tmp_path / "corpus.db")
    for arxiv_id, pages in pages_by_paper.items():
        pdf = tmp_path / f"{arxiv_id}.pdf"
        pdf.write_bytes(f"%PDF-1.7 {arxiv_id}\n%%EOF\n".encode())
        record = DownloadedPaper(
            arxiv_id=arxiv_id,
            source_url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            license_url=None,
            local_path=str(pdf),
            sha256=file_sha256(pdf),
            byte_count=pdf.stat().st_size,
        )
        corpus.upsert_papers([paper(arxiv_id)])
        corpus.record_downloads([record])
        corpus.record_chunks(extract_pdf_chunks([record], extract_pages=lambda _, p=pages: p))
    return corpus


def test_search_finds_the_chunk_containing_an_exact_term(tmp_path: Path) -> None:
    with indexed_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Superconducting qubits decohere quickly."],
            "2608.22222v1": ["Photosynthesis converts light into chemical energy."],
        },
    ) as corpus:
        matches = corpus.search_chunks("qubits")

    assert [match.chunk.arxiv_id for match in matches] == ["2608.11111v1"]
    assert matches[0].chunk.page_number == 1
    assert "qubits" in matches[0].snippet


def test_search_can_be_restricted_to_one_paper(tmp_path: Path) -> None:
    with indexed_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Lattice models predict isotope abundance."],
            "2608.22222v1": ["Lattice gauge theory on a discrete grid."],
        },
    ) as corpus:
        everywhere = corpus.search_chunks("lattice")
        one_paper = corpus.search_chunks("lattice", arxiv_id="2608.22222v1")

    assert len(everywhere) == 2
    assert [match.chunk.arxiv_id for match in one_paper] == ["2608.22222v1"]


def test_search_index_drops_chunks_that_were_replaced_by_rechunking(tmp_path: Path) -> None:
    with indexed_corpus(tmp_path, {"2608.11111v1": ["Original wording about muons."]}) as corpus:
        record = corpus.all_downloads()[0]
        corpus.record_chunks(
            extract_pdf_chunks([record], extract_pages=lambda _: ["Replacement wording about kaons."])
        )
        stale = corpus.search_chunks("muons")
        fresh = corpus.search_chunks("kaons")

    assert stale == []
    assert [match.chunk.arxiv_id for match in fresh] == ["2608.11111v1"]


def test_search_backfills_the_index_for_a_database_written_before_it_existed(tmp_path: Path) -> None:
    database = tmp_path / "corpus.db"
    with indexed_corpus(tmp_path, {"2608.11111v1": ["Neutrino oscillation baseline."]}):
        pass
    # Simulate an upgrade: drop the index and reopen, as _initialize would find it.
    with CorpusStore(database) as corpus:
        corpus.connection.execute("DROP TABLE chunks_fts")
        corpus.connection.commit()
    with CorpusStore(database) as corpus:
        matches = corpus.search_chunks("neutrino")

    assert [match.chunk.arxiv_id for match in matches] == ["2608.11111v1"]


@pytest.mark.parametrize(
    "query",
    ['spin-orbit coupling', 'arXiv:2608.11111v1', 'a "b', 'NEAR(', '-- ; DROP', '*', '""'],
)
def test_search_never_raises_on_punctuation_in_a_query(tmp_path: Path, query: str) -> None:
    with indexed_corpus(tmp_path, {"2608.11111v1": ["Spin-orbit coupling in heavy atoms."]}) as corpus:
        corpus.search_chunks(query)


def test_build_fts_query_quotes_terms_and_keeps_prefix_search() -> None:
    assert build_fts_query("spin-orbit coupling") == '"spin-orbit" "coupling"'
    assert build_fts_query("suprac*") == '"suprac"*'
    assert build_fts_query("   ") == ""


def test_format_match_renders_a_traceable_citation_header() -> None:
    from verbatim.chunking import Chunk
    from verbatim.cli import format_match
    from verbatim.corpus import ChunkMatch

    match = ChunkMatch(
        chunk=Chunk(
            chunk_id="abc123",
            arxiv_id="2608.05440v1",
            source_url="https://arxiv.org/abs/2608.05440v1",
            license_url=None,
            source_sha256="f" * 64,
            page_number=16,
            chunk_index=2,
            char_start=40,
            char_end=90,
            text="Lorentz contraction compresses the flux into a smaller area.",
            chunker_version="2:1200:200",
        ),
        score=3.74,
        snippet="…compresses the [flux] into…",
    )

    rendered = format_match(1, match)
    full = format_match(1, match, full=True)

    assert "arXiv:2608.05440v1" in rendered
    assert "p. 16" in rendered
    assert "chars 40-90" in rendered
    assert "https://arxiv.org/abs/2608.05440v1" in rendered
    assert "no licence reported" in rendered
    assert "…compresses the [flux] into…" in rendered
    assert "Lorentz contraction compresses" in full


def test_a_result_shows_the_section_it_came_from() -> None:
    """Which part of a paper a passage came from is context the reader needs."""
    from verbatim.chunking import Chunk
    from verbatim.cli import format_match
    from verbatim.corpus import ChunkMatch

    def rendered(section_path: str) -> str:
        chunk = Chunk(
            chunk_id="abc", arxiv_id="2608.05440v1",
            source_url="https://arxiv.org/abs/2608.05440v1", license_url=None,
            source_sha256="f" * 64, page_number=6, chunk_index=0, char_start=0,
            char_end=10, text="body text", chunker_version="6:1200:200",
            section_path=section_path,
        )
        return format_match(1, ChunkMatch(chunk=chunk, score=1.0, snippet="body text"))

    deep = rendered("Paper Title > 3 Results > 3.1 Ablations")
    assert "section: 3.1 Ablations" in deep
    # The title is shared by every passage in a paper, so it is not shown.
    assert "Paper Title" not in deep
    assert "section:" not in rendered("")


def test_a_result_from_html_is_not_given_a_page_number() -> None:
    """page_number is a section ordinal for HTML; printing "p.4" invents a page."""
    from verbatim.chunking import Chunk
    from verbatim.cli import format_match
    from verbatim.corpus import ChunkMatch

    chunk = Chunk(
        chunk_id="abc", arxiv_id="2410.16588v3",
        source_url="https://arxiv.org/abs/2410.16588v3", license_url=None,
        source_sha256="f" * 64, page_number=4, chunk_index=0, char_start=0,
        char_end=9, text="body text", chunker_version="7:1200:200",
        section_path="The power spectrum > II Formalism", source_kind="html",
    )

    rendered = format_match(1, ChunkMatch(chunk=chunk, score=1.0, snippet="body text"))

    assert "p. 4" not in rendered and "p.4" not in rendered
    assert "section II Formalism" in rendered
    # The header already says where it is, so it is not repeated below.
    assert rendered.count("II Formalism") == 1


def filtered_corpus(tmp_path: Path) -> CorpusStore:
    """Two papers, different dates and categories, each with one passage."""
    from verbatim.arxiv import Paper

    corpus = CorpusStore(tmp_path / "corpus.db")
    for arxiv_id, published, category in [
        ("2601.11111v1", "2026-01-15T00:00:00Z", "cs.CL"),
        ("2608.22222v1", "2026-08-15T00:00:00Z", "gr-qc"),
    ]:
        pdf = tmp_path / f"{arxiv_id}.pdf"
        pdf.write_bytes(f"%PDF-1.7 {arxiv_id}\n%%EOF\n".encode())
        record = DownloadedPaper(
            arxiv_id=arxiv_id, source_url="u", pdf_url="p", license_url=None,
            local_path=str(pdf), sha256=file_sha256(pdf), byte_count=pdf.stat().st_size,
        )
        corpus.upsert_papers([Paper(
            arxiv_id=arxiv_id, title=f"Paper {arxiv_id}", summary="Abstract.",
            authors=["Ada Lovelace"], categories=[category], published=published,
            updated=published, source_url="u", pdf_url="p", license_url=None,
        )])
        corpus.record_downloads([record])
        corpus.record_chunks(
            extract_pdf_chunks([record], extract_pages=lambda _: ["Lattice models of flux."])
        )
    return corpus


def test_search_can_be_restricted_to_papers_published_since_a_date(tmp_path: Path) -> None:
    """Similarity has no notion of recency, so the filter has to supply one."""
    from verbatim.corpus import PaperFilter

    with filtered_corpus(tmp_path) as corpus:
        everything = corpus.search_chunks("lattice")
        recent = corpus.search_chunks("lattice", papers=PaperFilter(since="2026-06-01"))

    assert len(everything) == 2
    assert [m.chunk.arxiv_id for m in recent] == ["2608.22222v1"]


def test_search_can_be_restricted_to_one_category(tmp_path: Path) -> None:
    from verbatim.corpus import PaperFilter

    with filtered_corpus(tmp_path) as corpus:
        in_cl = corpus.search_chunks("lattice", papers=PaperFilter(category="cs.CL"))

    assert [m.chunk.arxiv_id for m in in_cl] == ["2601.11111v1"]


def test_a_category_filter_does_not_match_a_longer_name(tmp_path: Path) -> None:
    """cs.CL must not also select cs.CLX, which a bare substring would."""
    from verbatim.corpus import PaperFilter

    with filtered_corpus(tmp_path) as corpus:
        assert corpus.search_chunks("lattice", papers=PaperFilter(category="cs.C")) == []


def test_an_empty_filter_changes_nothing(tmp_path: Path) -> None:
    from verbatim.corpus import PaperFilter

    with filtered_corpus(tmp_path) as corpus:
        assert len(corpus.search_chunks("lattice", papers=PaperFilter())) == 2
