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


def test_search_can_be_restricted_to_a_list_of_papers(tmp_path: Path) -> None:
    """The opposite of hiding papers: searching only within known ones."""
    from verbatim.corpus import PaperFilter

    with filtered_corpus(tmp_path) as corpus:
        kept = corpus.search_chunks(
            "lattice", papers=PaperFilter(include_ids=("2608.22222v1",))
        )
        both = corpus.search_chunks(
            "lattice",
            papers=PaperFilter(include_ids=("2601.11111v1", "2608.22222v1")),
        )
        # An include list and an exclude list are read together, not fought over.
        neither = corpus.search_chunks(
            "lattice",
            papers=PaperFilter(
                include_ids=("2608.22222v1",), exclude_ids=("2608.22222v1",)
            ),
        )

    assert [m.chunk.arxiv_id for m in kept] == ["2608.22222v1"]
    assert len(both) == 2
    assert neither == []


def rank_every_match(
    corpus: CorpusStore,
    query: str,
    *,
    limit: int,
    match: str = "any",
    arxiv_id: str | None = None,
    include_references: bool = False,
    papers=None,
) -> list[tuple[str, float, str]]:
    """The keyword search as it ran before narrowing: every matching row read, then ranked."""
    from verbatim.chunking import REFERENCES

    clauses, parameters = ["chunks_fts MATCH ?"], [build_fts_query(query, match=match)]
    if arxiv_id is not None:
        clauses.append("c.arxiv_id = ?")
        parameters.append(arxiv_id)
    if not include_references:
        clauses.append("c.kind != ?")
        parameters.append(REFERENCES)
    if papers:
        clause, more = papers.clause("c")
        clauses.append(clause)
        parameters.extend(more)
    rows = corpus.connection.execute(
        "SELECT c.chunk_id, bm25(chunks_fts), snippet(chunks_fts, 0, '[', ']', '…', 16)"
        " FROM chunks_fts JOIN chunks AS c ON c.rowid = chunks_fts.rowid"
        f" WHERE {' AND '.join(clauses)} ORDER BY bm25(chunks_fts) ASC LIMIT ?",
        [*parameters, limit],
    ).fetchall()
    return [(row[0], -row[1], row[2]) for row in rows]


def found(matches) -> list[tuple[str, float, str]]:
    return [(m.chunk.chunk_id, m.score, m.snippet) for m in matches]


def flux_corpus(tmp_path: Path) -> CorpusStore:
    """Passages that all say flux, the two that say it most being bibliographies."""
    from verbatim.chunking import REFERENCES

    corpus = indexed_corpus(
        tmp_path,
        {
            "2608.10001v1": ["Flux, A. (2019). Flux flux flux. Journal of Flux and Fluxes."],
            "2608.10002v1": ["Flux, B. (2020). Flux rings. Flux Letters, flux."],
            "2608.10003v1": ["Magnetic flux threads the ring.", "The flux tube carries charge."],
            "2608.10004v1": ["Flux pinning in superconductors.", "Vortex lattice and flux creep."],
            "2608.10005v1": ["Heat flux across a boundary layer of air."],
            "2608.10006v1": ["A plant converts sunlight.", "Neutron flux in a reactor core."],
            "2608.10007v1": ["Flux balance analysis of cell metabolism."],
        },
    )
    corpus.connection.execute(
        "UPDATE chunks SET kind = ? WHERE arxiv_id IN ('2608.10001v1', '2608.10002v1')",
        (REFERENCES,),
    )
    corpus.connection.commit()
    return corpus


@pytest.mark.parametrize("prefetch", [1, 2, 3, 1000])
@pytest.mark.parametrize("limit", [1, 2, 3, 20])
@pytest.mark.parametrize("query", ["flux", "magnetic flux ring", "flux reactor plant"])
def test_ranking_in_the_index_first_returns_what_ranking_every_match_returns(
    tmp_path: Path, monkeypatch, prefetch: int, limit: int, query: str
) -> None:
    """Reading every matching row before ranking took 28 s on 2.9 million passages.

    The index now ranks first and only its best rows are read. Small prefetches
    leave too few once bibliographies are filtered out, which must fall back to
    ranking every match rather than return less.
    """
    import verbatim.corpus as corpus_module

    monkeypatch.setattr(corpus_module, "KEYWORD_PREFETCH", prefetch)
    with flux_corpus(tmp_path) as corpus:
        for include_references in (False, True):
            for match in ("any", "all"):
                assert found(corpus.search_chunks(
                    query, limit=limit, match=match, include_references=include_references,
                )) == rank_every_match(
                    corpus, query, limit=limit, match=match,
                    include_references=include_references,
                )


@pytest.mark.parametrize("prefetch", [1, 1000])
def test_a_filtered_search_ranks_as_it_did_before_narrowing(
    tmp_path: Path, monkeypatch, prefetch: int
) -> None:
    import verbatim.corpus as corpus_module
    from verbatim.corpus import PaperFilter

    monkeypatch.setattr(corpus_module, "KEYWORD_PREFETCH", prefetch)
    only = PaperFilter(include_ids=("2608.10005v1", "2608.10007v1"))
    hidden = PaperFilter(exclude_ids=("2608.10003v1",))
    with flux_corpus(tmp_path) as corpus:
        for papers in (only, hidden):
            assert found(corpus.search_chunks("flux", limit=3, match="any", papers=papers)) == (
                rank_every_match(corpus, "flux", limit=3, papers=papers)
            )


def test_a_search_within_one_paper_ranks_as_it_did_before_narrowing(tmp_path: Path) -> None:
    """A scoped search reads only its paper's rowid range, and still filters by paper."""
    with flux_corpus(tmp_path) as corpus:
        papers = [row[0] for row in corpus.connection.execute("SELECT arxiv_id FROM papers")]
        for arxiv_id in papers:
            for include_references in (False, True):
                assert found(corpus.search_chunks(
                    "flux ring creep", limit=5, match="any", arxiv_id=arxiv_id,
                    include_references=include_references,
                )) == rank_every_match(
                    corpus, "flux ring creep", limit=5, arxiv_id=arxiv_id,
                    include_references=include_references,
                )


def test_a_search_within_a_paper_whose_passages_are_interleaved_stays_within_it(
    tmp_path: Path,
) -> None:
    """Passages written in one pass for two papers share a rowid range."""
    from verbatim.chunking import extract_pdf_chunks
    from helpers import downloaded

    with indexed_corpus(tmp_path, {}) as corpus:
        first, second = downloaded(tmp_path, "2608.20001v1"), downloaded(tmp_path, "2608.20002v1")
        corpus.upsert_papers([paper("2608.20001v1"), paper("2608.20002v1")])
        corpus.record_downloads([first, second])
        pages = {
            "2608.20001v1": ["Flux in the first paper.", "More flux, first paper."],
            "2608.20002v1": ["Flux in the second paper.", "More flux, second paper."],
        }
        chunks = extract_pdf_chunks([first, second], extract_pages=lambda pdf: pages[pdf.stem])
        # Alternate the two papers' passages, so each paper's rowids straddle the other's.
        ordered = sorted(chunks, key=lambda chunk: (chunk.page_number, chunk.arxiv_id))
        corpus.record_chunks(ordered)

        scoped = corpus.search_chunks("flux", limit=5, match="any", arxiv_id="2608.20001v1")
        expected = rank_every_match(corpus, "flux", limit=5, arxiv_id="2608.20001v1")

    assert {m.chunk.arxiv_id for m in scoped} == {"2608.20001v1"}
    assert len(scoped) == 2
    assert found(scoped) == expected


def test_a_search_within_a_paper_with_no_passages_finds_nothing(tmp_path: Path) -> None:
    with flux_corpus(tmp_path) as corpus:
        assert corpus.search_chunks("flux", match="any", arxiv_id="2608.99999v1") == []
