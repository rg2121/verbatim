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
