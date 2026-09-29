"""Exporting a small corpus that works without an ingest."""

from pathlib import Path

import pytest

from helpers import chunked_corpus
from verbatim.corpus import CorpusStore, EmbeddingModel
from verbatim.references import Reference
from verbatim.sampling import export_sample


def seeded(tmp_path: Path, papers: int = 3) -> Path:
    """A corpus with embedded passages, enough to sample from."""
    pages = {f"2608.1111{n}v1": [f"Paper {n} discusses superconducting qubits."] * 45
             for n in range(papers)}
    with chunked_corpus(tmp_path, pages) as corpus:
        corpus.register_embedding_model(
            EmbeddingModel(model_id="stub", dimensions=4, normalized=True, revision="v0")
        )
        corpus.record_embeddings(
            "stub", [(c.chunk_id, [1.0, 0.0, 0.0, 0.0]) for c in list(corpus.iter_chunks())]
        )
        # Only an open licence may be redistributed, and a sample redistributes
        # the text, so a sample can hold nothing else.
        corpus.connection.execute("UPDATE papers SET license_class = 'open'")
        corpus.connection.commit()
    return tmp_path / "corpus.db"


def test_a_sample_holds_papers_passages_and_their_vectors(tmp_path: Path) -> None:
    source = seeded(tmp_path)

    report = export_sample(source, tmp_path / "sample.db", papers=2, model_id="stub")

    assert report.papers == 2
    assert report.chunks > 0 and report.vectors == report.chunks
    assert report.bytes_written > 0


def test_a_sample_claims_no_files_it_does_not_ship(tmp_path: Path) -> None:
    """stats reporting 40 PDFs that are not there is the claim to avoid."""
    source = seeded(tmp_path)

    export_sample(source, tmp_path / "sample.db", papers=2, model_id="stub")

    with CorpusStore(tmp_path / "sample.db") as sample:
        downloads = sample.connection.execute("SELECT COUNT(*) FROM downloads").fetchone()[0]
        chunks = list(sample.iter_chunks())

    assert downloads == 0
    # The passages still carry everything a citation needs.
    assert all(chunk.source_url for chunk in chunks)


def test_a_sample_carries_its_papers_bibliographies(tmp_path: Path) -> None:
    """The graph command on a fresh clone should have records to walk."""
    source = seeded(tmp_path, papers=3)
    with CorpusStore(source) as corpus:
        corpus.record_references(
            [
                Reference(
                    citing_arxiv_id="2608.11111v1", ordinal=1,
                    raw_text="[1] arXiv:2608.11110", cited_arxiv_id="2608.11110",
                )
            ],
            source_sha256="abc123",
        )

    report = export_sample(source, tmp_path / "sample.db", papers=3, model_id="stub")

    assert report.references == 1
    with CorpusStore(tmp_path / "sample.db") as sample:
        citers = sample.papers_citing("2608.11110")
    assert [p.arxiv_id for p in citers] == ["2608.11111v1"]


def test_the_sample_prefers_a_cited_neighborhood_over_strangers(tmp_path: Path) -> None:
    """Enough papers cite one hub that the sample's graph shows edges."""
    source = seeded(tmp_path, papers=9)
    with CorpusStore(source) as corpus:
        # Papers 1-4 all cite paper 0; papers 5-8 cite nothing.
        for n in range(1, 5):
            corpus.record_references(
                [
                    Reference(
                        citing_arxiv_id=f"2608.1111{n}v1", ordinal=1,
                        raw_text="[1] arXiv:2608.11110", cited_arxiv_id="2608.11110",
                    )
                ],
                source_sha256="abc123",
            )

    # Eight slots leave two (8 // 4) for the neighbourhood: the hub and one citer.
    export_sample(source, tmp_path / "sample.db", papers=8, model_id="stub")

    with CorpusStore(tmp_path / "sample.db") as sample:
        stats = sample.citation_graph_stats()
    assert stats["edges"] >= 1


def test_a_sampled_passage_is_searchable(tmp_path: Path) -> None:
    """A sample where search returns nothing is worse than no sample."""
    source = seeded(tmp_path)

    export_sample(source, tmp_path / "sample.db", papers=2, model_id="stub")

    with CorpusStore(tmp_path / "sample.db") as sample:
        matches = sample.search_chunks("qubits")

    assert matches


def test_columns_are_copied_by_name_not_position(tmp_path: Path) -> None:
    """Column order can differ between databases, so rows are copied by name."""
    source = seeded(tmp_path)

    export_sample(source, tmp_path / "sample.db", papers=1, model_id="stub")

    with CorpusStore(tmp_path / "sample.db") as sample:
        chunk = list(sample.iter_chunks())[0]

    assert chunk.arxiv_id.startswith("2608.")
    assert len(chunk.source_sha256) == 64
    assert "qubits" in chunk.text


def test_sampling_an_unembedded_corpus_says_so(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Body text."]}):
        pass

    with pytest.raises(ValueError, match="run 'verbatim embed' first"):
        export_sample(tmp_path / "corpus.db", tmp_path / "sample.db",
                      papers=1, model_id="stub")


def test_papers_must_be_at_least_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        export_sample(tmp_path / "corpus.db", tmp_path / "s.db", papers=0, model_id="stub")


def test_only_redistributable_papers_are_sampled(tmp_path: Path) -> None:
    """A sample ships the papers' own words, so only `open` papers qualify."""
    import sqlite3

    from verbatim.sampling import choose_papers

    source = sqlite3.connect(":memory:")
    source.row_factory = sqlite3.Row
    source.executescript(
        """
        CREATE TABLE papers (arxiv_id TEXT, base_arxiv_id TEXT, is_latest INT,
                             categories_json TEXT, license_class TEXT);
        CREATE TABLE chunks (chunk_id TEXT, arxiv_id TEXT, kind TEXT);
        CREATE TABLE embeddings (chunk_id TEXT, model_id TEXT);
        CREATE TABLE paper_references (citing_arxiv_id TEXT, cited_arxiv_id TEXT);
        """
    )
    for arxiv_id, licence in [
        ("2608.11111v1", "open"),
        ("2608.22222v1", "arxiv-nonexclusive"),
        ("2608.33333v1", "no-derivatives"),
        ("2608.44444v1", "unknown"),
    ]:
        source.execute(
            "INSERT INTO papers VALUES (?, ?, 1, '[\"cs.CL\"]', ?)",
            (arxiv_id, arxiv_id.split("v")[0], licence),
        )
        # Everyone cites the non-open papers, yet the shared licence gate keeps them out.
        source.execute(
            "INSERT INTO paper_references VALUES (?, '2608.22222')", (arxiv_id,)
        )
        for n in range(60):
            source.execute(
                "INSERT INTO chunks VALUES (?, ?, 'body')", (f"{arxiv_id}-{n}", arxiv_id)
            )
            source.execute(
                "INSERT INTO embeddings VALUES (?, 'stub')", (f"{arxiv_id}-{n}",)
            )

    chosen = choose_papers(source, papers=10, model_id="stub")

    assert chosen == ["2608.11111v1"]
