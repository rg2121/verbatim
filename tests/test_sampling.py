"""Exporting a small corpus that works without an ingest."""

from pathlib import Path

import pytest

from helpers import chunked_corpus
from verbatim.corpus import CorpusStore, EmbeddingModel
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
            "stub", [(c.chunk_id, [1.0, 0.0, 0.0, 0.0]) for c in corpus.all_chunks()]
        )
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
        chunks = sample.all_chunks()

    assert downloads == 0
    # The passages still carry everything a citation needs.
    assert all(chunk.source_url for chunk in chunks)


def test_a_sampled_passage_is_searchable(tmp_path: Path) -> None:
    """A sample where search returns nothing is worse than no sample."""
    source = seeded(tmp_path)

    export_sample(source, tmp_path / "sample.db", papers=2, model_id="stub")

    with CorpusStore(tmp_path / "sample.db") as sample:
        matches = sample.search_chunks("qubits")

    assert matches


def test_columns_are_copied_by_name_not_position(tmp_path: Path) -> None:
    """This corpus has a rebuilt chunks table, so the orders disagree.

    Copying positionally puts each value in the wrong column and reports
    success, which is worse than failing.
    """
    source = seeded(tmp_path)

    export_sample(source, tmp_path / "sample.db", papers=1, model_id="stub")

    with CorpusStore(tmp_path / "sample.db") as sample:
        chunk = sample.all_chunks()[0]

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
