from pathlib import Path

import pytest

from verbatim.corpus import CorpusStore, EmbeddingModel
from verbatim.retrieval import _similarity, vector_search
from verbatim.vectors import SqliteVecIndex, ensure_current

from helpers import chunked_corpus


class WordCountEmbedder:
    """Vectors that are trivially predictable, so ranking is checkable by hand."""

    def __init__(self) -> None:
        self.model = EmbeddingModel(
            model_id="stub", dimensions=3, normalized=True, revision="v0"
        )

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        raw = [
            float("flux" in lowered),
            float("charge" in lowered),
            float("plant" in lowered),
        ]
        norm = sum(value * value for value in raw) ** 0.5 or 1.0
        return [value / norm for value in raw]

    def embed_passages(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def embedded_corpus(tmp_path: Path, pages_by_paper):
    from verbatim.embedding import embed_pending

    corpus = chunked_corpus(tmp_path, pages_by_paper)
    embed_pending(corpus, WordCountEmbedder())
    return corpus


def test_vector_search_ranks_by_meaning_not_shared_words(tmp_path: Path) -> None:
    with embedded_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Magnetic flux threads the ring."],
            "2608.22222v1": ["A plant converts sunlight."],
        },
    ) as corpus:
        matches = vector_search(corpus, WordCountEmbedder(), "flux", limit=2)

    assert matches[0].chunk.arxiv_id == "2608.11111v1"
    assert matches[0].score > matches[1].score


def test_vector_search_can_be_scoped_to_one_paper(tmp_path: Path) -> None:
    with embedded_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Magnetic flux threads the ring."],
            "2608.22222v1": ["Charge moves through the wire."],
        },
    ) as corpus:
        scoped = vector_search(
            corpus, WordCountEmbedder(), "flux", limit=5, arxiv_id="2608.22222v1"
        )

    assert [match.chunk.arxiv_id for match in scoped] == ["2608.22222v1"]


def test_vector_search_returns_nothing_before_anything_is_embedded(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux."]}) as corpus:
        assert vector_search(corpus, WordCountEmbedder(), "flux") == []


def test_index_rebuilds_only_when_it_has_drifted(tmp_path: Path) -> None:
    with embedded_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux.", "Charge moves."]}) as corpus:
        index = SqliteVecIndex(corpus.connection, 3)
        assert ensure_current(index, corpus.embedding_count("stub"), "stub") is True
        assert index.count() == 2
        assert ensure_current(index, corpus.embedding_count("stub"), "stub") is False


def test_index_rejects_a_query_of_the_wrong_width(tmp_path: Path) -> None:
    with embedded_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux."]}) as corpus:
        index = SqliteVecIndex(corpus.connection, 3)
        index.rebuild("stub")

        with pytest.raises(ValueError, match="expects 3"):
            index.search([1.0, 0.0], limit=1)


def test_similarity_inverts_l2_distance_for_unit_vectors() -> None:
    assert _similarity(0.0) == pytest.approx(1.0)  # identical
    assert _similarity(2.0**0.5) == pytest.approx(0.0)  # orthogonal
    assert _similarity(2.0) == pytest.approx(-1.0)  # opposite
