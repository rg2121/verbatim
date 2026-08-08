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


def test_hybrid_beats_either_leg_when_the_answer_needs_both(tmp_path: Path) -> None:
    """Exact identifier lives in one passage, the concept in another.

    Keyword alone misses the concept; vector alone misses the identifier.
    """
    from verbatim.retrieval import hybrid_search

    with embedded_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Magnetic flux threads the ring."],
            "2608.22222v1": ["Charge conservation holds in tokamak-7 exactly."],
        },
    ) as corpus:
        keyword = corpus.search_chunks("flux tokamak-7", match="any", limit=10)
        vector = vector_search(corpus, WordCountEmbedder(), "flux tokamak-7", limit=10)
        fused = hybrid_search(corpus, WordCountEmbedder(), "flux tokamak-7", limit=10)

    assert {m.chunk.arxiv_id for m in fused} == {"2608.11111v1", "2608.22222v1"}
    assert len(fused) >= max(len(keyword), len(vector))


def test_hybrid_ranks_a_passage_found_by_both_legs_first(tmp_path: Path) -> None:
    from verbatim.retrieval import hybrid_search

    with embedded_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Magnetic flux threads the ring."],
            "2608.22222v1": ["An unrelated plant grows."],
            "2608.33333v1": ["Charge moves along the wire."],
        },
    ) as corpus:
        fused = hybrid_search(corpus, WordCountEmbedder(), "flux", limit=3)

    assert fused[0].chunk.arxiv_id == "2608.11111v1"
    assert fused[0].score > fused[1].score


def test_hybrid_scopes_to_one_paper(tmp_path: Path) -> None:
    from verbatim.retrieval import hybrid_search

    with embedded_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Magnetic flux threads the ring."],
            "2608.22222v1": ["Charge moves along the wire."],
        },
    ) as corpus:
        fused = hybrid_search(
            corpus, WordCountEmbedder(), "flux charge", limit=5, arxiv_id="2608.22222v1"
        )

    assert {m.chunk.arxiv_id for m in fused} == {"2608.22222v1"}


def test_cap_per_paper_keeps_rank_order_and_promotes_other_papers(tmp_path: Path) -> None:
    from verbatim.corpus import ChunkMatch
    from verbatim.retrieval import cap_per_paper

    with chunked_corpus(
        tmp_path,
        {"2608.11111v1": ["flux one", "flux two", "flux three"], "2608.22222v1": ["charge one"]},
    ) as corpus:
        chunks = {c.arxiv_id: [] for c in corpus.all_chunks()}
        for c in corpus.all_chunks():
            chunks[c.arxiv_id].append(c)
        ranked = [
            ChunkMatch(chunk=chunks["2608.11111v1"][0], score=0.9, snippet=""),
            ChunkMatch(chunk=chunks["2608.11111v1"][1], score=0.8, snippet=""),
            ChunkMatch(chunk=chunks["2608.11111v1"][2], score=0.7, snippet=""),
            ChunkMatch(chunk=chunks["2608.22222v1"][0], score=0.6, snippet=""),
        ]

    capped = cap_per_paper(ranked, 2)

    assert [m.chunk.arxiv_id for m in capped] == [
        "2608.11111v1",
        "2608.11111v1",
        "2608.22222v1",
    ]
    assert [m.score for m in capped] == [0.9, 0.8, 0.6]


def test_hybrid_cap_promotes_a_second_paper_into_the_results(tmp_path: Path) -> None:
    from verbatim.retrieval import hybrid_search

    with embedded_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Magnetic flux one.", "Magnetic flux two.", "Magnetic flux three."],
            "2608.22222v1": ["Charge moves here."],
        },
    ) as corpus:
        uncapped = hybrid_search(corpus, WordCountEmbedder(), "flux charge", limit=3)
        capped = hybrid_search(
            corpus, WordCountEmbedder(), "flux charge", limit=3, max_per_paper=1
        )

    assert len({m.chunk.arxiv_id for m in capped}) >= len({m.chunk.arxiv_id for m in uncapped})
    assert sum(1 for m in capped if m.chunk.arxiv_id == "2608.11111v1") == 1


def test_index_filters_by_kind_and_paper_without_passing_ids(tmp_path: Path) -> None:
    """Filtering happens inside the KNN query, so it survives a large corpus."""
    from verbatim.chunking import REFERENCES
    from verbatim.corpus import CorpusStore

    with embedded_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Magnetic flux threads the ring."],
            "2608.22222v1": ["Charge moves along the wire."],
        },
    ) as corpus:
        # Mark one paper's chunk as a bibliography and reindex.
        corpus.connection.execute(
            "UPDATE chunks SET kind = ? WHERE arxiv_id = ?", (REFERENCES, "2608.22222v1")
        )
        corpus.connection.commit()
        index = SqliteVecIndex(corpus.connection, 3)
        index.rebuild("stub")
        query = WordCountEmbedder().embed_query("flux charge")

        everything = index.search(query, limit=10)
        body_only = index.search(query, limit=10, kind="body")
        one_paper = index.search(query, limit=10, arxiv_id="2608.22222v1")

    assert len(everything) == 2
    assert len(body_only) == 1
    assert len(one_paper) == 1


def test_an_index_built_before_metadata_columns_is_replaced(tmp_path: Path) -> None:
    with embedded_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux."]}) as corpus:
        SqliteVecIndex(corpus.connection, 3)  # loads the sqlite-vec extension
        # Simulate the older schema: no arxiv_id or kind columns.
        corpus.connection.execute("DROP TABLE vec_chunks")
        corpus.connection.execute(
            "CREATE VIRTUAL TABLE vec_chunks USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[3])"
        )
        corpus.connection.commit()

        index = SqliteVecIndex(corpus.connection, 3)  # should detect and replace
        index.rebuild("stub")
        hits = index.search(WordCountEmbedder().embed_query("flux"), limit=5, kind="body")

    assert len(hits) == 1
