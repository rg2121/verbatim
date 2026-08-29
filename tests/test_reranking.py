import pytest

from verbatim.chunking import Chunk
from verbatim.corpus import ChunkMatch
from verbatim.reranking import rerank


def match(chunk_id: str, text: str, score: float) -> ChunkMatch:
    chunk = Chunk(
        chunk_id=chunk_id, arxiv_id="2608.11111v1", source_url="u", license_url=None,
        source_sha256="s", page_number=1, chunk_index=0, char_start=0, char_end=len(text),
        text=text, chunker_version="6:1200:200",
    )
    return ChunkMatch(chunk=chunk, score=score, snippet=text[:40])


class KeywordReranker:
    """Scores by how many query words a passage contains — predictable by hand."""

    def score(self, query, passages):
        wanted = set(query.lower().split())
        return [len(wanted & set(p.lower().split())) / len(wanted) for p in passages]


def test_reranking_promotes_the_passage_that_actually_matches() -> None:
    """Retrieval ranked the wrong one first; the cross-encoder should fix it."""
    matches = [
        match("a", "unrelated discussion of lattice surrogate models", 0.9),
        match("b", "nuclear fission releases energy from a nucleus", 0.1),
    ]

    reranked = rerank(KeywordReranker(), "nuclear fission energy", matches)

    assert [m.chunk.chunk_id for m in reranked] == ["b", "a"]
    assert reranked[0].score > reranked[1].score


def test_reranking_reports_the_cross_encoder_score_not_the_fused_one() -> None:
    matches = [match("a", "nuclear fission energy", 0.42)]

    (reranked,) = rerank(KeywordReranker(), "nuclear fission energy", matches)

    assert reranked.score == 1.0          # all three query words present
    assert reranked.chunk.chunk_id == "a"


def test_reranking_can_trim_to_a_limit() -> None:
    matches = [
        match("a", "nuclear fission", 0.5),
        match("b", "fission energy nuclear", 0.5),
        match("c", "something else entirely", 0.5),
    ]

    reranked = rerank(KeywordReranker(), "nuclear fission energy", matches, limit=2)

    assert len(reranked) == 2
    assert "c" not in [m.chunk.chunk_id for m in reranked]


def test_reranking_nothing_returns_nothing() -> None:
    assert rerank(KeywordReranker(), "anything", []) == []


def test_a_reranker_returning_the_wrong_count_is_an_error() -> None:
    class Broken:
        def score(self, query, passages):
            return [1.0]

    with pytest.raises(RuntimeError, match="1 scores for 2 passages"):
        rerank(Broken(), "q", [match("a", "x", 1.0), match("b", "y", 1.0)])
