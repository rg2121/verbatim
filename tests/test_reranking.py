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


def test_the_reranker_uses_the_same_persistent_model_cache(monkeypatch, tmp_path) -> None:
    import fastembed.rerank.cross_encoder as module

    from verbatim import config
    from verbatim.reranking import CrossEncoderReranker

    seen: dict = {}

    class FakeCrossEncoder:
        def __init__(self, model_name, **options):
            seen.update(options)

        def rerank(self, query, passages):
            return [0.0 for _ in passages]

    monkeypatch.setattr(module, "TextCrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(config, "MODEL_CACHE", tmp_path / "models")

    CrossEncoderReranker().score("q", ["a passage"])

    assert seen["cache_dir"] == str(tmp_path / "models")


def test_maxsim_sums_each_query_tokens_best_match() -> None:
    import numpy as np

    from verbatim.reranking import maxsim

    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    passage = np.array([[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]])

    # First query token's best match is 1.0, the second's is 1.0.
    assert maxsim(query, passage) == pytest.approx(2.0)
    assert maxsim(query, np.array([[0.5, 0.5]])) == pytest.approx(1.0)


def test_late_interaction_scores_each_passage_by_maxsim() -> None:
    import numpy as np

    from verbatim.reranking import LateInteractionReranker

    class TokenVectors:
        """Two-dimensional 'tokens': fission words point one way, anything else the other."""

        def _tokens(self, text):
            return np.array([[1.0, 0.0] if w in {"fission", "nucleus"} else [0.0, 1.0]
                             for w in text.split()])

        def query_embed(self, queries):
            return (self._tokens(q) for q in queries)

        def embed(self, passages, batch_size=256):
            return (self._tokens(p) for p in passages)

    reranker = LateInteractionReranker(_encoder=TokenVectors())
    scores = reranker.score("fission", ["lattice models", "fission of a nucleus"])

    assert scores[1] > scores[0]
    assert LateInteractionReranker(_encoder=None).score("q", []) == []


def test_late_interaction_encodes_passages_one_at_a_time() -> None:
    """A batch pads every passage to the longest; at 50 that doubled the time and
    pushed the corpus out of the page cache, slowing the searches after it."""
    import numpy as np

    from verbatim.reranking import LateInteractionReranker

    batches = []

    class Encoder:
        def query_embed(self, queries):
            return (np.ones((2, 2)) for _ in queries)

        def embed(self, passages, batch_size=256):
            batches.append(batch_size)
            return (np.ones((3, 2)) for _ in passages)

    LateInteractionReranker(_encoder=Encoder()).score("q", ["a", "b", "c"])

    assert batches == [1]


def test_rerankers_are_chosen_by_name() -> None:
    from verbatim.reranking import (
        CrossEncoderReranker,
        LateInteractionReranker,
        make_reranker,
    )

    assert isinstance(make_reranker("cross"), CrossEncoderReranker)
    assert isinstance(make_reranker("colbert"), LateInteractionReranker)
    with pytest.raises(ValueError, match="colbert"):
        make_reranker("bm42")


def test_the_default_search_reranks_as_it_was_measured(monkeypatch) -> None:
    """ColBERT, only where the legs disagree, never on a definition (docs/findings.md)."""
    from verbatim import config
    from verbatim.reranking import LateInteractionReranker
    from verbatim.retrieval import default_search_options

    monkeypatch.setattr(config, "RERANKER", "colbert")
    options = default_search_options()

    assert isinstance(options["reranker"], LateInteractionReranker)
    assert options["use_graph"] is True
    assert options["rerank_when_legs_disagree"] is True
    assert options["rerank_skips_definitions"] is True


def test_the_default_reranker_is_built_once_per_process(monkeypatch) -> None:
    """A server would otherwise load the model again for every request."""
    from verbatim import config
    from verbatim.reranking import default_reranker

    monkeypatch.setattr(config, "RERANKER", "colbert")

    assert default_reranker() is default_reranker()


def test_the_default_reranker_can_be_turned_off(monkeypatch) -> None:
    from verbatim import config
    from verbatim.retrieval import default_search_options

    monkeypatch.setattr(config, "RERANKER", "off")

    assert default_search_options()["reranker"] is None
    assert default_search_options()["use_graph"] is True


def test_an_unknown_default_reranker_names_the_setting(monkeypatch) -> None:
    from verbatim import config
    from verbatim.reranking import default_reranker

    monkeypatch.setattr(config, "RERANKER", "bm42")

    with pytest.raises(ValueError, match="VERBATIM_RERANKER"):
        default_reranker()
