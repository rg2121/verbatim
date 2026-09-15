"""Answering a compound question in parts.

The loop is off by default and must never make a question worse, so most of
these are about it declining to interfere or failing safely.
"""

import pytest

from verbatim.research import MAX_PARTS, gather_for_parts, plan_subquestions


def reply(text: str):
    return lambda system, user: text


def test_a_compound_question_is_split_into_its_parts() -> None:
    parts = plan_subquestions(
        reply("what does MOND predict in clusters\nwhat does dark matter predict"),
        "how do MOND and dark matter differ in galaxy clusters?",
    )

    assert parts == [
        "what does MOND predict in clusters",
        "what does dark matter predict",
    ]


def test_a_single_subject_question_is_left_alone() -> None:
    """Splitting one that needs no splitting buys three mediocre retrievals."""
    assert plan_subquestions(reply("what is entropy?"), "what is entropy?") == [
        "what is entropy?"
    ]


def test_planning_failure_falls_back_to_the_question_asked() -> None:
    def broken(system: str, user: str) -> str:
        raise RuntimeError("Cannot reach Ollama")

    assert plan_subquestions(broken, "what is entropy?") == ["what is entropy?"]
    assert plan_subquestions(reply("  \n  "), "what is entropy?") == ["what is entropy?"]


def test_a_runaway_plan_is_capped() -> None:
    parts = plan_subquestions(reply("\n".join(f"part {n}" for n in range(20))), "q")

    assert len(parts) <= MAX_PARTS


def test_repeated_parts_are_not_retrieved_for_twice() -> None:
    parts = plan_subquestions(reply("what is entropy\nWhat is  entropy\nother"), "q")

    assert parts == ["what is entropy", "other"]


def test_every_part_keeps_its_share_of_the_passages(tmp_path) -> None:
    """Fusing lets the better-matching half crowd the other out entirely."""
    from helpers import chunked_corpus

    with chunked_corpus(
        tmp_path,
        {
            "2608.11111v1": ["MOND explains cluster dynamics without dark matter."],
            "2608.22222v1": ["Dark matter haloes dominate cluster mass profiles."],
        },
    ) as corpus:
        matches = gather_for_parts(
            corpus,
            _KeywordOnly(corpus),
            ["MOND clusters", "dark matter haloes"],
            per_part=2,
        )

    assert {m.chunk.arxiv_id for m in matches} == {"2608.11111v1", "2608.22222v1"}


class _KeywordOnly:
    """An embedder stand-in whose vector leg returns nothing.

    hybrid_search tolerates an empty vector leg, so this exercises the quota
    without needing a real model.
    """

    def __init__(self, corpus) -> None:
        from verbatim.corpus import EmbeddingModel

        self.model = EmbeddingModel(
            model_id="stub", dimensions=4, normalized=True, revision="v0"
        )

    def embed_passages(self, texts):
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0, 0.0, 0.0, 0.0]


def test_max_parts_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        plan_subquestions(reply("x"), "q", max_parts=0)


def test_the_passages_used_are_returned_so_the_subject_check_still_runs(tmp_path) -> None:
    """Returning only an answer would disable the one safeguard that works."""
    from helpers import chunked_corpus
    from verbatim.research import research

    class StubCompletion:
        def complete(self, system, user, schema):
            return {"answer": "Nothing to say.", "citations": []}

    with chunked_corpus(tmp_path, {"2608.11111v1": ["MOND in clusters."]}) as corpus:
        answer, parts, matches = research(
            corpus, _KeywordOnly(corpus), StubCompletion(),
            reply("MOND clusters"), "what about MOND?",
        )

    assert parts == ["MOND clusters"]
    assert [m.chunk.arxiv_id for m in matches] == ["2608.11111v1"]


def test_the_search_flags_reach_every_part(monkeypatch) -> None:
    """`ask --plan` parsed --arxiv-id, --no-graph and the rest, then dropped them.

    A question scoped to one paper was answered from the whole corpus.
    """
    import verbatim.research as module

    seen: list[dict] = []

    def record(corpus, embedder, part, **options):
        seen.append(options)
        return []

    monkeypatch.setattr(module, "hybrid_search", record)

    class Silent:
        def complete(self, system, user, schema):
            raise AssertionError("nothing was retrieved, so nothing is written")

    module.research(
        None, None, Silent(), reply("part one\npart two"), "q",
        max_per_paper=5, arxiv_id="2608.11111v1", include_references=True,
        use_graph=False, use_compare=True,
    )

    assert len(seen) == 2
    for options in seen:
        assert options["arxiv_id"] == "2608.11111v1"
        assert options["max_per_paper"] == 5
        assert options["include_references"] is True
        assert options["use_graph"] is False and options["use_compare"] is True
