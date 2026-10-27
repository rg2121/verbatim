"""Answering a compound question in parts."""

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
    """An embedder stand-in whose vector leg returns nothing."""

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


def test_the_passage_budget_is_shared_by_the_parts_actually_planned(monkeypatch) -> None:
    """A question the planner leaves whole gets the full budget, as plain `ask` does."""
    import verbatim.research as module

    limits: list[int] = []

    def record(corpus, embedder, part, **options):
        limits.append(options["limit"])
        return []

    monkeypatch.setattr(module, "hybrid_search", record)

    class Silent:
        def complete(self, system, user, schema):
            raise AssertionError("nothing was retrieved, so nothing is written")

    module.research(None, None, Silent(), reply("the whole question"), "q", passages=8)
    module.research(None, None, Silent(), reply("part one\npart two"), "q", passages=8)

    assert limits == [8, 4, 4]


def test_ask_plan_retrieves_what_its_evaluation_measured(monkeypatch) -> None:
    """The eval's hybrid+plan mode is the measurement behind `ask --plan`.

    The two split the budget separately once, and with a small budget the
    shipped path handed the writer twice the passages the measured one did.
    """
    from types import SimpleNamespace

    import verbatim.research as module
    from verbatim.evaluation import retrieval_modes

    def found(corpus, embedder, part, **options):
        return [SimpleNamespace(chunk=SimpleNamespace(chunk_id=f"{part}-{n}"))
                for n in range(options["limit"])]

    monkeypatch.setattr(module, "hybrid_search", found)
    monkeypatch.setattr(module, "answer_from_matches", lambda *args: None)
    plan = reply("one\ntwo\nthree")

    for budget in (3, 8, 10):
        measured = retrieval_modes(
            None, None, limit=budget, max_per_paper=2, complete=plan, decompose=True
        )["hybrid+plan"]("q")
        _, _, shipped = module.research(
            None, None, None, plan, "q", passages=budget, max_per_paper=2
        )

        ids = [m.chunk.chunk_id for m in shipped]
        assert ids == [m.chunk.chunk_id for m in measured]
        assert len(ids) <= budget
