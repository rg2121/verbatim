"""The roles, as contracts rather than as functions that happen to fit."""

from verbatim.answers import Usage
from verbatim.contracts import (
    Budget,
    BudgetGuard,
    Cost,
    Gathered,
    Plan,
    satisfies,
)


def test_a_plan_with_no_sections_is_refused() -> None:
    import pytest

    assert Plan("topic", ("what is it?",)).sections == ("what is it?",)
    with pytest.raises(ValueError, match="not a plan"):
        Plan("topic", ())


def test_gathered_names_the_sources_it_holds(matched=None) -> None:
    from verbatim.chunking import Chunk
    from verbatim.corpus import ChunkMatch

    def match(chunk_id: str, arxiv_id: str) -> ChunkMatch:
        chunk = Chunk(
            chunk_id=chunk_id, arxiv_id=arxiv_id, source_url="u", license_url=None,
            source_sha256="s", page_number=1, chunk_index=0, char_start=0,
            char_end=1, text="t", chunker_version="v",
        )
        return ChunkMatch(chunk=chunk, score=1.0, snippet="")

    gathered = Gathered(
        "q", (match("a", "2608.1v1"), match("b", "2608.1v1"), match("c", "2608.2v1")),
        strategy="hybrid",
    )

    assert gathered.chunk_ids == ("a", "b", "c")
    # Papers are deduplicated in order, so a ledger line stays readable.
    assert gathered.paper_ids == ("2608.1v1", "2608.2v1")


def test_a_role_can_be_checked_against_its_contract() -> None:
    """What makes the roles independently testable rather than only jointly."""

    class Verifier:
        def __call__(self, answer, closeness):
            from verbatim.contracts import Verdict

            return Verdict(True, "fine")

    assert satisfies("verifier", Verifier()) is True
    assert satisfies("verifier", "not callable") is False

    import pytest

    with pytest.raises(ValueError, match="unknown role"):
        satisfies("nonesuch", Verifier())


def test_a_step_is_costed_as_a_difference_not_a_total() -> None:
    """A run outlives a step, so a step's cost is what it added."""
    usage = Usage()
    usage.record({"prompt_eval_count": 900, "eval_count": 100, "total_duration": 9_000_000_000})
    before = usage.snapshot()
    usage.record({"prompt_eval_count": 10, "eval_count": 5, "total_duration": 1_000_000_000})

    step = usage.since(before)

    assert (step.calls, step.prompt_tokens, step.output_tokens) == (1, 10, 5)
    assert step.seconds == 1.0
    assert Cost(1, 10, 5, 1.0).render() == "1 calls, 15 tokens, 1.0s"


def test_a_budget_stops_a_run_on_whichever_ceiling_is_reached_first() -> None:
    usage = Usage()
    ticks = iter([0, 0, 1, 2, 3, 4, 5])
    guard = BudgetGuard(Budget(max_tokens=100), usage, clock=lambda: next(ticks))

    assert guard.exhausted() is None
    usage.record({"prompt_eval_count": 60, "eval_count": 10})
    assert guard.exhausted() is None          # 70 spent, under the ceiling
    usage.record({"prompt_eval_count": 40, "eval_count": 0})
    assert "110 tokens" in guard.exhausted()  # over it

    # Time is measured from construction, not from the first attempt.
    slow = BudgetGuard(Budget(max_seconds=10), None, clock=iter([0, 11]).__next__)
    assert "reached the limit" in slow.exhausted()


def test_an_unbounded_budget_never_stops_anything() -> None:
    guard = BudgetGuard(Budget(), Usage())
    for _ in range(50):
        guard.take()

    assert guard.exhausted() is None
