"""M2d: the answer graph, in LangGraph, on scripted runs.

The opening search and the subject check run as v1's plain ask runs them; the
Lead decides each step through the real line protocol and contract; the
Writer and Verifier are M2c's attempt, as nodes; assembly returns a legal
attempt, and every run ends in a terminal state (I1-I5, I9-I11).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from helpers import chunked_corpus
from verbatim.agents.contract import (
    ABSTAINED,
    BUDGET_EXHAUSTED,
    COMPLETE,
    ERROR,
    PROTOCOL_FAILURE,
    marking,
)
from verbatim.agents.graphs import Deps, run
from verbatim.agents.runtime import checkpoints
from verbatim.agents.scoped import ScopedCorpus
from verbatim.answers import Usage, answer_from_matches
from verbatim.contracts import Budget
from verbatim.retrieval import default_search_options, hybrid_search

FIRST, SECOND = "2608.11111v1", "2608.22222v1"
PASSAGE = ("Transformers replace recurrence with attention. "
           "The model reaches 28.4 BLEU on the English-to-German translation task. "
           "Training took three and a half days on eight GPUs.")
OTHER = "Dropout of 0.1 was applied to the output of each sub-layer of the model."
QUESTION = "How well does the model translate?"
GOOD = "The model reaches 28.4 BLEU on the English-to-German translation task"
SEARCH = {"limit": 8, **default_search_options()}


class Embedder:
    """No vectors are stored, so the vector leg finds nothing; every text embeds alike."""

    model = SimpleNamespace(model_id="none", dimensions=3)

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class Model:
    """Every role's model at once, replying from one script in call order and
    recording each call: (kind, system, user)."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str, str]] = []
        self.usage = Usage()

    def for_role(self, name: str) -> Model:
        return self

    def complete(self, system: str, user: str, schema: dict) -> Any:
        return self._reply("complete", system, user)

    def revise(self, system: str, user: str, schema: dict) -> Any:
        return self._reply("revise", system, user)

    def complete_text(self, system: str, user: str) -> Any:
        return self._reply("text", system, user)

    def judge(self, system: str, user: str) -> Any:
        return self._reply("judge", system, user)

    def _reply(self, kind: str, system: str, user: str) -> Any:
        self.calls.append((kind, system, user))
        self.usage.record({"prompt_eval_count": 10, "eval_count": 5})
        if not self.replies:
            raise AssertionError(f"the script ran out at a {kind} call")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def store(tmp_path: Path):
    corpus = chunked_corpus(tmp_path, {FIRST: [PASSAGE], SECOND: [OTHER]})
    with corpus:
        yield corpus


def chunk_of(store, paper: str) -> str:
    return store.chunks_for_paper(paper)[0].chunk_id


def cite(answer: str, *citations: tuple[int, str, str]) -> dict:
    return {"answer": answer, "citations": [
        {"marker": marker, "chunk_id": chunk_id, "quoted_text": quote}
        for marker, chunk_id, quote in citations]}


def ask(store, model: Model, *, hidden=(), budget: Budget | None = None, **options: Any):
    deps = Deps(scope=ScopedCorpus(store, hidden), embedder=Embedder(), model=model,
                search=SEARCH, budget=budget or Budget(), **options)
    return run(QUESTION, deps)


# -- the shortest path, and where it matches v1 -----------------------------------------


def test_the_shortest_path_is_v1s_plain_ask_and_two_lead_calls(store) -> None:
    first = chunk_of(store, FIRST)
    model = Model("Machine translation results", "NEXT: writer", cite(f"{GOOD} [1].", (1, first, GOOD)),
                  "FINAL: 1")
    outcome = ask(store, model)
    assert outcome.status == COMPLETE and outcome.returned == 1
    assert [kind for kind, _, _ in model.calls] == ["judge", "text", "complete", "text"]  # 4 calls
    # I3: the Writer's call is v1's plain ask, word for word, on v1's own search.
    v1 = Model(cite(f"{GOOD} [1].", (1, first, GOOD)))
    answer_from_matches(store, v1, QUESTION, hybrid_search(store, Embedder(), QUESTION, **SEARCH),
                        reflection=False)
    assert model.calls[2] == v1.calls[0]
    # I4: what is shown is the checked attempt, unreworded.
    assert outcome.answer.text == f"{GOOD} [1]." and len(outcome.answer.citations) == 1
    # I5: every model call is a ledger step, in the order made.
    assert [step.name for step in outcome.ledger] == ["describer", "lead", "writer", "lead"]
    assert sum(step.cost.calls for step in outcome.ledger) == 4


def test_the_lead_reads_the_facts_and_may_inspect_a_passage(store) -> None:
    first = chunk_of(store, FIRST)
    model = Model("Machine translation results", f"INSPECT: {first}", "NEXT: writer",
                  cite(f"{GOOD} [1].", (1, first, GOOD)), "FINAL: 1")
    outcome = ask(store, model)
    assert outcome.status == COMPLETE
    first_view, second_view = model.calls[1][2], model.calls[2][2]
    assert "Machine translation results" in first_view and first in first_view
    # The inspected passage comes back word for word, marked as data (I7).
    assert f'<passage id="{first}">{PASSAGE}</passage>' in second_view
    assert "<passage" not in first_view.split("PASSAGES ON THE BOARD")[0]


# -- repairs, and I10 -------------------------------------------------------------------


def failed_draft(store) -> dict:
    first, second = chunk_of(store, FIRST), chunk_of(store, SECOND)
    return cite(f"{GOOD} [1]. Dropout was applied to every sub-layer [2].",
                (1, first, GOOD), (2, second, "Dropout of 0.2 was applied"))


def test_a_draft_the_check_flags_goes_to_the_verifier_and_back_for_repair(store) -> None:
    first = chunk_of(store, FIRST)
    repaired = cite(f"{GOOD} [1].", (1, first, GOOD))
    model = Model("Machine translation results", "NEXT: writer", failed_draft(store),
                  "ACTION: [2] repair\nNEXT: writer", repaired, "FINAL: 2")
    outcome = ask(store, model)
    assert outcome.status == COMPLETE and outcome.returned == 2
    assert [kind for kind, _, _ in model.calls] == ["judge", "text", "complete", "text", "revise", "text"]
    assert "What the checker found:" in model.calls[4][2]  # v1's repair prompt
    assert [d.revises for d in outcome.state["drafts"]] == [None, 1]


def test_a_revision_that_loses_a_verified_passage_cannot_be_returned(store) -> None:
    second = chunk_of(store, SECOND)
    worse = cite("Dropout of 0.1 was applied to the output of each sub-layer of the model [1].",
                 (1, second, "Dropout of 0.1 was applied to the output of each sub-layer of the model"))
    model = Model("Machine translation results", "NEXT: writer", failed_draft(store),
                  "ACTION: [2] repair\nNEXT: writer", worse, "FINAL: 2", "FINAL: 1")
    outcome = ask(store, model)
    assert outcome.status == COMPLETE and outcome.returned == 1
    assert "loses the verified citation" in model.calls[-1][2]  # the refusal went back to the Lead


def test_a_dropped_citation_is_shown_unverified_never_verified(store) -> None:
    model = Model("Machine translation results", "NEXT: writer", failed_draft(store),
                  "ACTION: [2] drop\nNEXT: done", "FINAL: 1")
    outcome = ask(store, model)
    assert outcome.status == COMPLETE
    # I1: the dropped citation is not listed, and its claim is marked unverified.
    assert len(outcome.answer.citations) == 1 and outcome.answer.rejected == []
    assert outcome.answer.text == f"{GOOD} [1]. Dropout was applied to every sub-layer [unverified]."
    assert outcome.tally.verified == 1


# -- every terminal state (I2) ----------------------------------------------------------


def test_the_lead_may_abstain(store) -> None:
    model = Model("Optimizer schedules", "ABSTAIN: the passages discuss another subject")
    outcome = ask(store, model)
    assert outcome.status == ABSTAINED and outcome.answer is None
    assert outcome.reason == "the passages discuss another subject"
    assert marking(outcome.status)


def test_a_run_out_of_calls_returns_its_best_attempt_marked_incomplete(store) -> None:
    first = chunk_of(store, FIRST)
    model = Model("Machine translation results", "NEXT: writer", cite(f"{GOOD} [1].", (1, first, GOOD)))
    outcome = ask(store, model, budget=Budget(max_calls=3))
    assert outcome.status == BUDGET_EXHAUSTED and outcome.returned == 1
    assert len(model.calls) == 3 and marking(outcome.status) == "incomplete (budget)"


def test_a_retry_cannot_pass_the_call_budget(store) -> None:
    # The Lead's reply is unusable, and its one retry would be the fourth call of three.
    model = Model("Machine translation results", "I am thinking.")
    outcome = ask(store, model, budget=Budget(max_calls=2))
    assert outcome.status == BUDGET_EXHAUSTED and len(model.calls) == 2


def test_a_lead_that_cannot_reply_in_the_format_ends_the_run(store) -> None:
    first = chunk_of(store, FIRST)
    model = Model("Machine translation results", "NEXT: writer", cite(f"{GOOD} [1].", (1, first, GOOD)),
                  "I think we are done.", "Yes, finished.")
    outcome = ask(store, model)
    assert outcome.status == PROTOCOL_FAILURE and outcome.returned == 1
    assert marking(outcome.status) == "incomplete (protocol)"


def test_a_run_that_fails_ends_in_error_with_no_answer(store) -> None:
    model = Model("Machine translation results", "NEXT: writer", RuntimeError("the model server went away"))
    outcome = ask(store, model)
    assert outcome.status == ERROR and outcome.answer is None
    assert "the model server went away" in outcome.reason


def test_with_nothing_found_the_writer_is_refused_and_the_lead_abstains(store) -> None:
    model = Model("NEXT: writer", "ABSTAIN: nothing in the corpus matches")
    deps = Deps(scope=ScopedCorpus(store, [FIRST, SECOND]), embedder=Embedder(), model=model,
                search=SEARCH)
    outcome = run(QUESTION, deps)
    assert outcome.status == ABSTAINED
    assert "no passages to write from" in model.calls[1][2]  # no subject check on nothing


# -- I9, I11 and checkpoints ------------------------------------------------------------


def without_times(answer):
    """An answer with its citations' retrieval times taken out: no two runs share them."""
    return replace(answer, citations=[replace(c, retrieved_at="") for c in answer.citations])


def test_the_same_replies_give_the_same_state(store) -> None:
    # I9, compared without the times no two runs share: seconds and retrieval stamps.
    def once():
        first = chunk_of(store, FIRST)
        model = Model("Machine translation results", "NEXT: writer", failed_draft(store),
                      "ACTION: [2] repair\nNEXT: writer", cite(f"{GOOD} [1].", (1, first, GOOD)), "FINAL: 2")
        state = ask(store, model).state
        return {**state,
                "ledger": [replace(s, cost=replace(s.cost, seconds=0.0)) for s in state["ledger"]],
                "drafts": [replace(d, answer=without_times(d.answer)) for d in state["drafts"]],
                "result": without_times(state["result"])}

    assert once() == once()


def test_a_hidden_paper_never_reaches_the_lead_or_the_writer(store) -> None:
    second = chunk_of(store, SECOND)
    model = Model("Machine translation results", f"INSPECT: {chunk_of(store, FIRST)}", "NEXT: writer",
                  cite("Dropout of 0.1 was applied to the output of each sub-layer of the model [1].",
                       (1, second, "Dropout of 0.1 was applied to the output of each sub-layer of the model")),
                  "FINAL: 1")
    outcome = ask(store, model, hidden=[FIRST])
    assert outcome.status == COMPLETE
    assert all("2608.11111" not in user and "28.4 BLEU" not in user for _, _, user in model.calls)


def test_a_run_saved_to_a_checkpoint_reads_back_as_it_ended(store, tmp_path: Path) -> None:
    first = chunk_of(store, FIRST)
    model = Model("Machine translation results", "NEXT: writer", cite(f"{GOOD} [1].", (1, first, GOOD)),
                  "FINAL: 1")
    with checkpoints(tmp_path / "checkpoints.db") as saver:
        deps = Deps(scope=ScopedCorpus(store), embedder=Embedder(), model=model, search=SEARCH)
        outcome = run(QUESTION, deps, saver=saver, thread_id="job-1")
        from verbatim.agents.graphs import build_ask
        saved = build_ask(saver).get_state({"configurable": {"thread_id": "job-1"}}).values
    assert outcome.status == COMPLETE and saved["status"] == COMPLETE and saved["returned"] == 1


# -- the paths between: problems, caps and the Lead's passages ---------------------------


def test_a_call_that_fails_is_still_counted(store) -> None:
    from verbatim.answers import FormatError

    first = chunk_of(store, FIRST)
    model = Model("Machine translation results", "NEXT: writer", FormatError("no ANSWER line", "hmm"),
                  cite(f"{GOOD} [1].", (1, first, GOOD)), "FINAL: 1")
    outcome = ask(store, model)
    assert outcome.status == COMPLETE
    assert [step.name for step in outcome.ledger] == ["describer", "lead", "writer", "writer", "lead"]


def test_a_writer_that_cannot_write_in_the_format_hands_back_to_the_lead(store) -> None:
    from verbatim.answers import FormatError

    model = Model("Machine translation results", "NEXT: writer", FormatError("no", "a"),
                  FormatError("no", "b"), "ABSTAIN: the writer could not answer")
    outcome = ask(store, model)
    assert outcome.status == ABSTAINED
    assert "writer: its write could not be read, twice" in model.calls[-1][2]


def test_the_verifiers_research_request_reaches_the_lead_as_a_note(store) -> None:
    model = Model("Machine translation results", "NEXT: writer", failed_draft(store),
                  "ACTION: [2] keep_unverified\nNEXT: done\nRESEARCH: who measured dropout", "FINAL: 1")
    outcome = ask(store, model)
    assert outcome.status == COMPLETE
    assert "verifier: on attempt 1, asks for research: who measured dropout" in model.calls[-1][2]


def test_a_repair_goes_to_the_lead_not_back_to_the_verifier(store) -> None:
    # v1 repairs once; a flagged repair is the Lead's to judge, not another loop.
    first = chunk_of(store, FIRST)
    flagged = cite(f"{GOOD} [1]. Larger ensembles would generalise better across languages.",
                   (1, first, GOOD))
    model = Model("Machine translation results", "NEXT: writer", failed_draft(store),
                  "ACTION: [2] repair\nNEXT: writer", flagged, "FINAL: 2")
    outcome = ask(store, model)
    assert outcome.status == COMPLETE
    assert [kind for kind, _, _ in model.calls][-2:] == ["revise", "text"]


def test_the_lead_can_give_the_writer_other_passages(store) -> None:
    second = chunk_of(store, SECOND)
    model = Model("Machine translation results", f"NEXT: writer\nPASSAGES: {second}",
                  cite("Dropout of 0.1 was applied to the output of each sub-layer of the model [1].",
                       (1, second, "Dropout of 0.1 was applied to the output of each sub-layer of the model")),
                  "FINAL: 1")
    outcome = ask(store, model)
    assert outcome.status == COMPLETE
    writer_prompt = model.calls[2][2]
    assert second in writer_prompt and chunk_of(store, FIRST) not in writer_prompt


def test_the_leads_turns_and_the_writers_attempts_are_capped(store, monkeypatch) -> None:
    from verbatim.agents import graphs

    first = chunk_of(store, FIRST)
    monkeypatch.setattr(graphs, "MAX_LEAD_TURNS", 2)
    looping = Model("Machine translation results", f"INSPECT: {first}", f"INSPECT: {first}")
    outcome = ask(store, looping)
    assert outcome.status == BUDGET_EXHAUSTED and "turns" in outcome.reason
    monkeypatch.setattr(graphs, "MAX_LEAD_TURNS", 12)
    one_attempt = Model("Machine translation results", "NEXT: writer", failed_draft(store),
                        "ACTION: [2] repair\nNEXT: writer")
    outcome = ask(store, one_attempt, budget=Budget(max_attempts=1))
    assert outcome.status == BUDGET_EXHAUSTED and outcome.returned == 1
    assert "attempts reached the limit" in outcome.reason
