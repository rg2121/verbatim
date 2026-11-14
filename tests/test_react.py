"""M2e: the Researcher and the Skeptic, ReAct subgraphs from build_agent(spec).

An agent calls one of its tools per turn in the line protocol, code runs the tool
through the scope (I7, I11), what the tool finds goes on the board with how it was
found, and each model and tool call is a ledger step (I5). The Lead queues them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from test_ask_graph import (  # noqa: F401  (store is a fixture)
    FIRST,
    GOOD,
    QUESTION,
    SEARCH,
    Embedder,
    Model,
    chunk_of,
    cite,
    store,
)
from verbatim.agents.contract import COMPLETE
from verbatim.agents.graphs import Deps, run
from verbatim.agents.protocol import ProtocolError, parse_step
from verbatim.agents.react import build_agent
from verbatim.agents.roster import RESEARCHER, SKEPTIC
from verbatim.agents.scoped import ScopedCorpus
from verbatim.agents.state import Allowance, Done, ToolCall
from verbatim.tools import TOOL_SCHEMAS

TOOLS = {schema["name"]: schema for schema in TOOL_SCHEMAS}


def research(store, model: Model, spec=RESEARCHER, *, hidden=(), allowance=Allowance(),
             brief: str = "find the BLEU score") -> dict:
    deps = Deps(scope=ScopedCorpus(store, hidden), embedder=Embedder(), model=model, search=SEARCH)
    return build_agent(spec).invoke(
        {"question": QUESTION, "brief": brief, "at": [], "allowance": allowance}, context=deps)


# -- the protocol -----------------------------------------------------------------------


def test_a_tool_call_is_read_with_its_arguments_in_their_types() -> None:
    # Arguments are read after the CALL line: reasoning before it can look like one.
    reply = ("I should search first.\nquery: an earlier thought\nCALL: **search_chunks**\n"
             "Query: BLEU on English-German\nlimit: 5 passages\narxiv_id: `2608.11111v1`")
    assert parse_step(reply, TOOLS) == ToolCall("search_chunks", (
        ("query", "BLEU on English-German"), ("limit", 5), ("arxiv_id", "2608.11111v1")))
    assert parse_step("CALL: read_pages\narxiv_id: 2608.11111v1\npages: 2, 3", TOOLS) == ToolCall(
        "read_pages", (("arxiv_id", "2608.11111v1"), ("pages", (2, 3))))
    assert parse_step("DONE: passage abc reports 28.4 BLEU", TOOLS) == Done("passage abc reports 28.4 BLEU")


@pytest.mark.parametrize(("reply", "says"), [
    ("CALL: search_everything\nquery: x", "not one of your tools"),
    ("CALL: search_chunks", "needs query"),
    ("CALL: search_chunks\nquery: x\nlimit: many", "limit takes a number"),
    ("Let me think about it.", "exactly one"),
    ("DONE:", "what you found"),
])
def test_a_step_that_cannot_be_read_says_how_to_fix_it(reply: str, says: str) -> None:
    with pytest.raises(ProtocolError, match=says):
        parse_step(reply, TOOLS)


# -- one agent, run alone ---------------------------------------------------------------


def test_the_researcher_searches_and_reports_what_it_found(store) -> None:
    model = Model("CALL: search_chunks\nquery: BLEU translation", "DONE: 28.4 BLEU on English-German")
    out = research(store, model)
    first = chunk_of(store, FIRST)
    assert out["status"] == "done" and out["note"] == "28.4 BLEU on English-German"
    found = {e.chunk_id: e for e in out["board"]}[first].found
    assert [(f.agent, f.query, f.stance) for f in found] == [("researcher", "BLEU translation", "direct")]
    # I5: each model call and each tool call is a ledger step, in order.
    assert [(s.kind, s.name) for s in out["ledger"]] == [
        ("model", "researcher"), ("tool", "search_chunks"), ("model", "researcher")]
    # I7: what the tool returned is shown as data.
    assert f'<passage id="{first}" paper="2608.11111v1"' in model.calls[1][2]


def test_a_tool_that_refuses_its_arguments_answers_with_the_error(store) -> None:
    model = Model("CALL: get_paper\nidentifier: ?!", "DONE: nothing to report")
    out = research(store, model)
    assert out["status"] == "done" and "error: " in model.calls[1][2]


def test_an_agent_stops_at_its_step_cap_and_at_its_budget(store) -> None:
    from dataclasses import replace

    capped = Model("CALL: search_chunks\nquery: BLEU")
    out = research(store, capped, replace(RESEARCHER, max_steps=1))
    assert out["status"] == "step cap" and len(capped.calls) == 1
    poor = Model("CALL: search_chunks\nquery: BLEU")
    out = research(store, poor, allowance=Allowance(calls=1))
    assert out["status"] == "budget" and len(poor.calls) == 1


def test_the_skeptic_finds_counter_evidence_with_search_alone(store) -> None:
    model = Model("CALL: get_paper\nidentifier: 2608.11111v1", "CALL: search_chunks\nquery: BLEU fails",
                  "DONE: nothing against it")
    out = research(store, model, SKEPTIC, brief="the model reaches 28.4 BLEU")
    assert "not one of your tools" in model.calls[1][2]  # refused, and sent back once
    assert all(f.stance == "adversarial" for e in out["board"] for f in e.found)


def test_a_researcher_cannot_reach_a_hidden_paper(store) -> None:
    model = Model("CALL: search_chunks\nquery: 28.4 BLEU English-German translation",
                  "CALL: read_pages\narxiv_id: 2608.11111v1\npages: 1", "DONE: nothing")
    out = research(store, model, hidden=[FIRST])
    assert all(e.arxiv_id != FIRST for e in out["board"])
    assert all("28.4 BLEU on the" not in user for _, _, user in model.calls)


# -- queued by the Lead -----------------------------------------------------------------


def test_the_lead_queues_research_and_reads_what_came_back(store) -> None:
    first = chunk_of(store, FIRST)
    model = Model("Machine translation results", "NEXT: researcher, writer\nBRIEF: find the BLEU score",
                  "CALL: search_chunks\nquery: BLEU", "DONE: 28.4 BLEU is reported",
                  cite(f"{GOOD} [1].", (1, first, GOOD)), "FINAL: 1")
    deps = Deps(scope=ScopedCorpus(store), embedder=Embedder(), model=model, search=SEARCH)
    outcome = run(QUESTION, deps)
    assert outcome.status == COMPLETE
    assert "find the BLEU score" in model.calls[2][2]  # the Lead's brief reached the researcher
    assert "researcher: 28.4 BLEU is reported" in model.calls[-1][2]  # and its note, the Lead
    assert "researcher (rank 1)" in model.calls[-1][2]  # what it found is on the board
    assert [s.name for s in outcome.ledger][:5] == ["describer", "lead", "researcher", "search_chunks",
                                                    "researcher"]


def test_research_that_cannot_reply_in_the_format_hands_back_to_the_lead(store, tmp_path: Path) -> None:
    model = Model("Machine translation results", "NEXT: researcher\nBRIEF: find the BLEU score",
                  "I will look around.", "Still looking.", "ABSTAIN: research failed")
    deps = Deps(scope=ScopedCorpus(store), embedder=Embedder(), model=model, search=SEARCH)
    outcome = run(QUESTION, deps)
    assert outcome.status == "abstained"
    assert "researcher: its replies could not be used, twice" in model.calls[-1][2]


def test_research_that_meets_the_runs_limit_ends_the_run(store) -> None:
    from verbatim.agents.contract import BUDGET_EXHAUSTED
    from verbatim.contracts import Budget

    model = Model("Machine translation results", "NEXT: researcher\nBRIEF: find the BLEU score",
                  "CALL: search_chunks\nquery: BLEU")
    deps = Deps(scope=ScopedCorpus(store), embedder=Embedder(), model=model, search=SEARCH,
                budget=Budget(max_calls=3))
    outcome = run(QUESTION, deps)
    assert outcome.status == BUDGET_EXHAUSTED and outcome.returned is None
    assert [s.name for s in outcome.ledger] == ["describer", "lead", "researcher", "search_chunks"]
