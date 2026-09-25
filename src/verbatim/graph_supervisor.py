"""The same ladder, expressed as a LangGraph state machine.

A second supervisor, not a replacement. The roles this project already
defines as Protocols - retriever, writer, verifier - become nodes here, and
the escalation that `work_section` writes as a `for` loop becomes a
conditional edge back to the retriever. Everything inside the nodes is the
same code the built-in supervisor calls: the same `retrieve`, the same
verified-answer path, the same deterministic `verdict_for`, the same
`choose_best`. Only the control flow differs.

That is the whole point of it. Two implementations behind one set of
contracts is the only thing that shows the contracts are a real boundary
rather than decoration, and running both over the same held-out questions
says whether the boundary holds. If the two disagree, the abstraction leaks
somewhere and the ledger will say where.

LangGraph is an optional dependency (`uv sync --extra langgraph`). The
default install stays at seven direct dependencies, because a local-first
project should not oblige anyone to pull a framework tree to answer a
question. Importing this module without it raises with that instruction
rather than an opaque ImportError.
"""

from __future__ import annotations

from collections.abc import Callable
import sqlite3
from typing import Any, TypedDict

from verbatim import config
from verbatim.answers import (
    Answer,
    Completion,
    subject_check,
)
from verbatim.agent import ToolCaller
from verbatim.contracts import BudgetGuard, Cost
from verbatim.corpus import CorpusStore
from verbatim.critique import critique as run_critique
from verbatim.embedding import Embedder
from verbatim.evidence import describe_evidence
from verbatim.orchestrator import (
    ACCEPTED,
    LADDERS,
    RETRY,
    Attempt,
    SectionLedger,
    answered_from_records,
    choose_best,
    question_shape,
    retrieve,
    verdict_for,
    write as write_answer,
)
from verbatim.report import ReportSection

MISSING = (
    "--supervisor langgraph needs the optional langgraph extra. "
    "Install it with: uv sync --extra langgraph"
)


class LadderState(TypedDict, total=False):
    """What flows between nodes.

    Deliberately only data. The corpus, the models and the tool caller are
    closed over when the graph is built: a SQLite connection is not state, and
    putting one here would be the first thing to break under a checkpointer.
    """

    heading: str
    shape: str
    ladder: tuple[str, ...]
    rung: int
    gathered: Any
    answer: Any
    subject: str | None
    closeness: float | None
    error: str
    # Set when the budget refuses a rung: the built-in loop stops there
    # rather than trying every remaining rung only to be refused again.
    stopped: bool
    spent_before: Any
    candidates: list[tuple[Attempt, ReportSection]]


def build_graph(
    corpus: CorpusStore,
    embedder: Embedder,
    completion: Completion,
    *,
    caller: ToolCaller | None = None,
    per_section: int = 6,
    max_per_paper: int | None = 2,
    check_subject: bool = True,
    warn_below: float = config.SUBJECT_WARN_BELOW,
    guard: BudgetGuard | None = None,
    on_attempt: Callable[[str, int, int], None] | None = None,
    complete_text: Callable[[str, str], str] | None = None,
    critic: bool = False,
) -> Any:
    """Compile the ladder into a graph: route, retrieve, write, verify, repeat.

    Takes every option `work_section` takes. orchestrate passes the critic's
    two to whichever supervisor it runs, and when this accepted fewer,
    `report --agents --supervisor langgraph` failed before its first section.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as error:  # pragma: no cover - exercised by its own test
        raise RuntimeError(MISSING) from error

    usage = getattr(completion, "usage", None)

    def route(state: LadderState) -> dict[str, Any]:
        shape = question_shape(state["heading"])
        return {
            "shape": shape,
            "ladder": state.get("ladder") or LADDERS[shape],
            "rung": 0,
            "candidates": [],
        }

    def gather(state: LadderState) -> dict[str, Any]:
        strategy = state["ladder"][state["rung"]]
        if on_attempt is not None:
            on_attempt(strategy, state["rung"] + 1, len(state["ladder"]))
        if guard is not None:
            stop = guard.exhausted()
            if stop is not None:
                return {"error": f"budget: {stop}", "stopped": True}
            guard.take()
        before = usage.snapshot() if usage is not None else None
        try:
            found = retrieve(
                strategy, corpus, embedder, state["heading"],
                limit=per_section, max_per_paper=max_per_paper, caller=caller,
            )
        except (RuntimeError, OSError, sqlite3.Error) as failure:
            return {"error": str(failure)[:200], "spent_before": before}
        return {"gathered": found, "error": "", "spent_before": before}

    def write(state: LadderState) -> dict[str, Any]:
        if state.get("error"):
            return {}
        try:
            return {
                "answer": write_answer(
                    corpus, completion, state["heading"], state["gathered"],
                    state["shape"],
                )
            }
        except (RuntimeError, OSError, sqlite3.Error) as failure:
            return {"error": str(failure)[:200]}

    def failed(heading: str, strategy: str, candidates: list, error: str) -> dict[str, Any]:
        candidates.append((
            Attempt(strategy=strategy, passages=0, verified=0, rejected=0,
                    closeness=None, verdict=RETRY, error=error),
            ReportSection(
                heading=heading,
                answer=Answer(
                    question=heading,
                    text=f"This section could not be written: {error}",
                    citations=[], rejected=[], declined=True,
                ),
            ),
        ))
        return {"candidates": candidates}

    def verify(state: LadderState) -> dict[str, Any]:
        """Record this rung as a candidate, whatever happened to it."""
        strategy = state["ladder"][state["rung"]]
        candidates = list(state.get("candidates") or [])
        if state.get("error"):
            return failed(state["heading"], strategy, candidates, state["error"])

        answer = state["answer"]
        matches = list(state["gathered"].matches)
        subject = closeness = None
        try:
            if matches and check_subject:
                subject, closeness = subject_check(
                    completion, embedder, state["heading"], matches
                )
            found = (
                run_critique(
                    state["heading"], answer, state["gathered"],
                    complete_text=complete_text,
                )
                if critic else None
            )
        except (RuntimeError, OSError, sqlite3.Error) as failure:
            # The checks call the model too. The built-in loop contains a
            # failure here as a failed rung; this node raised, and one section
            # losing its model took the whole review down with it.
            return failed(state["heading"], strategy, candidates, str(failure)[:200])
        verdict = verdict_for(
            answer, closeness, warn_below,
            listed=answered_from_records(state["gathered"], state["shape"]),
        )
        # A critique escalates only where a different retrieval could help,
        # exactly as in the built-in supervisor.
        accepted = verdict.accepted and not (found.unaddressed if found else ())
        spent = Cost()
        if usage is not None and state.get("spent_before") is not None:
            delta = usage.since(state["spent_before"])
            spent = Cost(delta.calls, delta.prompt_tokens, delta.output_tokens,
                         delta.seconds)
        evidence = describe_evidence(
            corpus, state["heading"], matches, subject=subject, closeness=closeness
        )
        candidates.append((
            Attempt(
                strategy=strategy, passages=len(matches),
                verified=len(answer.citations), rejected=len(answer.rejected),
                closeness=closeness,
                verdict=ACCEPTED if accepted else RETRY,
                route=state["gathered"].route,
                papers=state["gathered"].paper_ids,
                chunks=state["gathered"].chunk_ids, cost=spent, critique=found,
            ),
            ReportSection(
                heading=state["heading"], answer=answer,
                subject=subject, subject_closeness=closeness,
                evidence_lines=tuple(evidence.lines()),
            ),
        ))
        return {"candidates": candidates, "subject": subject, "closeness": closeness}

    def escalate(state: LadderState) -> dict[str, Any]:
        return {"rung": state["rung"] + 1, "error": "", "gathered": None,
                "answer": None}

    def next_step(state: LadderState) -> str:
        """The `for` loop's condition, as an edge.

        Accept on a verified, on-subject answer; otherwise climb, unless the
        ladder has run out. A rung that errored is not accepted, so a dead
        model server escalates to the next strategy exactly as it does in the
        built-in supervisor.
        """
        last, _section = state["candidates"][-1]
        if last.verdict == ACCEPTED or state.get("stopped"):
            return "done"
        if state["rung"] + 1 >= len(state["ladder"]):
            return "done"
        return "escalate"

    graph = StateGraph(LadderState)
    graph.add_node("route", route)
    graph.add_node("retrieve", gather)
    graph.add_node("write", write)
    graph.add_node("verify", verify)
    graph.add_node("escalate", escalate)
    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "write")
    graph.add_edge("write", "verify")
    graph.add_conditional_edges(
        "verify", next_step, {"escalate": "escalate", "done": END}
    )
    graph.add_edge("escalate", "retrieve")
    return graph.compile()


def work_section_with_graph(
    corpus: CorpusStore,
    embedder: Embedder,
    completion: Completion,
    heading: str,
    *,
    ladder: tuple[str, ...] | None = None,
    **options: Any,
) -> tuple[ReportSection, SectionLedger]:
    """Climb the ladder through the graph, returning what `work_section` does.

    Same signature and same return type as the built-in supervisor, so callers
    and measurements cannot tell which one ran except by asking.
    """
    compiled = build_graph(corpus, embedder, completion, **options)
    state: LadderState = {"heading": heading}
    if ladder is not None:
        state["ladder"] = ladder
    # The recursion limit has to clear one pass per rung: route, then four
    # nodes per attempt. LangGraph counts supersteps, not nodes, so this is
    # generous rather than exact - an exhausted ladder ends on its own.
    final = compiled.invoke(state, {"recursion_limit": 64})
    return choose_best(final["candidates"], heading, final["shape"])
