"""A report-shaped slice of v2's runtime with scripted agents, for test_graph_runtime.py.

An Editor plans sections and fans them out with Send to `section_team`, a wrapper
node that runs the compiled section graph (research, then write) on its own state
and returns one SectionResult. A review joins the branches and may send a section
back once; finalize joins the sections in plan order. No model or corpus is used.

Run as a script, it is the child process the kill and two-process tests start:

    python tests/graph_slice.py start|resume DB THREAD WORK [--sections N] [--block I]
        [--delay S]

WORK receives calls.log, one line per scripted call; a blocked section waits for
WORK/release before it drafts.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import threading
import time
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, Send

from verbatim.agents.runtime import (
    Deadline,
    checkpoints,
    limit_reached,
    position,
    refuse_used_thread,
    require_checkpointable,
    shares,
)
from verbatim.agents.state import (
    Allowance,
    Evidence,
    Found,
    SectionResult,
    Step,
    keyed_append,
    ledger_append,
    merge_evidence,
)
from verbatim.contracts import Budget, Cost

COMPLETE = "complete"
BUDGET_EXHAUSTED = "budget_exhausted"
SHARED_CHUNK = "c-shared"  # every section finds it, so the board must hold it once
REVIEW_CALLS = 1  # what each fan-out keeps back for the review after its join


class FakeClock:
    """Time that moves only when a scripted call says so, so deadlines are exact."""

    def __init__(self) -> None:
        self._now = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds


class ScriptedModel:
    """Stands in for every agent's model: each call is logged, may wait, and costs
    the same tokens, so a run's ledger is known in advance."""

    def __init__(self, *, tokens: tuple[int, int] = (10, 5),
                 delay: Callable[[tuple], float] | None = None,
                 clock: FakeClock | None = None, seconds_per_call: float = 0.0,
                 log: Path | None = None, fail_on: str | None = None) -> None:
        self.tokens = tokens
        self.delay = delay
        self.clock = clock
        self.seconds_per_call = seconds_per_call
        self.log = log
        self.fail_on = fail_on
        self.calls: list[tuple] = []
        self._lock = threading.Lock()

    def call(self, key: tuple[int | str, ...], name: str) -> Step:
        started = time.perf_counter()
        if self.delay is not None:
            time.sleep(self.delay(key))
        if name == self.fail_on:
            raise RuntimeError(f"scripted failure in {name}")
        with self._lock:
            self.calls.append(key)
            if self.log is not None:
                with self.log.open("a", encoding="utf-8") as out:
                    out.write(f"{os.getpid()} {name} {key[2]}\n")
            if self.clock is not None:
                self.clock.advance(self.seconds_per_call)
        tokens_in, tokens_out = self.tokens
        return Step(key=key, kind="model", name=name,
                    cost=Cost(calls=1, tokens_in=tokens_in, tokens_out=tokens_out,
                              seconds=time.perf_counter() - started))


@dataclass(frozen=True)
class Script:
    """What the scripted agents do: the plan, the order the Editor starts its
    sections in (plan order unless given), how many searches each section makes
    (2 unless listed), which sections the review sends back after the first round,
    and which section waits for a gate file before it drafts."""

    plan: tuple[str, ...] = ("Background", "Methods", "Results")
    order: tuple[int, ...] = ()
    searches: tuple[tuple[int, int], ...] = ()
    send_back: tuple[int, ...] = ()
    block: int | None = None
    gate: Path | None = None

    def searches_for(self, index: int) -> int:
        return dict(self.searches).get(index, 2)


@dataclass
class Deps:
    """The run's context. Never checkpointed: whoever resumes a run supplies it."""

    model: ScriptedModel
    script: Script = field(default_factory=Script)
    budget: Budget = field(default_factory=Budget)
    deadline: Deadline = field(default_factory=Deadline)


class RunState(TypedDict, total=False):
    topic: str
    plan: list[str]
    round: int
    board: Annotated[list[Evidence], merge_evidence]
    ledger: Annotated[list[Step], ledger_append]
    sections: Annotated[list[SectionResult], keyed_append]
    status: str
    result: str


class SectionBrief(TypedDict):
    """What the Editor sends a section: plain values and records, never a tuple."""

    index: int
    heading: str
    round: int
    allowance: Allowance


class SectionState(TypedDict, total=False):
    at: list[int | str]  # the wrapper's position, which every key in here starts with
    index: int
    heading: str
    round: int
    allowance: Allowance
    board: Annotated[list[Evidence], merge_evidence]
    ledger: Annotated[list[Step], ledger_append]
    draft: str
    status: str


# -- the section graph, run by the wrapper on its own state ---------------------------


def research(state: SectionState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    index, here = state["index"], position(state["at"])
    get_stream_writer()({"event": "working", "index": index, "heading": state["heading"]})
    steps: list[Step] = []
    found: list[Evidence] = []
    for seq in range(deps.script.searches_for(index)):
        if limit_reached(state["allowance"], steps, deps.deadline):
            return {"ledger": steps, "board": found, "status": BUDGET_EXHAUSTED}
        steps.append(deps.model.call((*here, seq), "researcher"))
        provenance = (Found(round=state["round"], task=state["heading"], query=f"q{seq}",
                            rank=seq, agent=f"researcher-{index}"),)
        found.append(Evidence(f"c{index}-{seq}", f"2611.{index:05d}v1", provenance))
        found.append(Evidence(SHARED_CHUNK, "2611.99999v1", provenance))
    return {"ledger": steps, "board": found}


def after_research(state: SectionState) -> str:
    return END if state.get("status") == BUDGET_EXHAUSTED else "write"


def write(state: SectionState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    script = deps.script
    if script.block == state["index"] and script.gate is not None and not script.gate.exists():
        if deps.model.log is not None:
            with deps.model.log.open("a", encoding="utf-8") as out:
                out.write(f"{os.getpid()} waiting {state['index']}\n")
        while not script.gate.exists():
            time.sleep(0.02)
    if limit_reached(state["allowance"], state["ledger"], deps.deadline):
        return {"status": BUDGET_EXHAUSTED}
    step = deps.model.call((*position(state["at"]), 0), "writer")
    passages = ", ".join(evidence.chunk_id for evidence in state["board"])
    return {"ledger": [step], "draft": f"{state['heading']}, round {state['round']}: {passages}",
            "status": COMPLETE}


def build_section() -> Any:
    graph = StateGraph(SectionState, context_schema=Deps)
    graph.add_node("research", research)
    graph.add_node("write", write)
    graph.add_edge(START, "research")
    graph.add_conditional_edges("research", after_research, ["write", END])
    graph.add_edge("write", END)
    return graph.compile()


SECTION = build_section()


# -- the run graph --------------------------------------------------------------------


def editor(state: RunState, runtime: Runtime[Deps]) -> Command:
    deps = runtime.context
    if deps.deadline.passed():
        return Command(goto="finalize", update={"status": BUDGET_EXHAUSTED})
    step = deps.model.call((*position(), 0), "editor")
    plan = list(deps.script.plan)
    allowances = shares(deps.budget, [*state.get("ledger", []), step], len(plan),
                        keep_calls=REVIEW_CALLS)
    order = deps.script.order or tuple(range(len(plan)))
    return Command(
        goto=[Send("section_team", SectionBrief(index=i, heading=plan[i], round=0,
                                                allowance=allowances[i])) for i in order],
        update={"plan": plan, "round": 0, "ledger": [step]},
    )


def section_team(brief: SectionBrief, runtime: Runtime[Deps]) -> dict[str, Any]:
    """Run one section's graph on its own state; return only keys with a reducer."""
    at = list(position(branch=brief["index"]))
    out = SECTION.invoke({**brief, "at": at, "board": [], "ledger": []})
    result = SectionResult(index=brief["index"], heading=brief["heading"], round=brief["round"],
                           status=out.get("status", COMPLETE), text=out.get("draft", ""))
    get_stream_writer()({"event": "section", "index": result.index, "heading": result.heading,
                         "status": result.status})
    return {"sections": [result], "ledger": out["ledger"], "board": out["board"]}


def review(state: RunState, runtime: Runtime[Deps]) -> Command:
    """The Editor at review time: one call, then send a section back or finish.
    With no budget left for that call the run ends budget_exhausted, never complete."""
    deps = runtime.context
    round_ = state["round"]
    if limit_reached(shares(deps.budget, state["ledger"], 1)[0], [], deps.deadline):
        return Command(goto="finalize", update={"status": BUDGET_EXHAUSTED})
    step = deps.model.call((*position(), 0), "review")
    back = [s for s in state["sections"] if round_ == 0 and s.index in deps.script.send_back]
    if not back:
        return Command(goto="finalize", update={"ledger": [step]})
    allowances = shares(deps.budget, [*state["ledger"], step], len(back), keep_calls=REVIEW_CALLS)
    return Command(
        goto=[Send("section_team", SectionBrief(index=s.index, heading=s.heading,
                                                round=round_ + 1, allowance=share))
              for s, share in zip(back, allowances, strict=True)],
        update={"ledger": [step], "round": round_ + 1},
    )


def shown(sections: list[SectionResult]) -> list[SectionResult]:
    """What the report shows of each section, in plan order: how its latest round
    ended, with the latest draft any round wrote, so a round cut short never blanks
    a finished section. M5 replaces this with I10's rank."""
    rounds: dict[int, list[SectionResult]] = {}
    for section in sections:  # in (index, round) order
        rounds.setdefault(section.index, []).append(section)
    out = []
    for attempts in (rounds[index] for index in sorted(rounds)):
        drafted = [attempt for attempt in attempts if attempt.text] or attempts
        out.append(replace(drafted[-1], status=attempts[-1].status))
    return out


def finalize(state: RunState) -> dict[str, Any]:
    sections = shown(state.get("sections", []))
    status = state.get("status") or (
        COMPLETE if sections and all(s.status == COMPLETE for s in sections) else BUDGET_EXHAUSTED)
    get_stream_writer()({"event": "report", "status": status, "sections": len(sections)})
    return {"status": status, "result": "\n\n".join(s.text for s in sections)}


def build(checkpointer: Any = None) -> Any:
    require_checkpointable(RunState, SectionState, SectionBrief)
    graph = StateGraph(RunState, context_schema=Deps)
    graph.add_node("editor", editor, destinations=("section_team", "finalize"))
    graph.add_node("section_team", section_team)
    graph.add_node("review", review, destinations=("section_team", "finalize"))
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "editor")
    graph.add_edge("section_team", "review")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


# -- child processes ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start", "resume"])
    parser.add_argument("db", type=Path)
    parser.add_argument("thread")
    parser.add_argument("work", type=Path)
    parser.add_argument("--sections", type=int, default=3)
    parser.add_argument("--block", type=int)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds each call takes")
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    script = Script(plan=tuple(f"Section {i}" for i in range(args.sections)), block=args.block,
                    gate=args.work / "release")
    config = {"configurable": {"thread_id": args.thread}}
    with checkpoints(args.db) as saver:
        if args.command == "start":
            try:
                refuse_used_thread(saver, config)
            except ValueError as error:
                raise SystemExit(f"error: {error}") from None
        build(saver).invoke({"topic": args.thread} if args.command == "start" else None, config,
                            context=Deps(model=ScriptedModel(
                                log=args.work / "calls.log",
                                delay=(lambda key: args.delay) if args.delay else None),
                                script=script),
                            durability="sync")
    return 0


def open_at_once(path: str, barrier: Any) -> None:
    """For the race test: open a new checkpoint file the moment every process is ready."""
    barrier.wait()
    with checkpoints(Path(path)):
        pass


if __name__ == "__main__":
    raise SystemExit(main())
