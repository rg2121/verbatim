"""The answer graph: v2's `ask`, in LangGraph.

    opening -> lead -> the steps the Lead queues (writer, verifier) -> ... -> finalize

- opening (code): v1's plain-ask search, through the scope, and v1's subject
  check, which the Lead reads as a fact.
- lead (agent): queues steps, inspects passages, returns an attempt or abstains.
  It is consulted again when its queue ends or a step reports a problem.
- writer (agent): drafts with v1's prompt, or repairs. Code checks every draft, and
  the verifier follows a draft the check flags, where v1 would have repaired.
- verifier (agent): decides the check's findings, and may send the draft back.
- finalize (code): the terminal state, and what is shown: the attempt the Lead
  returned, or by I10's rank when the run ended incomplete, decisions applied.

Every model call goes through Metered, which checks the run's limits before it,
the second call of a retry included, and records it in the ledger after (I5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from verbatim import tools
from verbatim.agents import toolbox
from verbatim.agents.attempts import (
    REPAIR_MODE,
    WRITE_MODE,
    assemble,
    check,
    decide,
    needs_verifier,
    repair_facts,
    verify,
    write,
)
from verbatim.agents.contract import (
    ABSTAINED,
    BUDGET_EXHAUSTED,
    COMPLETE,
    ERROR,
    PROTOCOL_FAILURE,
    LEAD_STEPS,
    first_by_rank,
    lead_action_refused,
    tally,
)
from verbatim.agents.lead import AgentPolicy, LeadPolicy, lead_view
from verbatim.agents.react import PROTOCOL, AgentSpec, build_agent
from verbatim.agents.roster import RESEARCHER, SKEPTIC
from verbatim.agents.runtime import (
    Deadline,
    Metered,
    OutOfBudget,
    position,
    refuse_used_thread,
    require_checkpointable,
    shares,
)
from verbatim.agents.scoped import ScopedCorpus
from verbatim.agents.state import (
    Abstain,
    CitationRecord,
    Draft,
    Evidence,
    Final,
    Found,
    Inspect,
    Next,
    Note,
    Step,
    Tally,
    WriterBrief,
    keyed_append,
    ledger_append,
    merge_evidence,
)
from verbatim.answers import Answer, subject_check
from verbatim.contracts import Budget
from verbatim.corpus import ChunkMatch
from verbatim.reflection import passage_texts

STEPS = LEAD_STEPS  # researcher, skeptic, writer, verifier
MAX_LEAD_TURNS = 12  # a step cap: past it the run ends budget_exhausted
RECURSION_LIMIT = 200  # LangGraph's own, set past anything MAX_LEAD_TURNS allows


@dataclass
class Deps:
    """A run's context, never checkpointed: whoever resumes a run supplies it.
    `model` serves every role (RoleCompletion): complete and revise write,
    judge describes, for_role names the Lead's and the Verifier's clients.
    `search` is the opening search's options, as v1's CLI sets them."""

    scope: ScopedCorpus
    embedder: Any
    model: Any
    search: dict[str, Any] = field(default_factory=dict)
    subject_check: bool = True
    budget: Budget = field(default_factory=Budget)
    deadline: Deadline = field(default_factory=Deadline)
    policy: LeadPolicy = field(default_factory=AgentPolicy)


class AskState(TypedDict, total=False):
    question: str
    opening: list[str]  # the opening search's passages, in rank order: the Writer's default
    board: Annotated[list[Evidence], merge_evidence]
    subject: str
    closeness: float
    drafts: Annotated[list[Draft], keyed_append]
    citations: Annotated[list[CitationRecord], keyed_append]
    ledger: Annotated[list[Step], ledger_append]
    notes: Annotated[list[Note], keyed_append]
    queue: list[str]  # the steps the Lead queued and not yet taken
    brief: str  # what the Lead's queued research steps must find
    passages: list[str]  # the Lead's PASSAGES for the next writer step
    repair: WriterBrief | None  # a repair the Verifier asked for, for the next writer step
    inspected: list[str]
    status: str
    reason: str
    returned: int | None
    result: Answer | None
    shown: Tally | None


def _meter(deps: Deps, client: Any, ledger: Sequence[Step], name: str) -> Metered:
    """A metered client for one sequential step: it may spend what the run has left."""
    return Metered(client, shares(deps.budget, ledger, 1)[0], deps.deadline, name, position())


def passages_for(scope: ScopedCorpus, ids: Sequence[str]) -> list[ChunkMatch]:
    """The passages a brief names, in its order, read through the scope."""
    found = {chunk.chunk_id: chunk for chunk in scope.chunks_by_id(list(ids))}
    missing = [chunk_id for chunk_id in ids if chunk_id not in found]
    if missing:
        raise ValueError(f"passage {missing[0]} cannot be read")
    return [ChunkMatch(chunk=found[chunk_id], score=0.0, snippet="") for chunk_id in ids]


def _finish(status: str, update: dict[str, Any], reason: str = "") -> Command:
    return Command(goto="finalize", update={**update, "status": status, "reason": reason})


def _next(queue: Sequence[str]) -> str:
    return queue[0] if queue else "lead"


# -- nodes ------------------------------------------------------------------------------


def opening(state: AskState, runtime: Runtime[Deps]) -> Command:
    deps = runtime.context
    question = state["question"]
    matches = deps.scope.search(deps.embedder, question, **deps.search)
    update: dict[str, Any] = {
        "opening": [match.chunk.chunk_id for match in matches],
        "board": [Evidence(match.chunk.chunk_id, match.chunk.arxiv_id, (Found(
            round=0, task="question", query=question, rank=rank, agent="opening"),))
            for rank, match in enumerate(matches, start=1)],
        "queue": [], "passages": [], "inspected": [], "repair": None}
    get_stream_writer()({"event": "searched", "passages": len(matches)})
    if matches and deps.subject_check:  # as v1, which checks no subject with no passages
        meter = _meter(deps, deps.model, state.get("ledger", []), "describer")
        try:
            subject, closeness = subject_check(meter, deps.embedder, question, matches)
        except OutOfBudget as stop:
            return _finish(BUDGET_EXHAUSTED, {**update, "ledger": meter.steps}, str(stop))
        update["ledger"] = meter.steps
        if subject:
            update.update(subject=subject, closeness=closeness)
    return Command(goto="lead", update=update)


def lead(state: AskState, runtime: Runtime[Deps]) -> Command:
    deps = runtime.context
    ledger = state.get("ledger", [])
    turns = {step.key[:-1] for step in ledger if step.name == "lead"}
    if len(turns) >= MAX_LEAD_TURNS:
        return _finish(BUDGET_EXHAUSTED, {}, f"the Lead took its {MAX_LEAD_TURNS} turns")
    drafts = state.get("drafts", [])
    board = state.get("board", [])
    meter = _meter(deps, deps.model.for_role("lead"), ledger, "lead")
    view = lead_view(
        state["question"], subject=state.get("subject"), closeness=state.get("closeness"),
        board=_board_rows(deps.scope, board), drafts=drafts, records=state.get("citations", []),
        inspected=toolbox.inspect(deps.scope, state.get("inspected", [])),
        notes=state.get("notes", []), limits=_limits(meter, deps.deadline), steps=STEPS)
    ids = {evidence.chunk_id for evidence in board}
    attempts = [draft.attempt for draft in drafts]
    try:
        action = deps.policy.decide(view, lambda action: lead_action_refused(
            action, board=ids, attempts=attempts, steps=STEPS), meter.complete_text)
    except OutOfBudget as stop:
        return _finish(BUDGET_EXHAUSTED, {"ledger": meter.steps}, str(stop))
    update: dict[str, Any] = {"ledger": meter.steps}
    get_stream_writer()({"event": "decided", "action": type(action).__name__ if action else None})
    if action is None:
        return _finish(PROTOCOL_FAILURE, update, "the Lead's replies could not be used, twice")
    if isinstance(action, Next):
        return Command(goto=action.steps[0], update={
            **update, "queue": list(action.steps), "brief": action.brief,
            "passages": list(action.passages)})
    if isinstance(action, Inspect):
        return Command(goto="lead", update={**update, "inspected": list(action.ids)})
    if isinstance(action, Final):
        return _finish(COMPLETE, {**update, "returned": action.attempt})
    assert isinstance(action, Abstain)
    return _finish(ABSTAINED, update, action.reason)


def writer(state: AskState, runtime: Runtime[Deps]) -> Command:
    deps = runtime.context
    queue = list(state.get("queue", []))[1:]
    drafts = state.get("drafts", [])
    if deps.budget.max_attempts is not None and len(drafts) >= deps.budget.max_attempts:
        return _finish(BUDGET_EXHAUSTED, {}, f"{len(drafts)} attempts reached the limit")
    brief = state.get("repair") or WriterBrief(
        task=state["question"], passage_ids=tuple(state.get("passages") or state.get("opening", [])))
    matches = passages_for(deps.scope, brief.passage_ids)
    meter = _meter(deps, deps.model, state.get("ledger", []), "writer")
    repaired = next((draft for draft in drafts if draft.number == brief.draft), None)
    try:
        payload = write(meter, brief, matches, repaired)
    except OutOfBudget as stop:
        return _finish(BUDGET_EXHAUSTED, {"ledger": meter.steps}, str(stop))
    update: dict[str, Any] = {"ledger": meter.steps, "queue": queue, "repair": None, "passages": []}
    if payload is None:  # a problem: back to the Lead, whatever was queued
        note = Note(at=(*meter.at, 0), agent="writer",
                    text=f"its {brief.mode} could not be read" + (", twice" if brief.mode == WRITE_MODE else ""))
        return Command(goto="lead", update={**update, "queue": [], "notes": [note]})
    number = max((draft.number for draft in drafts), default=0) + 1
    papers = deps.scope.papers_by_id(sorted({match.chunk.arxiv_id for match in matches}))
    checked = check(number, brief, payload, matches, papers)
    update.update(drafts=[checked.draft], citations=list(checked.records))
    get_stream_writer()({"event": "drafted", "attempt": number})
    if brief.mode == WRITE_MODE and needs_verifier(checked.draft) and queue[:1] != ["verifier"]:
        queue = ["verifier", *queue]
        update["queue"] = queue
    return Command(goto=_next(queue), update=update)


def verifier(state: AskState, runtime: Runtime[Deps]) -> Command:
    deps = runtime.context
    queue = list(state.get("queue", []))[1:]
    latest = max(state["drafts"], key=lambda draft: draft.number)
    records = [record for record in state.get("citations", []) if record.attempt == latest.number]
    matches = passages_for(deps.scope, latest.brief.passage_ids)
    meter = _meter(deps, deps.model.for_role("verifier"), state.get("ledger", []), "verifier")
    try:
        verification, _ = verify(meter.complete_text, latest, records, passage_texts(matches))
    except OutOfBudget as stop:
        return _finish(BUDGET_EXHAUSTED, {"ledger": meter.steps}, str(stop))
    update: dict[str, Any] = {"ledger": meter.steps, "queue": queue}
    if verification is None:
        note = Note(at=(*meter.at, 0), agent="verifier", text="its replies could not be used, twice")
        return Command(goto="lead", update={**update, "queue": [], "notes": [note]})
    decided = decide(records, verification)
    update["citations"] = list(decided)
    if verification.research:  # a problem: the Lead hears of it once any repair is written
        update["notes"] = [Note(at=(*meter.at, 0), agent="verifier",
                                text=f"on attempt {latest.number}, asks for research: {verification.research}")]
        queue = []
    if verification.repair:
        update["repair"] = WriterBrief(task=latest.brief.task, passage_ids=latest.brief.passage_ids,
                                       mode=REPAIR_MODE, repair_facts=repair_facts(latest, decided),
                                       draft=latest.number)
        queue = ["writer", *queue]
    update["queue"] = queue
    return Command(goto=_next(queue), update=update)


def researching(spec: AgentSpec) -> Any:
    """The node that runs one ReAct agent's subgraph on its own state, with the
    Lead's brief: what it finds goes on the board and its last word to the Lead
    as a note. An agent that met its limits says so in that note: whether the
    run can go on is the next metered call's to say. One whose replies could not
    be used hands the run back to the Lead."""
    agent = build_agent(spec)

    def node(state: AskState, runtime: Runtime[Deps]) -> Command:
        deps = runtime.context
        queue = list(state.get("queue", []))[1:]
        at = list(position())
        out = agent.invoke({"question": state["question"], "brief": state.get("brief", ""), "at": at,
                            "allowance": shares(deps.budget, state.get("ledger", []), 1)[0],
                            "turns": [], "board": [], "ledger": []})
        update: dict[str, Any] = {"board": out.get("board", []), "ledger": out.get("ledger", []),
                                  "queue": queue}
        if out["status"] == PROTOCOL:
            update.update(queue=[], notes=[Note(at=(*at, 0), agent=spec.name,
                                                 text="its replies could not be used, twice")])
            return Command(goto="lead", update=update)
        update["notes"] = [Note(at=(*at, 0), agent=spec.name, text=out.get("note", ""))]
        get_stream_writer()({"event": "researched", "agent": spec.name,
                             "passages": len(out.get("board", []))})
        return Command(goto=_next(queue), update=update)

    return node


def finalize(state: AskState, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    status = state["status"]
    drafts = {draft.number: draft for draft in state.get("drafts", [])}
    if status == COMPLETE:
        returned = state["returned"]
    elif status in (BUDGET_EXHAUSTED, PROTOCOL_FAILURE):
        returned = first_by_rank([draft.attempt for draft in drafts.values()])
    else:
        returned = None
    update: dict[str, Any] = {"returned": returned, "result": None, "shown": None}
    if returned is not None:
        draft = drafts[returned]
        records = [record for record in state.get("citations", []) if record.attempt == returned]
        matches = passages_for(deps.scope, draft.brief.passage_ids)
        papers = deps.scope.papers_by_id(sorted({match.chunk.arxiv_id for match in matches}))
        answer = assemble(draft, records, matches, papers)
        update.update(result=answer, shown=tally(answer, passage_texts(matches)))
    get_stream_writer()({"event": "finished", "status": status, "attempt": returned})
    return update


def _board_rows(scope: ScopedCorpus, board: Sequence[Evidence]) -> list[dict[str, Any]]:
    found = {evidence.chunk_id: evidence for evidence in board}
    rows = tools.passage_rows(scope, scope.chunks_by_id(list(found)))  # type: ignore[arg-type]
    for row in rows:
        ways = sorted(found[row["chunk_id"]].found)
        row["found"] = ", ".join(f"{way.agent} (rank {way.rank})" for way in ways)
        row["first"] = min((way.round, way.rank) for way in ways)
    return sorted(rows, key=lambda row: (row["first"], row["chunk_id"]))  # as found, best first


def _limits(meter: Metered, deadline: Deadline) -> str:
    allowance = meter.allowance
    left = []
    if allowance.calls is not None:
        left.append(f"{allowance.calls} model calls")
    if allowance.tokens is not None:
        left.append(f"{allowance.tokens:,} tokens")
    if deadline.at is not None:
        left.append(f"{max(deadline.at - deadline.clock(), 0):.0f} seconds")
    return ", ".join(left) + " left" if left else "none"


# -- the graph and its one entry point --------------------------------------------------


def build_ask(checkpointer: Any = None) -> Any:
    require_checkpointable(AskState)
    graph = StateGraph(AskState, context_schema=Deps)
    steps = ("lead", *STEPS, "finalize")
    graph.add_node("opening", opening, destinations=("lead", "finalize"))
    graph.add_node("lead", lead, destinations=steps)
    graph.add_node("researcher", researching(RESEARCHER), destinations=steps)
    graph.add_node("skeptic", researching(SKEPTIC), destinations=steps)
    graph.add_node("writer", writer, destinations=steps)
    graph.add_node("verifier", verifier, destinations=steps)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "opening")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


@dataclass(frozen=True)
class Outcome:
    """How a run ended (I2): its terminal state, what is shown, and why."""

    status: str
    answer: Answer | None
    returned: int | None
    reason: str
    tally: Tally | None
    ledger: tuple[Step, ...]
    state: dict[str, Any]


def run(question: str, deps: Deps, *, saver: Any = None, thread_id: str = "ask") -> Outcome:
    """Ask a question through the answer graph. A run with a checkpointer takes a
    thread of its own; any failure of the machinery ends it in `error`."""
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id},
                              "recursion_limit": RECURSION_LIMIT}
    options: dict[str, Any] = {}
    if saver is not None:  # LangGraph 1.2.11 fails on `durability` without a checkpointer
        refuse_used_thread(saver, config)
        options["durability"] = "sync"
    try:
        state = build_ask(saver).invoke({"question": question}, config, context=deps, **options)
    except Exception as error:  # noqa: BLE001 - every failure ends in a terminal state
        return Outcome(ERROR, None, None, f"{type(error).__name__}: {error}", None, (), {})
    return Outcome(state["status"], state.get("result"), state.get("returned"),
                   state.get("reason", ""), state.get("shown"), tuple(state.get("ledger", ())), state)
