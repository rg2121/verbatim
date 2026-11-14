"""ReAct agents: a specialist that calls one tool a turn until it is done.

build_agent(spec) compiles a subgraph of two nodes. `think` is a model call through
Metered whose reply is a CALL or a DONE line (protocol.parse_step), sent back once
if it cannot be used. `act` runs the call through the toolbox on the scope (I11)
and shows the result back marked as data (I7); a tool that refuses its arguments
answers with the error, as v1's tool agent does. Passages a tool returns go on
the board with how they were found, and every model and tool call is a ledger
step (I5). An agent stops when it is done, at its step cap, or when the run's
limits refuse its next call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from verbatim.agents import toolbox
from verbatim.agents.protocol import ask_twice, parse_step
from verbatim.agents.runtime import Metered, OutOfBudget, position, require_checkpointable
from verbatim.agents.state import (
    Allowance,
    Done,
    Evidence,
    Found,
    Step,
    ToolCall,
    Turn,
    keyed_append,
    ledger_append,
    merge_evidence,
)
from verbatim.contracts import Cost
from verbatim.tools import TOOL_SCHEMAS

DONE, STEP_CAP, BUDGET, PROTOCOL = "done", "step cap", "budget", "protocol failure"
SHOWN_PASSAGES, EXCERPT, RECORD_CHARACTERS = 8, 300, 1200


@dataclass(frozen=True)
class AgentSpec:
    """One ReAct agent: its name, the model role it calls, its prompt, the tools
    it may call, its step cap, and the stance of what it finds."""

    name: str
    role: str
    system: str
    tools: tuple[str, ...]
    max_steps: int = 4
    stance: str = "direct"


class AgentState(TypedDict, total=False):
    question: str
    brief: str
    at: list[int | str]  # the wrapper's position, which every key in here starts with
    allowance: Allowance
    turns: Annotated[list[Turn], keyed_append]
    pending: ToolCall | None
    board: Annotated[list[Evidence], merge_evidence]
    ledger: Annotated[list[Step], ledger_append]
    note: str
    status: str


def build_agent(spec: AgentSpec) -> Any:
    """The agent's subgraph; its context is the run's (graphs.Deps)."""
    schemas = {schema["name"]: schema for schema in TOOL_SCHEMAS if schema["name"] in spec.tools}

    def think(state: AgentState, runtime: Runtime[Any]) -> Command:
        deps = runtime.context
        turns = state.get("turns", [])
        if len(turns) >= spec.max_steps:
            return Command(goto=END, update={"status": STEP_CAP,
                                             "note": f"it stopped at its cap of {spec.max_steps} tool calls"})
        meter = Metered(deps.model.for_role(spec.role), state["allowance"], deps.deadline, spec.name,
                        position(state["at"]), state.get("ledger", []))
        try:
            action, _ = ask_twice(meter.complete_text, spec.system, view(spec, schemas, state),
                                  lambda reply: parse_step(reply, schemas))
        except OutOfBudget as stop:
            return Command(goto=END, update={"ledger": meter.steps, "status": BUDGET, "note": str(stop)})
        update: dict[str, Any] = {"ledger": meter.steps}
        if action is None:
            return Command(goto=END, update={**update, "status": PROTOCOL})
        if isinstance(action, Done):
            return Command(goto=END, update={**update, "status": DONE, "note": action.note})
        return Command(goto="act", update={**update, "pending": action})

    def act(state: AgentState, runtime: Runtime[Any]) -> Command:
        deps = runtime.context
        call = state["pending"]
        assert call is not None
        arguments = dict(call.arguments)
        started = time.perf_counter()
        try:
            result = toolbox.call(call.tool, arguments, deps.scope, deps.embedder)
        except (ValueError, TypeError) as error:
            result = {"error": str(error)}
        step = Step(key=(*position(state["at"]), 0), kind="tool", name=call.tool,
                    cost=Cost(seconds=time.perf_counter() - started))
        query = str(arguments.get("query") or f"{call.tool} {json.dumps(arguments, default=str)}")
        round_ = int(state["at"][0]) if state["at"] else 0
        found = [Evidence(row["chunk_id"], row["arxiv_id"], (Found(
            round=round_, task=state["brief"], query=query, rank=rank, agent=spec.name,
            stance=spec.stance),)) for rank, row in enumerate(passages_in(result), start=1)]
        turn = Turn(number=len(state.get("turns", [])) + 1, call=call, shown=render(result))
        return Command(goto="think", update={"turns": [turn], "board": found, "ledger": [step],
                                             "pending": None})

    require_checkpointable(AgentState)
    graph = StateGraph(AgentState)
    graph.add_node("think", think, destinations=("act", END))
    graph.add_node("act", act, destinations=("think",))
    graph.add_edge(START, "think")
    return graph.compile()


def passages_in(result: Any) -> list[Mapping[str, Any]]:
    """The passages a tool returned: search_chunks' rows, or read_pages' passages."""
    rows = result.get("passages", []) if isinstance(result, dict) else result
    return [row for row in rows or [] if isinstance(row, dict) and "chunk_id" in row]


def render(result: Any) -> str:
    """A tool's result as the agent reads it, paper text marked as data (I7):
    passages as short excerpts, catalogue records compactly, as v1's tool agent
    sends them, and an error as an error."""
    if isinstance(result, dict) and "error" in result:
        return f"error: {result['error']}"
    rows = passages_in(result)
    if rows:
        shown = [f'<passage id="{row["chunk_id"]}" paper="{row["arxiv_id"]}" '
                 f'page="{row.get("page", row.get("position"))}">{_excerpt(row["text"])}</passage>'
                 for row in rows[:SHOWN_PASSAGES]]
        more = f", the first {SHOWN_PASSAGES} shown" if len(rows) > SHOWN_PASSAGES else ""
        return f"{len(rows)} passages{more}:\n" + "\n".join(shown)
    if not result:
        return "nothing found"
    return f"<data>{json.dumps(result, default=str)[:RECORD_CHARACTERS]}</data>"


def view(spec: AgentSpec, schemas: Mapping[str, Mapping[str, Any]], state: AgentState) -> str:
    """What the agent reads each turn: its task, its tools, and what it has done."""
    shown = [f"QUESTION: {state['question']}", f"YOUR TASK: {state['brief']}", "", "YOUR TOOLS:"]
    for name, schema in schemas.items():
        parameters = ", ".join(schema.get("parameters", {}).get("properties", {}))
        shown.append(f"- {name}({parameters}): {schema['description']}")
    turns = state.get("turns", [])
    shown += ["", "WHAT YOU HAVE DONE:" if turns else "WHAT YOU HAVE DONE: nothing yet"]
    for turn in turns:
        arguments = "; ".join(f"{key}: {value}" for key, value in turn.call.arguments)
        shown += [f"{turn.number}. CALL: {turn.call.tool} ({arguments})", turn.shown]
    shown += ["", f"TOOL CALLS LEFT: {spec.max_steps - len(turns)}",
              "Reply with exactly one decision: CALL: <tool>, then one line per argument "
              "(name: value); or DONE: <what you found, for the lead>."]
    return "\n".join(shown)


def _excerpt(text: str) -> str:
    plain = " ".join(text.split())
    return plain if len(plain) <= EXCERPT else plain[:EXCERPT].rsplit(" ", 1)[0] + " ..."
