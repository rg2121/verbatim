"""What every v2 graph runs on: the checkpoint file, the run's limits, where a
call was made, and v1's events.

Calls and tokens are worked out from the ledger, not from the model clients'
counters: the ledger is checkpointed, so a resumed run, or one in another
process, sees what was already spent. Before a fan-out each branch gets an equal
share of what is left, less what is kept back for the steps after the join; what
a branch leaves unspent is there again at the join. Time is one deadline for the
whole run, which every branch checks before each call; a resumed run gets only
what its budget has left after the time it already used.

One thread holds one run: a new run takes a new thread id, and an unfinished one
is resumed with no input.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
import dataclasses
from datetime import datetime
from pathlib import Path
import sqlite3
import time
import typing
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_config

from verbatim.agents.state import RECORDS, Allowance, Step, spent
from verbatim.contracts import Budget, Cost

BUSY_TIMEOUT = 30  # seconds a writer waits for another process instead of failing


@dataclasses.dataclass(frozen=True)
class Deadline:
    """One time limit for the whole run. None is no limit."""

    at: float | None = None
    clock: Callable[[], float] = time.time

    @classmethod
    def for_budget(cls, budget: Budget, used: float = 0.0,
                   clock: Callable[[], float] = time.time) -> Deadline:
        """The deadline of a run under `budget` that has already used `used`
        seconds: 0 for a new run, `time_used` for a resumed one."""
        if budget.max_seconds is None:
            return cls(None, clock)
        return cls(clock() + max(budget.max_seconds - used, 0.0), clock)

    def passed(self) -> bool:
        return self.at is not None and self.clock() >= self.at


def position(at: Sequence[int | str] = (), branch: int = -1) -> tuple[int | str, ...]:
    """Where the running node is in its run: the position handed down by the graph
    that runs it (`at`; empty at the top), then its own superstep, name and branch.

    LangGraph numbers supersteps the same whatever order branches finish in, and
    after a resume, so ledger keys made from this come out the same every run and
    sort in the order the run went.
    """
    metadata = get_config()["metadata"]
    return (*at, metadata["langgraph_step"], metadata["langgraph_node"], branch)


def shares(budget: Budget, ledger: Sequence[Step], branches: int, *,
           keep_calls: int = 0, keep_tokens: int = 0) -> tuple[Allowance, ...]:
    """Split what the budget has left after the ledger, less what is kept back for
    the steps after the join, into equal shares, the remainder going to the first
    branches; never below zero. No branches, no shares."""
    if branches < 1:
        return ()
    used = spent(ledger)

    def split(limit: int | None, taken: int) -> list[int | None]:
        if limit is None:
            return [None] * branches
        each, extra = divmod(max(limit - taken, 0), branches)
        return [each + (i < extra) for i in range(branches)]

    return tuple(Allowance(calls, tokens) for calls, tokens in zip(
        split(budget.max_calls, used.calls + keep_calls),
        split(budget.max_tokens, used.tokens + keep_tokens), strict=True))


def limit_reached(allowance: Allowance, steps: Sequence[Step], deadline: Deadline) -> str | None:
    """Why a branch must stop before its next call, or None to go on.

    Calls are checked before a call, since each is one. Tokens are checked once
    spent, since a call's size is known only after it, so a branch can go past
    its token share by one call.
    """
    if deadline.passed():
        return "the deadline passed"
    used = spent(steps)
    if allowance.calls is not None and used.calls + 1 > allowance.calls:
        return f"{used.calls} of {allowance.calls} calls spent"
    if allowance.tokens is not None and used.tokens >= allowance.tokens:
        return f"{used.tokens:,} of {allowance.tokens:,} tokens spent"
    return None


class OutOfBudget(Exception):
    """A model call the run's limits do not allow."""


class Metered:
    """A model client that checks a branch's limits before every call, counting
    the calls `before` it, and records every call, failed ones too, as a ledger
    step after it (I5). `at` is where the calls are made (`position()`)."""

    def __init__(self, client: Any, allowance: Allowance, deadline: Deadline, name: str,
                 at: Sequence[int | str], before: Sequence[Step] = ()) -> None:
        self.client, self.allowance, self.deadline, self.name = client, allowance, deadline, name
        self.at, self.before = tuple(at), list(before)
        self.steps: list[Step] = []

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> Any:
        return self._call("complete", system, user, schema)

    def revise(self, system: str, user: str, schema: dict[str, Any]) -> Any:
        return self._call("revise", system, user, schema)

    def complete_text(self, system: str, user: str) -> str:
        return self._call("complete_text", system, user)

    def judge(self, system: str, user: str) -> str:
        return self._call("judge", system, user)

    def _call(self, method: str, *arguments: Any) -> Any:
        if why := limit_reached(self.allowance, [*self.before, *self.steps], self.deadline):
            raise OutOfBudget(why)
        before = self.client.usage.snapshot()
        try:
            return getattr(self.client, method)(*arguments)
        finally:
            used = self.client.usage.since(before)
            self.steps.append(Step(
                key=(*self.at, len(self.steps)), kind="model", name=self.name,
                cost=Cost(calls=max(used.calls, 1), tokens_in=used.prompt_tokens,
                          tokens_out=used.output_tokens, seconds=used.seconds)))


def serializer(records: Iterable[type] = RECORDS) -> JsonPlusSerializer:
    """Rebuilds LangGraph's own types and the records named here; any other class
    in a checkpoint comes back as its fields, never as an object."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(record.__module__, record.__name__) for record in records])


def require_checkpointable(*schemas: Any, records: Iterable[type] = RECORDS) -> None:
    """Refuse state a checkpoint would not give back as itself, when a graph is
    built rather than at the first resume: a dataclass outside `records` would come
    back as a dict of its fields, and a tuple outside a record as a list. `schemas`
    are state TypedDicts and the briefs Send carries."""
    allowed, seen, problems = set(records), set(), []

    def walk(hint: Any, where: str, in_record: bool) -> None:
        origin = typing.get_origin(hint)
        if origin is not None:
            if origin is tuple and not in_record:
                problems.append(f"{where} is a tuple, which comes back as a list")
            for arg in typing.get_args(hint):
                walk(arg, where, in_record)
        elif isinstance(hint, type) and hint not in seen:
            seen.add(hint)
            if dataclasses.is_dataclass(hint):
                if hint not in allowed:
                    problems.append(f"{hint.__name__} is not in RECORDS, so it comes back as a dict")
                for name, value in typing.get_type_hints(hint).items():
                    walk(value, f"{hint.__name__}.{name}", True)
            elif typing.is_typeddict(hint):
                for name, value in typing.get_type_hints(hint, include_extras=True).items():
                    walk(value, f"{hint.__name__}.{name}", in_record)

    for schema in schemas:
        walk(schema, getattr(schema, "__name__", repr(schema)), False)
    if problems:
        raise TypeError("a checkpoint would not give this state back as itself: "
                        + "; ".join(problems))


@contextmanager
def checkpoints(path: Path, records: Iterable[type] = RECORDS) -> Iterator[SqliteSaver]:
    """The checkpoint file every run, job and process shares, with its tables made.
    WAL lets readers work beside a writer, and a writer waits BUSY_TIMEOUT for
    another instead of failing with "database is locked"."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False, timeout=BUSY_TIMEOUT)
    try:
        _use_wal(connection)
        saver = SqliteSaver(connection, serde=serializer(records))
        saver.setup()
        yield saver
    finally:
        connection.close()


def _use_wal(connection: sqlite3.Connection) -> None:
    """Switching a new file to WAL needs the only lock on it, and SQLite fails at
    once, without the busy timeout, while another process is switching it too."""
    give_up = time.monotonic() + BUSY_TIMEOUT
    while True:
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as error:
            if "database is locked" not in str(error) or time.monotonic() >= give_up:
                raise
            time.sleep(0.01)


def refuse_used_thread(checkpointer: Any, config: dict[str, Any]) -> None:
    """A new run needs a new thread. One that already holds a run would carry its
    ledger, board and sections into the new run, and its spend into the new
    budget; an unfinished run is resumed instead, with no input."""
    if checkpointer is not None and checkpointer.get_tuple(config) is not None:
        raise ValueError(f"thread {config['configurable']['thread_id']!r} already holds a run: "
                         "start the new run under a new thread id, or resume this one")


def time_used(checkpointer: Any, thread_id: str) -> float:
    """Wall time a run has used: from its first checkpoint to its latest, nested
    graphs' included. Time after the latest checkpoint, up to a kill, is lost."""
    stamps = [datetime.fromisoformat(saved.checkpoint["ts"])
              for saved in checkpointer.list({"configurable": {"thread_id": thread_id}})]
    return (max(stamps) - min(stamps)).total_seconds() if stamps else 0.0


def v1_events(graph: Any, inputs: Any, config: dict[str, Any],
              **options: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    """A run as v1's events: what the nodes emit, from any depth, in the order
    emitted, then `done`. A run that fails, or stops short of a terminal status
    (waiting at an interrupt, or with nothing left to run), ends with `failed`
    first, so a half-written run never looks finished. Stream items without an
    event name are not v1 events and are skipped."""
    final: dict[str, Any] = {}
    try:
        for namespace, mode, data in graph.stream(
                inputs, config, stream_mode=["custom", "values"], subgraphs=True, **options):
            if mode == "values":
                if not namespace:
                    final = data
            elif isinstance(data, dict) and "event" in data:
                yield data["event"], {key: value for key, value in data.items() if key != "event"}
    # Broad on purpose, as in v1's web page: anything else would end with `done`.
    except Exception as error:  # noqa: BLE001
        yield "failed", {"message": f"{type(error).__name__}: {error}"}
    else:
        if "__interrupt__" in final:
            yield "failed", {"message": "the run stopped to wait for input"}
        elif not final.get("status"):
            yield "failed", {"message": "the run stopped without a terminal state"}
    yield "done", {}
