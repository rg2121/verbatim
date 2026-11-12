"""The run's state: frozen records, and the reducers that make parallel work exact.

Every list a branch adds to is merged by key, never by arrival, so the same
inputs and replies give the same state whatever order branches finish in (I9),
and a write applied twice changes nothing. The state keeps no counters: counts
come from the ledger, so a resumed or forked run cannot add them up twice.

A checkpoint gives back plain values (strings, numbers, None, lists, dicts) and
the records in RECORDS, and nothing else as itself: a tuple outside a record
comes back as a list, and any other dataclass as a dict of its fields.
`runtime.require_checkpointable` refuses state that holds either.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Any

from verbatim.contracts import Cost


def _tuples(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_tuples(item) for item in value)
    return value


class Record:
    """For frozen dataclasses kept in state. A checkpoint stores every tuple as a
    list, so lists are turned back into tuples here, and a record comes back from
    a checkpoint equal to what was saved, and hashable."""

    def __post_init__(self) -> None:
        for item in fields(self):  # type: ignore[arg-type]
            value = getattr(self, item.name)
            if isinstance(value, list):
                object.__setattr__(self, item.name, _tuples(value))


@dataclass(frozen=True)
class Step(Record):
    """One model or tool call, with what it cost (I5).

    `key` is where the call was made (`runtime.position`), then its number within
    that node: unique in a run, and the order the ledger is kept in, which is the
    order the run went.
    """

    key: tuple[int | str, ...]
    kind: str
    name: str
    cost: Cost = Cost()


@dataclass(frozen=True, order=True)
class Found(Record):
    """One way a passage was found: in which round, for which task, by whom."""

    round: int
    task: str
    query: str
    rank: int
    agent: str
    stance: str = "direct"


@dataclass(frozen=True)
class Evidence(Record):
    """A passage on the board, under its chunk id, with every way it was found."""

    chunk_id: str
    arxiv_id: str
    found: tuple[Found, ...] = ()

    @property
    def key(self) -> str:
        return self.chunk_id

    @property
    def direct(self) -> bool:
        return any(way.stance == "direct" for way in self.found)


@dataclass(frozen=True)
class SectionResult(Record):
    """What one round of a section returns. Kept under (plan index, round), so a
    section sent back keeps every attempt, and which one a report shows is decided
    after the join (I10), never by which arrived last."""

    index: int
    heading: str
    status: str
    text: str = ""
    round: int = 0

    @property
    def key(self) -> tuple[int, int]:
        return (self.index, self.round)


@dataclass(frozen=True)
class Allowance(Record):
    """A branch's share of what the run has left. None is unbounded."""

    calls: int | None = None
    tokens: int | None = None


@dataclass(frozen=True)
class Tally(Record):
    """What the check found in one attempt: the counts I10 compares, and the
    passages the attempt has a verified citation for (contract.tally)."""

    verified: int = 0
    failed: int = 0
    unbacked: int = 0
    uncited: int = 0
    passages: tuple[str, ...] = ()


@dataclass(frozen=True)
class Attempt(Record):
    """One draft or revision, numbered in the order written; `revises` is the
    attempt it repairs, if any."""

    number: int
    tally: Tally
    revises: int | None = None


# -- the Lead's decisions; contract.lead_action_refused says when each is legal --


@dataclass(frozen=True)
class Next(Record):
    """Take these steps in order. `brief` says what a research step must find;
    `passages` are a writer step's passages, v1's (the opening search's, in rank
    order) when empty."""

    steps: tuple[str, ...]
    brief: str = ""
    passages: tuple[str, ...] = ()


@dataclass(frozen=True)
class Inspect(Record):
    """Read these passages' exact text before deciding."""

    ids: tuple[str, ...]


@dataclass(frozen=True)
class Final(Record):
    """Finish, returning this attempt."""

    attempt: int


@dataclass(frozen=True)
class Abstain(Record):
    """Finish without an answer: the corpus does not answer the question."""

    reason: str


# Every record a checkpoint may hold; the serializer rebuilds these and no others.
RECORDS: tuple[type, ...] = (Step, Found, Evidence, SectionResult, Allowance, Tally, Attempt,
                             Next, Inspect, Final, Abstain, Cost)


def keyed_append(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """One record per key, in key order; a later write for a key replaces it."""
    merged = {item.key: item for item in left or ()}
    for item in right or ():
        merged[item.key] = item
    return [merged[key] for key in sorted(merged)]


def ledger_append(left: list[Step] | None, right: list[Step] | None) -> list[Step]:
    """keyed_append for the ledger, which refuses two different steps under one
    key: one of them would go uncounted (I5). Keys come from LangGraph's step
    numbers, so only a bug can make two; the refusal ends that run for good, since
    LangGraph saves a write before it merges it."""
    merged = {step.key: step for step in left or ()}
    for step in right or ():
        if merged.get(step.key, step) != step:
            raise ValueError(f"two different ledger steps under the key {step.key}")
        merged[step.key] = step
    return [merged[key] for key in sorted(merged)]


def merge_evidence(left: list[Evidence] | None, right: list[Evidence] | None) -> list[Evidence]:
    """The board: one entry per passage whoever found it, every way it was found
    kept once and sorted, entries in chunk-id order."""
    merged: dict[str, Evidence] = {}
    for evidence in (*(left or ()), *(right or ())):
        held = merged.get(evidence.chunk_id)
        found = set(evidence.found) | (set(held.found) if held else set())
        merged[evidence.chunk_id] = Evidence(
            evidence.chunk_id, held.arxiv_id if held else evidence.arxiv_id, tuple(sorted(found)))
    return [merged[key] for key in sorted(merged)]


def spent(steps: Iterable[Step]) -> Cost:
    """What the steps cost together: the only count a run keeps."""
    calls = tokens_in = tokens_out = 0
    seconds = 0.0
    for step in steps:
        calls += step.cost.calls
        tokens_in += step.cost.tokens_in
        tokens_out += step.cost.tokens_out
        seconds += step.cost.seconds
    return Cost(calls=calls, tokens_in=tokens_in, tokens_out=tokens_out, seconds=seconds)
