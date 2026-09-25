"""What each role in the orchestrator promises, stated as types.

The roles were functions that happened to fit together. That is enough to run
and not enough to reason about: nothing said what a retriever must return, so
nothing could check that a new one behaved like the old, and "bounded roles
with explicit contracts" was a description of the intent rather than of the
code.

Each role below is a Protocol over frozen dataclasses. Two things follow. A
role can be replaced - a different retriever, a critic that is a model instead
of arithmetic - without its caller changing, because the caller depends on the
contract rather than on the implementation. And a role can be tested against
its contract alone, with no orchestrator present, which is what makes them
independently testable rather than only jointly.

Nothing here validates at runtime. These are static contracts and a cheap
assertion helper for tests; a schema validator on every call would cost more
than it catches in a single process where the only callers are in this
repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from verbatim.answers import Answer
from verbatim.corpus import ChunkMatch


@dataclass(frozen=True)
class Plan:
    """What a review will cover, one question per section."""

    topic: str
    sections: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.sections:
            raise ValueError("a plan with no sections is not a plan")


@dataclass(frozen=True)
class Gathered:
    """Passages for one question, and how they were found.

    `strategy` names the retrieval that produced them and `route` records what
    a tool-choosing rung actually called, so the ledger can say what happened
    rather than only that something did.

    `records` are catalogue rows a tool returned instead of passages - what a
    recency question is answered from. They carry no quotable text, so they
    are kept apart from `matches` rather than dropped: dropping them is how a
    question about what is new came to be answered by passage search.
    """

    question: str
    matches: tuple[ChunkMatch, ...]
    strategy: str
    route: str = ""
    records: tuple[dict[str, Any], ...] = ()

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(match.chunk.chunk_id for match in self.matches)

    @property
    def paper_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(match.chunk.arxiv_id for match in self.matches))


@dataclass(frozen=True)
class Cost:
    """What one step spent. A run outlives a step, so this is a difference."""

    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def render(self) -> str:
        return f"{self.calls} calls, {self.tokens:,} tokens, {self.seconds:.1f}s"


@dataclass(frozen=True)
class Budget:
    """A ceiling for a whole run, not a step count per role.

    None means unbounded, which is the default: a budget that cannot be
    exceeded on a laptop is a ceremony, and one that stops a review halfway
    needs to be asked for.
    """

    max_tokens: int | None = None
    max_seconds: float | None = None
    max_attempts: int | None = None


@dataclass(frozen=True)
class Verdict:
    """Whether a written section may stand, decided from numbers."""

    accepted: bool
    reason: str


@dataclass(frozen=True)
class Critique:
    """What is missing or unsupported in a written section.

    Every field is either counted or checked against the passages; none is a
    model's opinion of quality. `unsupported` are sentences carrying no
    citation marker, `unaddressed` are subjects the question raises that no
    retrieved passage mentions, and `conflicts` are pairs of cited passages a
    model flagged and whose ids were confirmed to have been retrieved.
    """

    supported: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()
    unaddressed: tuple[str, ...] = ()
    conflicts: tuple[tuple[str, str], ...] = ()
    action: str = "accept"

    @property
    def clean(self) -> bool:
        return not (self.unsupported or self.unaddressed or self.conflicts)


@runtime_checkable
class Planner(Protocol):
    """Turns a topic into the questions a review will answer."""

    def __call__(self, topic: str, *, max_sections: int) -> Plan: ...


@runtime_checkable
class Retriever(Protocol):
    """Finds candidate evidence for one question, by one strategy."""

    def __call__(self, question: str, *, strategy: str, limit: int) -> Gathered: ...


@runtime_checkable
class Writer(Protocol):
    """Writes an answer that may cite only what it was given."""

    def __call__(self, question: str, gathered: Gathered) -> Answer: ...


@runtime_checkable
class Verifier(Protocol):
    """Decides whether a section stands. Deterministic, by contract."""

    def __call__(self, answer: Answer, closeness: float | None) -> Verdict: ...


@runtime_checkable
class Critic(Protocol):
    """Says what is unsupported or unaddressed. Never whether it is good."""

    def __call__(self, question: str, answer: Answer, gathered: Gathered) -> Critique: ...


ROLES: dict[str, type] = {
    "planner": Planner,
    "retriever": Retriever,
    "writer": Writer,
    "verifier": Verifier,
    "critic": Critic,
}


def satisfies(role: str, candidate: object) -> bool:
    """Whether something can stand in for a role. For tests, not for calls."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; known: {', '.join(sorted(ROLES))}")
    return isinstance(candidate, ROLES[role])


class BudgetGuard:
    """Tracks a run against its ceiling, and says when to stop.

    Reads spend from the model client rather than accumulating its own count,
    so it cannot drift from what was actually spent. Sections may run in
    parallel, so `attempts` is incremented under a lock; the token and time
    reads are already atomic.
    """

    def __init__(self, budget: Budget, usage: object = None, clock=None) -> None:
        import threading
        from datetime import UTC, datetime

        self.budget = budget
        self.usage = usage
        self._clock = clock or (lambda: datetime.now(UTC).timestamp())
        self._started = self._clock()
        self._attempts = 0
        self._lock = threading.Lock()
        self._spent_at_start = self._tokens()

    def _tokens(self) -> int:
        usage = self.usage
        if usage is None:
            return 0
        return int(getattr(usage, "prompt_tokens", 0)) + int(getattr(usage, "output_tokens", 0))

    def take(self) -> None:
        """Count one attempt against the budget."""
        with self._lock:
            self._attempts += 1

    def exhausted(self) -> str | None:
        """Why the run should stop, or None to continue."""
        budget = self.budget
        if budget.max_attempts is not None and self._attempts >= budget.max_attempts:
            return f"{self._attempts} attempts reached the limit of {budget.max_attempts}"
        if budget.max_tokens is not None:
            spent = self._tokens() - self._spent_at_start
            if spent >= budget.max_tokens:
                return f"{spent:,} tokens reached the limit of {budget.max_tokens:,}"
        if budget.max_seconds is not None:
            elapsed = self._clock() - self._started
            if elapsed >= budget.max_seconds:
                return f"{elapsed:.0f}s reached the limit of {budget.max_seconds:.0f}s"
        return None
