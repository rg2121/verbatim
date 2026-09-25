"""The data the orchestrator's roles pass between them.

The roles - planner, retriever, writer, verifier, critic - are plain
functions, and these frozen dataclasses are what they hand each other: what
was gathered and how, what a step cost, what a run may spend, and what the
verifier and critic concluded. Both supervisors, the built-in loop and the
LangGraph one, work in these types and nothing else, which is what lets their
ledgers be compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verbatim.corpus import ChunkMatch


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


class BudgetGuard:
    """Tracks a run against its ceiling, and says when to stop.

    Reads spend from the model clients rather than accumulating its own count,
    so it cannot drift from what was actually spent. Sections may run in
    parallel, so `attempts` is incremented under a lock; the token and time
    reads are already atomic.

    Every client's meter is read, each once. The answering model and the
    tool-choosing one keep separate counters, and reading only the first let
    a run's routing turns spend past its ceiling; they can also be one client
    with one counter, which must not be counted twice.
    """

    def __init__(self, budget: Budget, *usages: object, clock=None) -> None:
        import threading
        from datetime import UTC, datetime

        self.budget = budget
        self.usages = list({id(u): u for u in usages if u is not None}.values())
        self._clock = clock or (lambda: datetime.now(UTC).timestamp())
        self._started = self._clock()
        self._attempts = 0
        self._lock = threading.Lock()
        self._spent_at_start = self._tokens()

    def _tokens(self) -> int:
        return sum(
            int(getattr(usage, "prompt_tokens", 0)) + int(getattr(usage, "output_tokens", 0))
            for usage in self.usages
        )

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
