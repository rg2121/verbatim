"""The data the orchestrator's roles pass between them.

Both supervisors, the built-in loop and the LangGraph one, work only in these
types, so their ledgers can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import threading
from typing import Any

from verbatim.corpus import ChunkMatch


@dataclass(frozen=True)
class Gathered:
    """Passages for one question, and how they were found.

    `route` records what a tool-choosing rung called. `records` are catalogue
    rows returned instead of passages, as for a recency question.
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
    """A ceiling for a whole run. None means unbounded, the default."""

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

    Every field is counted or checked against the passages, never a model's
    opinion: `unsupported` sentences carry no citation, `unaddressed` subjects
    appear in no passage, and `conflicts` are retrieved passages a model
    flagged as disagreeing.
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
    """Tracks a run against its budget and says when to stop.

    Spend is read from the model clients, each counted once, so it cannot drift
    from what was spent. Attempts are counted under a lock because sections may
    run in parallel.
    """

    def __init__(self, budget: Budget, *usages: object, clock=None) -> None:
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
