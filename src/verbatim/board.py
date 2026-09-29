"""A shared evidence board for the `team` workers.

Workers append typed records - tasks, evidence, claims - and the coordinator
decides what to do next from what is missing. Records are never edited or
deleted, and the board only counts: it never judges whether evidence is good.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import itertools

OPEN = "open"
SATISFIED = "satisfied"
ABANDONED = "abandoned"

DIRECT = "direct"
ADVERSARIAL = "adversarial"


@dataclass(frozen=True)
class Task:
    """One thing the answer needs established before it can be written."""

    task_id: str
    objective: str
    needs: int = 1
    status: str = OPEN


@dataclass(frozen=True)
class Evidence:
    """A passage a worker put forward for a task.

    `stance` records what the worker was looking for, such as counter-evidence,
    not what the passage proves.
    """

    evidence_id: str
    task_id: str
    chunk_id: str
    arxiv_id: str
    quote: str
    found_by: str
    stance: str = DIRECT


@dataclass(frozen=True)
class Claim:
    """A statement, and the evidence rows it was written from."""

    claim_id: str
    task_id: str
    statement: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class Missing:
    """Something the board can see is absent. Structural, never evaluative."""

    task_id: str
    kind: str
    detail: str


@dataclass(frozen=True)
class Decision:
    """What the coordinator did next, and which gap made it do that."""

    step: int
    action: str
    task_id: str
    because: str


@dataclass
class ResearchState:
    """The board. Workers append; code reads."""

    question: str
    tasks: list[Task] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    _ids: itertools.count = field(
        default_factory=lambda: itertools.count(1), repr=False
    )

    def add_task(self, objective: str, *, needs: int = 1) -> Task:
        task = Task(task_id=f"t{next(self._ids)}", objective=objective, needs=needs)
        self.tasks.append(task)
        return task

    def add_evidence(
        self,
        task_id: str,
        *,
        chunk_id: str,
        arxiv_id: str,
        quote: str,
        found_by: str,
        stance: str = DIRECT,
    ) -> Evidence | None:
        """Append one passage, or return None if the task already has this chunk.

        A duplicate would inflate the count that decides whether a task is done.
        """
        if any(
            e.task_id == task_id and e.chunk_id == chunk_id for e in self.evidence
        ):
            return None
        row = Evidence(
            evidence_id=f"e{next(self._ids)}", task_id=task_id, chunk_id=chunk_id,
            arxiv_id=arxiv_id, quote=quote, found_by=found_by, stance=stance,
        )
        self.evidence.append(row)
        return row

    def add_claim(
        self, task_id: str, statement: str, evidence_ids: Sequence[str]
    ) -> Claim | None:
        """Append a claim, or return None unless all its evidence was filed for its task."""
        known = {e.evidence_id for e in self.evidence if e.task_id == task_id}
        if not evidence_ids or not set(evidence_ids) <= known:
            return None
        claim = Claim(
            claim_id=f"c{next(self._ids)}", task_id=task_id, statement=statement,
            evidence_ids=tuple(evidence_ids),
        )
        self.claims.append(claim)
        return claim

    def record(self, action: str, task_id: str, because: str) -> None:
        self.decisions.append(
            Decision(step=len(self.decisions) + 1, action=action,
                     task_id=task_id, because=because)
        )

    def close(self, task_id: str, status: str) -> None:
        self.tasks = [
            replace(t, status=status) if t.task_id == task_id else t
            for t in self.tasks
        ]

    # -- reading the board: arithmetic only ------------------------------

    def evidence_for(self, task_id: str, *, stance: str | None = None) -> list[Evidence]:
        return [
            e for e in self.evidence
            if e.task_id == task_id and (stance is None or e.stance == stance)
        ]

    def papers_for(self, task_id: str) -> set[str]:
        return {e.arxiv_id for e in self.evidence_for(task_id)}

    def open_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.status == OPEN]

    def attempted(self, task_id: str, worker: str) -> bool:
        """Whether a worker was already sent at this task, whatever it found."""
        return any(
            d.task_id == task_id and d.action == worker for d in self.decisions
        )

    def render(self) -> str:
        lines = [f"  question: {self.question}", ""]
        for task in self.tasks:
            direct = len(self.evidence_for(task.task_id, stance=DIRECT))
            adversarial = len(self.evidence_for(task.task_id, stance=ADVERSARIAL))
            papers = len(self.papers_for(task.task_id))
            lines.append(
                f"  [{task.status}] {task.task_id} {task.objective}"
            )
            lines.append(
                f"      {direct} direct, {adversarial} adversarial, {papers} paper(s),"
                f" {len([c for c in self.claims if c.task_id == task.task_id])} claim(s)"
            )
        if self.decisions:
            lines.append("")
            lines.append("  what the board made the coordinator do:")
            for d in self.decisions:
                lines.append(f"    {d.step}. {d.action} on {d.task_id} - {d.because}")
        return "\n".join(lines)
