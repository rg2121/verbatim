"""A shared evidence board: what the team has established, and what it has not.

The orchestrator's roles are stages in a ladder. Output flows one way, and
nothing a role discovers changes what another role does next - which is a
workflow, not a team. This is the state that makes the difference: workers
append structured records, and what gets done next is read off the board
rather than decided in advance.

Two rules keep it from becoming a place to hide judgements.

**Append-only, and structured.** An agent may add an `Evidence` row naming a
chunk and a quote, or a `Claim` naming the evidence it rests on. It may not
write prose onto the board, and it may not delete or edit what another agent
put there. A role that can only append typed records cannot smuggle an opinion
past the verifier, which is the same reason this project's critic describes
and lets code decide.

**Nothing here judges.** Every question the board answers - which tasks have no
evidence, which are thinner than they asked, what has not been attempted -
is arithmetic over the records. The board says what is missing. It never says
whether what is present is any good.
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
    """A passage a worker put forward, and what it was looking for.

    `stance` records what the worker was *seeking*, not what the passage
    proves. A worker searching for limitations files what it finds as
    `adversarial`, which means "an adversarial query returned this" and
    nothing more. Whether the passage contradicts anything is a judgement this
    board will not make, and the label deliberately does not imply it.
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
        """Append one passage. Returns None if this chunk is already on the board.

        Deduplicated by (task, chunk) because two workers searching the same
        task will find the same passage, and a task's evidence count is used to
        decide whether it is satisfied - counting one passage twice would
        close a task nobody finished.
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
        """Append a claim. Refused if it cites evidence that is not on the board.

        The one place the board enforces rather than records: a claim resting
        on an evidence id nobody filed is unfalsifiable by construction, and
        the whole point of the board is that a statement can be traced to the
        rows it came from.
        """
        known = {e.evidence_id for e in self.evidence}
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
        """Whether a worker has already been sent at this task.

        Read from the decisions rather than from the evidence: a worker that
        ran and found nothing has still been tried, and sending it again would
        be the blind retry this project's own sources warn against.
        """
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
