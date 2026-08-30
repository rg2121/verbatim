"""Running the ingest stages until there is nothing left to do.

Each stage creates work for the ones after it: growing the corpus leaves PDFs
to download, downloading leaves renderings to fetch, fetching leaves papers to
chunk, and chunking leaves passages to embed. Running each stage once, in
order, therefore does not finish the job - it leaves behind whatever arrived
after its stage had already run. Done by hand overnight, that stranded 81
papers unchecked, 350 unchunked and 69 bibliographies unread.

So the stages are run in rounds until the pending counts stop falling. Stages
run as separate processes for two reasons: chunking is memory-hungry and is
better bounded by a process boundary, and a stage that dies - a DNS failure
took out a licence run here - must not take the driver down with it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import subprocess
import sys

# Ordered so that each stage's output is the next one's input. `index` last,
# because it reconciles the vector index with whatever embedding produced.
STAGES: tuple[tuple[str, str], ...] = (
    ("download", "downloads_pending"),
    ("html", "html_pending"),
    ("licenses", "licences_pending"),
    ("chunk", "chunking_pending"),
    ("references", "references_pending"),
    ("embed", "embedding_pending"),
    ("index", ""),
)
# Extra arguments a stage needs to be safe inside a loop.
STAGE_ARGUMENTS: dict[str, tuple[str, ...]] = {
    # Reading a paper from its PDF before its rendering has been looked for is
    # work that re-chunking will throw away.
    "chunk": ("--only-checked",),
    # Likewise, embedding a passage an older chunker produced.
    "embed": ("--only-current",),
}
MAX_ROUNDS = 8


@dataclass
class StageResult:
    name: str
    ok: bool
    pending_before: int
    pending_after: int
    detail: str = ""

    @property
    def progressed(self) -> bool:
        return self.pending_after < self.pending_before


@dataclass
class RoundResult:
    number: int
    stages: list[StageResult] = field(default_factory=list)

    @property
    def progressed(self) -> bool:
        return any(stage.progressed for stage in self.stages)


def run_stage(name: str, arguments: Sequence[str], *, state: str) -> tuple[bool, str]:
    """Run one stage as its own process, and report whether it succeeded.

    A stage is allowed to fail. Its failure is recorded and the driver carries
    on, because the alternative - stopping the whole ingest because arXiv was
    briefly unreachable - is what this exists to prevent.
    """
    command = [sys.executable, "-m", "verbatim.cli", name, "--state", state, *arguments]
    try:
        finished = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=None
        )
    except OSError as error:  # pragma: no cover - depends on the environment
        return False, str(error)
    output = (finished.stdout or "").strip().splitlines()
    detail = output[-1] if output else ""
    if finished.returncode != 0:
        stderr = (finished.stderr or "").strip().splitlines()
        detail = stderr[-1] if stderr else detail
    return finished.returncode == 0, detail


def run_pipeline(
    pending: Callable[[], dict[str, int]],
    *,
    state: str,
    stages: Sequence[tuple[str, str]] = STAGES,
    execute: Callable[[str, Sequence[str], str], tuple[bool, str]] | None = None,
    max_rounds: int = MAX_ROUNDS,
    on_stage: Callable[[StageResult], None] | None = None,
) -> list[RoundResult]:
    """Run the stages in rounds until nothing is pending or nothing improves.

    Two stopping conditions, and the second matters more: some work never
    completes. Thirty-three papers in this corpus return a truncated PDF every
    time they are asked for, so "pending reaches zero" would loop for ever. A
    round that changes no count is the real signal that the ingest is done.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    runner = execute or (lambda name, args, st: run_stage(name, args, state=st))
    rounds: list[RoundResult] = []

    for number in range(1, max_rounds + 1):
        result = RoundResult(number=number)
        for name, counter in stages:
            before = pending().get(counter, 0) if counter else 0
            # A stage with nothing to do is skipped rather than started, so a
            # settled corpus costs one query per stage instead of a process.
            if counter and before == 0:
                continue
            ok, detail = runner(name, STAGE_ARGUMENTS.get(name, ()), state)
            after = pending().get(counter, 0) if counter else 0
            stage = StageResult(name, ok, before, after, detail)
            result.stages.append(stage)
            if on_stage is not None:
                on_stage(stage)
        rounds.append(result)
        if not result.stages or not result.progressed:
            break
    return rounds


def render_rounds(rounds: Sequence[RoundResult]) -> str:
    """Summarise what the ingest did, and say plainly if it stopped early."""
    lines: list[str] = []
    for round_result in rounds:
        lines.append(f"  round {round_result.number}")
        for stage in round_result.stages:
            # Signed against the pending count, so work done reads as a fall.
            moved = stage.pending_after - stage.pending_before
            status = "ok " if stage.ok else "FAILED"
            lines.append(
                f"    {status} {stage.name:<12} {stage.pending_before:>7,} -> "
                f"{stage.pending_after:>7,}  ({moved:+,})"
            )
            if not stage.ok and stage.detail:
                lines.append(f"      {stage.detail[:120]}")
    failed = [s for r in rounds for s in r.stages if not s.ok]
    if failed:
        lines.append(
            f"\n  {len(failed)} stage run(s) failed; the rest continued. "
            "Re-run to retry them."
        )
    return "\n".join(lines)
