"""Run the ingest stages in rounds until nothing is left to do.

Each stage creates work for the next, so one pass in order leaves work behind.
Rounds repeat until the pending counts stop falling. Each stage runs as its
own process, so a stage that fails or uses a lot of memory cannot take the
driver down.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import subprocess
import sys

# Each stage's output is the next one's input; `index` reconciles the vectors last.
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
    # Wait for the HTML check, so a later rendering does not discard PDF chunks.
    "chunk": ("--only-checked",),
    # Skip chunks an older chunker made; re-chunking would discard their vectors.
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
    """Run one stage as its own process; a failure is reported, not raised."""
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
        return False, stderr[-1] if stderr else detail
    if not output:
        # Every stage prints what it did; silence means it never ran.
        return False, "stage produced no output; it may not have run"
    return True, detail


def run_pipeline(
    pending: Callable[[], dict[str, int]],
    *,
    state: str,
    stages: Sequence[tuple[str, str]] = STAGES,
    execute: Callable[[str, Sequence[str], str], tuple[bool, str]] | None = None,
    max_rounds: int = MAX_ROUNDS,
    on_stage: Callable[[StageResult], None] | None = None,
    stage_arguments: dict[str, tuple[str, ...]] | None = None,
) -> list[RoundResult]:
    """Run the stages in rounds until nothing is pending or a round changes nothing.

    Some work never completes (a PDF that is always truncated), so a round with
    no progress ends the run. `stage_arguments` adds arguments per stage.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    runner = execute or (lambda name, args, st: run_stage(name, args, state=st))
    rounds: list[RoundResult] = []

    for number in range(1, max_rounds + 1):
        result = RoundResult(number=number)
        for name, counter in stages:
            before = pending().get(counter, 0) if counter else 0
            # A stage with nothing to do is skipped rather than started.
            if counter and before == 0:
                continue
            arguments = (
                *STAGE_ARGUMENTS.get(name, ()), *(stage_arguments or {}).get(name, ())
            )
            ok, detail = runner(name, arguments, state)
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
            # Negative means work was done.
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
