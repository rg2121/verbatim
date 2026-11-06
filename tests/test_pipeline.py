"""Running the ingest stages to a fixpoint."""

import pytest

from verbatim.pipeline import (
    LIMITED_STAGES,
    STAGES as INGEST_STAGES,
    RoundResult,
    StageResult,
    render_left,
    render_rounds,
    run_pipeline,
)

STAGES = (("alpha", "alpha_pending"), ("beta", "beta_pending"))


def driver(state: dict[str, int], effects: dict[str, dict[str, int]], failures=()):
    """A fake stage runner that applies `effects` to the pending counts."""
    calls: list[str] = []

    def execute(name, arguments, st):
        calls.append(name)
        if name in failures:
            return False, f"{name} exploded"
        for counter, delta in effects.get(name, {}).items():
            state[counter] = max(state[counter] + delta, 0)
        return True, f"{name} ok"

    return execute, calls


def test_work_created_by_an_earlier_stage_is_picked_up_in_a_later_round() -> None:
    """Running each stage once leaves behind whatever arrived after it ran."""
    state = {"alpha_pending": 2, "beta_pending": 0}
    # alpha clears itself but creates work for beta, which has already run.
    execute, calls = driver(state, {"alpha": {"alpha_pending": -2, "beta_pending": 5},
                                    "beta": {"beta_pending": -5}})

    rounds = run_pipeline(lambda: dict(state), state="x", stages=STAGES, execute=execute)

    assert state == {"alpha_pending": 0, "beta_pending": 0}
    assert calls.count("beta") >= 1
    assert len(rounds) >= 2


def test_a_stage_with_nothing_pending_is_not_started() -> None:
    """A settled corpus should cost a query per stage, not a process."""
    state = {"alpha_pending": 0, "beta_pending": 0}
    execute, calls = driver(state, {})

    run_pipeline(lambda: dict(state), state="x", stages=STAGES, execute=execute)

    assert calls == []


def test_work_that_never_completes_does_not_loop_for_ever() -> None:
    """33 papers here return a truncated PDF every time they are asked for."""
    state = {"alpha_pending": 33, "beta_pending": 0}
    execute, calls = driver(state, {})  # alpha never reduces its count

    rounds = run_pipeline(lambda: dict(state), state="x", stages=STAGES, execute=execute)

    assert len(rounds) == 1  # no progress in the first round, so it stops
    assert calls == ["alpha"]


def test_a_failing_stage_does_not_stop_the_ones_after_it() -> None:
    """A DNS failure took out a licence run; the ingest should have continued."""
    state = {"alpha_pending": 1, "beta_pending": 1}
    execute, calls = driver(state, {"beta": {"beta_pending": -1}}, failures={"alpha"})

    rounds = run_pipeline(lambda: dict(state), state="x", stages=STAGES, execute=execute)

    assert "beta" in calls
    assert state["beta_pending"] == 0
    failed = {s.name for r in rounds for s in r.stages if not s.ok}
    assert failed == {"alpha"}
    # And it is retried while other stages are still making progress, since
    # the failure that prompted this was a transient network one.
    assert calls.count("alpha") > 1


def test_the_report_says_a_stage_failed_and_what_it_said() -> None:
    rendered = render_rounds([
        RoundResult(number=1, stages=[
            StageResult("alpha", False, 5, 5, "arXiv did not respond"),
        ])
    ])

    assert "FAILED alpha" in rendered
    assert "arXiv did not respond" in rendered
    assert "Re-run to retry them" in rendered


def test_work_done_reads_as_a_fall_in_the_pending_count() -> None:
    rendered = render_rounds([
        RoundResult(number=1, stages=[StageResult("alpha", True, 10, 4)])
    ])

    assert "10 ->       4  (-6)" in rendered


def test_max_rounds_is_honoured() -> None:
    state = {"alpha_pending": 100, "beta_pending": 0}
    # Always progresses, so only the cap can stop it.
    execute, calls = driver(state, {"alpha": {"alpha_pending": -1}})

    rounds = run_pipeline(
        lambda: dict(state), state="x", stages=STAGES, execute=execute, max_rounds=3
    )

    assert len(rounds) == 3


def test_max_rounds_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_pipeline(lambda: {}, state="x", max_rounds=0)


def test_a_stage_that_prints_nothing_is_treated_as_not_having_run() -> None:
    """Silence is how a no-op stage passed for a successful one."""
    import subprocess
    from unittest.mock import patch

    from verbatim.pipeline import run_stage

    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=completed):
        ok, detail = run_stage("chunk", (), state="corpus.db")

    assert not ok
    assert "may not have run" in detail


def test_a_stage_that_reports_what_it_did_is_a_success() -> None:
    import subprocess
    from unittest.mock import patch

    from verbatim.pipeline import run_stage

    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="Chunked 12 papers into 900 passages.\n", stderr=""
    )
    with patch("subprocess.run", return_value=completed):
        ok, detail = run_stage("chunk", (), state="corpus.db")

    assert ok
    assert detail == "Chunked 12 papers into 900 passages."


def test_a_run_can_add_arguments_to_a_stage() -> None:
    seen = {}

    def execute(name, arguments, state):
        seen[name] = tuple(arguments)
        return True, "done"

    counts = {"embedding_pending": 3}

    def pending():
        return dict(counts)

    def run_embed(name, arguments, state):
        counts["embedding_pending"] = 0
        return execute(name, arguments, state)

    run_pipeline(
        pending, state="x", stages=(("embed", "embedding_pending"),), execute=run_embed,
        stage_arguments={"embed": ("--model", "other/model")},
    )

    # The stage's own safety argument is kept, and the run's is added.
    assert seen["embed"] == ("--only-current", "--model", "other/model")


def honours_limit(state: dict[str, int], counter: str, default: int):
    """A stage that clears at most --limit items a run, or its own default without one."""
    calls: list[tuple[str, tuple[str, ...]]] = []

    def execute(name, arguments, st):
        args = tuple(arguments)
        calls.append((name, args))
        limit = int(args[args.index("--limit") + 1]) if "--limit" in args else default
        state[counter] -= min(limit, state[counter])
        return True, f"{name} ok"

    return execute, calls


def limit_of(arguments: tuple[str, ...]) -> str | None:
    return arguments[arguments.index("--limit") + 1] if "--limit" in arguments else None


def test_a_large_sync_is_ingested_in_one_run_not_three_papers_a_round() -> None:
    """`download` takes 3 PDFs a run by default, so after a 1,000-paper sync the
    pipeline added 24 papers in its 8 rounds and stopped."""
    state = {"downloads_pending": 1_000}
    execute, _ = honours_limit(state, "downloads_pending", default=3)

    run_pipeline(
        lambda: dict(state), state="x", stages=(("download", "downloads_pending"),),
        execute=execute,
    )

    assert state["downloads_pending"] == 0


@pytest.mark.parametrize("stage", ["download", "html", "licenses", "references"])
def test_a_stage_with_a_small_default_is_asked_for_everything_pending(stage) -> None:
    counter = f"{stage}_pending"
    state = {counter: 1_234}
    execute, calls = honours_limit(state, counter, default=50)

    run_pipeline(lambda: dict(state), state="x", stages=((stage, counter),), execute=execute)

    assert limit_of(calls[0][1]) == "1234"


def test_a_stage_that_takes_everything_by_default_gets_no_limit() -> None:
    state = {"embedding_pending": 500}
    execute, calls = honours_limit(state, "embedding_pending", default=10**9)

    run_pipeline(
        lambda: dict(state), state="x", stages=(("embed", "embedding_pending"),),
        execute=execute,
    )

    assert limit_of(calls[0][1]) is None


def test_every_limited_stage_is_a_real_stage_that_takes_a_limit() -> None:
    """A renamed stage or a dropped flag would fail at run time, not here."""
    import subprocess
    import sys

    assert LIMITED_STAGES <= {name for name, _ in INGEST_STAGES}
    for stage in sorted(LIMITED_STAGES):
        shown = subprocess.run(
            [sys.executable, "-m", "verbatim.cli", stage, "--help"],
            capture_output=True, text=True, check=True,
        )
        assert "--limit" in shown.stdout, stage


def test_each_stage_is_announced_before_it_runs() -> None:
    """Asked for everything pending, a stage can run for hours; with nothing
    printed until it ended, a long download looked like a hang."""
    state = {"alpha_pending": 4, "beta_pending": 0}
    execute, _ = driver(state, {"alpha": {"alpha_pending": -4}})
    heard: list[tuple[str, int]] = []

    run_pipeline(
        lambda: dict(state), state="x", stages=STAGES, execute=execute,
        on_start=lambda name, pending: heard.append((name, pending)),
    )

    assert heard == [("alpha", 4)]


def test_a_stage_that_keeps_no_count_is_announced_without_one() -> None:
    """`index` has no pending counter, and "0 pending" read as nothing to do."""
    heard: list[tuple[str, int | None]] = []

    run_pipeline(
        lambda: {}, state="x", stages=(("index", ""),),
        execute=lambda name, arguments, st: (True, "indexed"),
        on_start=lambda name, pending: heard.append((name, pending)),
    )

    assert heard == [("index", None)]


def test_the_closing_report_says_when_nothing_is_left() -> None:
    rounds = [RoundResult(number=1, stages=[StageResult("download", True, 3, 0)])]

    rendered = render_left({"downloads_pending": 0}, rounds, max_rounds=8)

    assert "Nothing is left to do" in rendered


def test_the_closing_report_names_what_is_left_after_the_round_limit() -> None:
    """Hitting the round limit used to print the rounds and nothing else."""
    rounds = [
        RoundResult(number=n, stages=[StageResult("download", True, 10 - n, 9 - n)])
        for n in (1, 2)
    ]

    rendered = render_left(
        {"downloads_pending": 7, "embedding_pending": 120}, rounds, max_rounds=2
    )

    assert "2-round limit" in rendered
    assert "download" in rendered and "7 pending" in rendered
    assert "embed" in rendered and "120 pending" in rendered
    assert "verbatim pipeline" in rendered


def test_the_closing_report_says_when_what_is_left_made_no_progress() -> None:
    rounds = [RoundResult(number=1, stages=[StageResult("download", False, 5, 5, "offline")])]

    rendered = render_left({"downloads_pending": 5}, rounds, max_rounds=8)

    assert "made no progress" in rendered
    assert "5 pending" in rendered
