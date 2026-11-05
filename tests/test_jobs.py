"""Slow work for an MCP client: queued, run in turn, stopped on request, readable later."""

from contextlib import closing
import sqlite3
import subprocess
import threading
import time

import pytest

from verbatim import config
from verbatim.jobs import (
    WAITING_LIMIT,
    JobCancelled,
    JobRunner,
    JobStore,
    Refused,
    Watched,
    preflight,
)


def _settled(runner: JobRunner, job_id: str, seconds: float = 5) -> dict:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        view = runner.view(job_id)
        if view["status"] not in ("queued", "running"):
            return view
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish: {runner.view(job_id)}")


class Gate:
    """Work that holds its job open until released, so a test can act meanwhile."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()
        self.ran: list[str] = []

    def __call__(self, job, progress) -> dict:
        self.ran.append(job.inputs["question"])
        self.entered.set()
        self.release.wait(5)
        progress.check()
        return {"answer": f"answered {job.inputs['question']}"}


def test_a_job_runs_on_its_own_thread_and_its_result_is_kept(tmp_path) -> None:
    runner = JobRunner(JobStore(tmp_path / "corpus.db"), lambda job, progress: {"answer": 42})

    started = runner.submit("ask", {"question": "q"})
    done = _settled(runner, started["job_id"])

    assert started["status"] in ("queued", "running")
    assert done["status"] == "done" and done["result"] == {"answer": 42}
    assert done["asked"] == "q" and done["error"] is None


def test_another_copy_of_the_server_reads_a_finished_job(tmp_path) -> None:
    """Claude Desktop ran two server processes; state in one is invisible to the other."""
    state = tmp_path / "corpus.db"
    first = JobRunner(JobStore(state), lambda job, progress: {"answer": "kept"})
    job_id = first.submit("ask", {"question": "q"})["job_id"]
    _settled(first, job_id)

    other = JobRunner(JobStore(state), lambda job, progress: {})

    assert other.view(job_id)["result"] == {"answer": "kept"}
    assert not state.exists()  # the jobs live beside the corpus, not in it


def test_a_job_whose_process_has_gone_is_reported_lost(tmp_path) -> None:
    store = JobStore(tmp_path / "corpus.db")
    runner = JobRunner(store, lambda job, progress: {})
    gone = subprocess.Popen(["true"])
    gone.wait()
    # The row a server left behind when it was quit mid-job.
    with closing(sqlite3.connect(store.path)) as database, database:
        database.execute(JobStore.SCHEMA)
        database.execute(
            "INSERT INTO jobs (job_id, kind, inputs_json, status, stage, pid, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("abandoned", "ask", '{"question": "q"}', "running", "writing the answer",
             gone.pid, "2026-11-05T20:00:00+00:00"),
        )

    view = runner.view("abandoned")

    assert view["status"] == "lost"
    assert "stopped before it finished" in view["error"]
    assert "again" in view["next"]


def test_an_unknown_job_is_named_in_the_refusal(tmp_path) -> None:
    runner = JobRunner(JobStore(tmp_path / "corpus.db"), lambda job, progress: {})

    with pytest.raises(ValueError, match="no job 'nope'"):
        runner.view("nope")
    with pytest.raises(ValueError, match="job_id is empty"):
        runner.view("  ")


def test_jobs_wait_their_turn_and_a_full_queue_is_refused(tmp_path) -> None:
    gate = Gate()
    runner = JobRunner(JobStore(tmp_path / "corpus.db"), gate)
    first = runner.submit("ask", {"question": "first"})["job_id"]
    assert gate.entered.wait(5)

    waiting = [runner.submit("ask", {"question": f"q{n}"}) for n in range(WAITING_LIMIT)]
    with pytest.raises(Refused, match="already waiting"):
        runner.submit("ask", {"question": "one too many"})

    assert [view["position"] for view in waiting] == list(range(1, WAITING_LIMIT + 1))
    assert all(view["status"] == "queued" for view in waiting)
    gate.release.set()
    for view in waiting:
        _settled(runner, view["job_id"])
    assert gate.ran == ["first"] + [f"q{n}" for n in range(WAITING_LIMIT)]
    assert runner.view(first)["status"] == "done"


def test_a_cancelled_job_that_was_waiting_never_runs(tmp_path) -> None:
    gate = Gate()
    runner = JobRunner(JobStore(tmp_path / "corpus.db"), gate)
    first = runner.submit("ask", {"question": "first"})["job_id"]
    assert gate.entered.wait(5)
    second = runner.submit("ask", {"question": "second"})["job_id"]

    cancelled = runner.cancel(second)
    gate.release.set()
    _settled(runner, first)
    # The job after it runs, so the queue moved past the cancelled one.
    _settled(runner, runner.submit("ask", {"question": "third"})["job_id"])

    assert cancelled["status"] == "cancelled"
    assert gate.ran == ["first", "third"]
    assert runner.view(second)["status"] == "cancelled"


def test_a_running_job_stops_at_its_next_model_call(tmp_path) -> None:
    gate = Gate()
    runner = JobRunner(JobStore(tmp_path / "corpus.db"), gate)
    job_id = runner.submit("ask", {"question": "first"})["job_id"]
    assert gate.entered.wait(5)

    stopping = runner.cancel(job_id)
    gate.release.set()
    finished = _settled(runner, job_id)

    assert stopping["status"] == "running" and "stopping" in stopping["stage"]
    assert finished["status"] == "cancelled"
    assert finished["result"] is None


def test_a_cancel_stops_a_job_at_its_next_section_too(tmp_path) -> None:
    def work(job, progress):
        job.stop.set()
        progress.enter("section 2 of 4 (What is flux?)")
        return {"reached": "the end"}

    runner = JobRunner(JobStore(tmp_path / "corpus.db"), work)

    assert _settled(runner, runner.submit("report", {"topic": "t"})["job_id"])["status"] == "cancelled"


def test_a_result_the_log_cannot_store_does_not_stop_the_next_job(tmp_path) -> None:
    results = iter([{"odd": {1, 2}}, {"plain": 1}])
    runner = JobRunner(JobStore(tmp_path / "corpus.db"), lambda job, progress: next(results))

    first = _settled(runner, runner.submit("ask", {"question": "q1"})["job_id"])
    second = _settled(runner, runner.submit("ask", {"question": "q2"})["job_id"])

    assert first["status"] == "done" and second["result"] == {"plain": 1}


def test_a_cancel_from_another_copy_of_the_server_reaches_the_job(tmp_path) -> None:
    state = tmp_path / "corpus.db"
    gate = Gate()
    runner = JobRunner(JobStore(state), gate)
    job_id = runner.submit("ask", {"question": "first"})["job_id"]
    assert gate.entered.wait(5)

    JobRunner(JobStore(state), lambda job, progress: {}).cancel(job_id)
    gate.release.set()

    assert _settled(runner, job_id)["status"] == "cancelled"


def test_a_failed_job_says_why_in_the_pipelines_own_words(tmp_path) -> None:
    def unreachable(job, progress):
        raise RuntimeError("Cannot reach Ollama at http://localhost:11434. Is it running?")

    runner = JobRunner(JobStore(tmp_path / "corpus.db"), unreachable)
    failed = _settled(runner, runner.submit("ask", {"question": "q"})["job_id"])

    assert failed["status"] == "failed"
    assert "Cannot reach Ollama" in failed["error"]


def test_an_unexpected_failure_keeps_its_details_off_the_wire(tmp_path, capsys) -> None:
    def broken(job, progress):
        raise KeyError("/Users/someone/secret/path")

    runner = JobRunner(JobStore(tmp_path / "corpus.db"), broken)
    failed = _settled(runner, runner.submit("ask", {"question": "q"})["job_id"])

    assert failed["status"] == "failed"
    assert "secret" not in failed["error"] and "KeyError" in failed["error"]
    # The server's log, which only the user reads, keeps the details.
    assert "secret" in capsys.readouterr().err


class Model:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.usage = object()

    def complete(self, system, user, schema):
        self.calls.append("complete")
        return {"answer": "a"}

    def judge(self, system, user):
        self.calls.append("judge")
        return "a subject"


def test_each_model_call_is_named_as_the_jobs_stage(tmp_path) -> None:
    seen: list[str] = []

    def work(job, progress):
        model = Watched(Model(), progress)
        model.complete("s", "u", {})
        seen.append(job.stage)
        progress.enter("section 2 of 3 (What is flux?)")
        model.judge("s", "u")
        seen.append(job.stage)
        return {"usage_passes_through": model.usage is not None}

    runner = JobRunner(JobStore(tmp_path / "corpus.db"), work)
    done = _settled(runner, runner.submit("report", {"topic": "t"})["job_id"])

    assert seen == [
        "writing the answer",
        "section 2 of 3 (What is flux?): naming what the passages are about",
    ]
    assert done["result"] == {"usage_passes_through": True}


def test_a_cancel_cannot_be_swallowed_by_a_fallback_that_catches_everything(tmp_path) -> None:
    """Planning falls back to the whole question on any failure; a cancel must not."""
    model = Model()

    def work(job, progress):
        watched = Watched(model, progress)
        job.stop.set()
        try:
            watched.complete("s", "u", {})
        except Exception:  # noqa: BLE001 - the fallback under test
            return {"fell_back": True}
        return {"fell_back": False}

    runner = JobRunner(JobStore(tmp_path / "corpus.db"), work)
    finished = _settled(runner, runner.submit("ask", {"question": "q"})["job_id"])

    assert issubclass(JobCancelled, BaseException) and not issubclass(JobCancelled, Exception)
    assert finished["status"] == "cancelled"
    assert model.calls == []


LOCAL = [{"name": "gemma4:e4b-mlx", "size": 9_639_236_355}]


def _listing(models):
    return lambda host, timeout: models


def test_a_cloud_model_is_refused_before_ollama_is_asked_anything() -> None:
    def must_not_be_called(host, timeout):
        raise AssertionError("asked Ollama")

    with pytest.raises(Refused, match="cloud model") as refused:
        preflight({"writer": "gemma4:31b-cloud"}, listed=must_not_be_called)

    assert "gemma4:31b-cloud" in str(refused.value) and "--llm" in str(refused.value)


def test_a_cloud_refusal_says_how_to_allow_one() -> None:
    with pytest.raises(Refused, match="--allow-cloud"):
        preflight({"writer": "gemma4:31b-cloud"}, listed=_listing(LOCAL))


def test_a_cloud_model_runs_when_the_server_allows_it() -> None:
    served = [{"name": "gemma4:31b-cloud", "remote_host": "https://ollama.com", "size": 312}]

    preflight({"writer": "gemma4:31b-cloud"}, allow_cloud=True, listed=_listing(served))


def test_allowing_the_cloud_still_needs_the_model_to_be_there() -> None:
    with pytest.raises(Refused, match="no model named 'gpt-oss:120b-cloud'"):
        preflight({"writer": "gpt-oss:120b-cloud"}, allow_cloud=True, listed=_listing(LOCAL))


def test_a_role_set_to_a_cloud_model_names_its_setting(monkeypatch) -> None:
    monkeypatch.setitem(config.ROLE_MODELS, "reflector", "glm-4.6:cloud")

    with pytest.raises(Refused, match="OLLAMA_MODEL_REFLECTOR"):
        preflight({"writer": "gemma4:e4b-mlx", "reflector": "glm-4.6:cloud"}, listed=_listing(LOCAL))


def test_a_model_ollama_serves_from_elsewhere_is_refused_too() -> None:
    remote = [{"name": "plain-name:7b", "remote_host": "https://ollama.com", "size": 312}]

    with pytest.raises(Refused, match="cloud model"):
        preflight({"writer": "plain-name:7b"}, listed=_listing(remote))


def test_a_missing_model_is_named_with_what_is_pulled() -> None:
    with pytest.raises(Refused, match="no model named 'llama9'") as refused:
        preflight({"writer": "llama9"}, listed=_listing(LOCAL))

    assert "gemma4:e4b-mlx" in str(refused.value)


def test_a_model_named_without_its_latest_tag_is_found() -> None:
    preflight({"writer": "llama3.1"}, listed=_listing([{"name": "llama3.1:latest"}]))


def test_ollama_not_running_is_refused_at_once_rather_than_after_retries() -> None:
    started = time.perf_counter()

    with pytest.raises(Refused, match="Ollama is not answering"):
        # Nothing listens on port 9 here; a refusal, not a minute of retries.
        preflight({"writer": "gemma4:e4b-mlx"}, host="http://127.0.0.1:9", timeout=2)

    assert time.perf_counter() - started < 5
