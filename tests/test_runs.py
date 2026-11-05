"""The ledger of what each command did."""

from pathlib import Path

import pytest

from verbatim.corpus import CorpusStore
from verbatim.runs import (
    ToolCallLog,
    last_run_per_command,
    recent_runs,
    record_run,
    render_runs,
)


def test_a_tool_call_is_logged_beside_the_corpus_not_in_it(tmp_path: Path) -> None:
    state = tmp_path / "corpus.db"
    CorpusStore(state).close()
    before = state.read_bytes()
    log = ToolCallLog(state)

    log.record("search_chunks", inputs={"query": "flux"}, outputs={"results": 3}, seconds=0.25)

    assert state.read_bytes() == before
    assert log.path == tmp_path / "corpus.calls.db"
    with CorpusStore(state) as corpus:
        assert recent_runs(corpus) == []

    (call,) = log.recent()
    assert call["command"] == "search_chunks"
    assert call["ok"] is True and call["error"] is None
    assert call["inputs"] == {"query": "flux"}
    assert call["outputs"] == {"results": 3}
    assert call["seconds"] == 0.25
    assert call["started_at"]


def test_a_failed_call_records_why(tmp_path: Path) -> None:
    log = ToolCallLog(tmp_path / "corpus.db")

    log.record("list_papers", inputs={"days": -5}, outputs=None, seconds=0.0, error="days must be")

    (call,) = log.recent()
    assert call["ok"] is False and call["error"] == "days must be"
    assert call["outputs"] == {}


def test_recent_calls_are_newest_first_and_can_be_filtered(tmp_path: Path) -> None:
    log = ToolCallLog(tmp_path / "corpus.db")
    for tool in ("get_paper", "search_chunks", "get_paper"):
        log.record(tool, inputs={}, outputs={}, seconds=0.0)
    log.record("read_pages", inputs={}, outputs={}, seconds=0.0)

    assert [c["command"] for c in log.recent(limit=2)] == ["read_pages", "get_paper"]
    assert [c["command"] for c in log.recent(tool="get_paper")] == ["get_paper", "get_paper"]


def test_reading_a_log_that_was_never_written_creates_nothing(tmp_path: Path) -> None:
    log = ToolCallLog(tmp_path / "corpus.db")

    assert log.recent() == []
    assert not log.path.exists()


def test_a_log_that_cannot_be_written_never_fails_the_call(tmp_path: Path) -> None:
    log = ToolCallLog(tmp_path / "no" / "such" / "folder" / "corpus.db")

    log.record("get_paper", inputs={}, outputs={}, seconds=0.0)

    assert log.recent() == []


def test_two_servers_can_write_one_log(tmp_path: Path) -> None:
    """Claude Desktop was seen running two copies of the server."""
    one, two = ToolCallLog(tmp_path / "corpus.db"), ToolCallLog(tmp_path / "corpus.db")

    one.record("get_paper", inputs={}, outputs={}, seconds=0.0)
    two.record("read_pages", inputs={}, outputs={}, seconds=0.0)

    assert [c["command"] for c in one.recent()] == ["read_pages", "get_paper"]


def test_the_table_fits_the_longest_tool_name() -> None:
    rendered = render_runs(
        [
            {"started_at": "2026-11-05T17:00:00", "command": "verify_citations", "ok": True,
             "seconds": 0.1, "outputs": {"verified": 2}, "error": None},
            {"started_at": "2026-11-05T17:00:01", "command": "get_paper", "ok": True,
             "seconds": 0.1, "outputs": {}, "error": None},
        ]
    ).splitlines()

    assert rendered[1].index("ok ") == rendered[2].index("ok ")


def test_a_successful_run_records_what_it_did(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        with record_run(corpus, "chunk", limit=300) as run:
            run.record(pdfs=12, passages=840)

        (recorded,) = recent_runs(corpus)

    assert recorded["command"] == "chunk"
    assert recorded["ok"] is True
    assert recorded["inputs"] == {"limit": 300}
    assert recorded["outputs"] == {"pdfs": 12, "passages": 840}
    assert recorded["seconds"] is not None and recorded["error"] is None


def test_a_failed_run_is_recorded_and_the_failure_still_raises(tmp_path: Path) -> None:
    """A ledger that only holds successes is a ledger of the wrong half."""
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        with pytest.raises(RuntimeError, match="unreachable"):
            with record_run(corpus, "embed") as run:
                run.record(chunks_embedded=40)
                raise RuntimeError("Ollama unreachable")

        (recorded,) = recent_runs(corpus)

    assert recorded["ok"] is False
    assert "RuntimeError: Ollama unreachable" in recorded["error"]
    # Whatever it managed before failing is kept.
    assert recorded["outputs"] == {"chunks_embedded": 40}


def test_the_corpus_is_snapshotted_before_and_after(tmp_path: Path) -> None:
    from helpers import chunked_corpus

    with chunked_corpus(tmp_path, {"2608.11111v1": ["Some text."]}) as corpus:
        with record_run(corpus, "chunk"):
            pass

        (recorded,) = recent_runs(corpus)

    assert recorded["corpus_before"]["chunks"] == recorded["corpus_after"]["chunks"]
    assert recorded["corpus_before"]["papers"] == 1


def test_bookkeeping_never_breaks_the_command_it_records(tmp_path: Path) -> None:
    """A command's job is the corpus, not the ledger."""

    class Broken:
        class connection:
            @staticmethod
            def executescript(_sql):
                import sqlite3

                raise sqlite3.OperationalError("disk I/O error")

            @staticmethod
            def execute(*_args):
                import sqlite3

                raise sqlite3.OperationalError("disk I/O error")

    done = []
    with record_run(Broken(), "chunk") as run:
        run.record(passages=1)
        done.append(True)

    assert done == [True]


def test_the_latest_run_of_each_command_is_findable(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        with record_run(corpus, "embed") as run:
            run.record(chunks_embedded=1)
        with record_run(corpus, "embed") as run:
            run.record(chunks_embedded=2)
        with record_run(corpus, "index") as run:
            run.record(added=5)

        latest = last_run_per_command(corpus)
        only_embed = recent_runs(corpus, command="embed")

    assert sorted(latest) == ["embed", "index"]
    assert latest["embed"]["outputs"] == {"chunks_embedded": 2}
    assert len(only_embed) == 2


def test_the_ledger_renders_failures_visibly() -> None:
    rendered = render_runs([
        {"started_at": "2026-09-09T10:00:00", "command": "embed", "ok": False,
         "seconds": 1.0, "outputs": {}, "error": "RuntimeError: no model"},
        {"started_at": "2026-09-09T09:00:00", "command": "chunk", "ok": True,
         "seconds": 2.0, "outputs": {"pdfs": 3}, "error": None},
    ])

    assert "FAIL" in rendered and "no model" in rendered
    assert "pdfs=3" in rendered
    assert render_runs([]) == "  no runs recorded yet"
