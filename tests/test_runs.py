"""The ledger of what each command did."""

from pathlib import Path

import pytest

from verbatim.corpus import CorpusStore
from verbatim.runs import (
    last_run_per_command,
    recent_runs,
    record_run,
    render_runs,
)


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
