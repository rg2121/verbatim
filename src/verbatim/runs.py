"""What each command actually did, kept where the corpus is kept.

This project measures retrieval carefully and recorded nothing at all about
its own executions. An ingest that runs unattended every night could fail, or
half-finish, or embed with a different model than yesterday, and the only
evidence was whatever scrolled past in a terminal. `stats` says what the
corpus holds now; nothing said how it got there.

One row per command invocation: when it started and finished, whether it
succeeded, what it was asked to do, what it produced, which models it used,
what the corpus looked like before and after, and what the generation cost.
Failures are recorded too - a run that stopped is the one worth finding
later, and a ledger that only holds successes is a ledger of the wrong half.

Recording is best-effort by construction. A command's job is the corpus, not
the bookkeeping, so a bookkeeping error is swallowed rather than allowed to
fail work that otherwise succeeded.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import sqlite3
from typing import Any
from uuid import uuid4

RUNS_TABLE = "runs"
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {RUNS_TABLE} (
    run_id TEXT PRIMARY KEY,
    command TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    seconds REAL,
    ok INTEGER,
    error TEXT,
    inputs_json TEXT NOT NULL DEFAULT '{{}}',
    outputs_json TEXT NOT NULL DEFAULT '{{}}',
    corpus_before_json TEXT NOT NULL DEFAULT '{{}}',
    corpus_after_json TEXT NOT NULL DEFAULT '{{}}'
);
CREATE INDEX IF NOT EXISTS runs_by_command ON {RUNS_TABLE} (command, started_at DESC);
"""
# The counts that make one run comparable with the next. Cheap to take: they
# are four COUNT(*)s against tables that are indexed anyway.
SNAPSHOT = {
    "papers": "SELECT COUNT(*) FROM papers",
    "chunks": "SELECT COUNT(*) FROM chunks",
    "vectors": "SELECT COUNT(*) FROM embeddings",
    "references": "SELECT COUNT(*) FROM paper_references",
}


@dataclass
class Run:
    """The row being written, and the handle a command adds its results to."""

    run_id: str
    command: str
    outputs: dict[str, Any] = field(default_factory=dict)

    def record(self, **outputs: Any) -> None:
        """Note what this run produced. Called as often as is useful."""
        self.outputs.update(outputs)


def _snapshot(connection: sqlite3.Connection) -> dict[str, int]:
    counts = {}
    for name, sql in SNAPSHOT.items():
        try:
            counts[name] = int(connection.execute(sql).fetchone()[0])
        except sqlite3.Error:  # a corpus that predates the table
            pass
    return counts


@contextmanager
def record_run(corpus: Any, command: str, **inputs: Any):
    """Record one command invocation, whether or not it succeeds.

    Yields a Run the caller adds outputs to. An exception is recorded with its
    message and then re-raised: the ledger must not swallow a failure it exists
    to remember.
    """
    connection = corpus.connection
    run = Run(run_id=uuid4().hex[:12], command=command)
    started = datetime.now(UTC)
    before: dict[str, int] = {}
    try:
        connection.executescript(SCHEMA)
        before = _snapshot(connection)
        with connection:
            connection.execute(
                f"INSERT INTO {RUNS_TABLE} (run_id, command, started_at, inputs_json,"
                f" corpus_before_json) VALUES (?, ?, ?, ?, ?)",
                (run.run_id, command, started.isoformat(),
                 json.dumps(_plain(inputs)), json.dumps(before)),
            )
    except Exception:  # noqa: BLE001 - deliberately total; see below
        # Not sqlite3.Error alone: whatever goes wrong writing a row, the
        # command's job is the corpus and the ledger is a convenience. A
        # bookkeeping failure that escaped here would fail work that had
        # otherwise succeeded, which is the opposite of what this is for.
        pass

    error: str | None = None
    try:
        yield run
    except BaseException as failure:  # noqa: BLE001 - recorded, then re-raised
        error = f"{type(failure).__name__}: {failure}"[:500]
        raise
    finally:
        finished = datetime.now(UTC)
        try:
            with connection:
                connection.execute(
                    f"UPDATE {RUNS_TABLE} SET finished_at = ?, seconds = ?, ok = ?,"
                    f" error = ?, outputs_json = ?, corpus_after_json = ?"
                    f" WHERE run_id = ?",
                    (
                        finished.isoformat(),
                        (finished - started).total_seconds(),
                        int(error is None),
                        error,
                        json.dumps(_plain(run.outputs)),
                        json.dumps(_snapshot(connection)),
                        run.run_id,
                    ),
                )
        except Exception:  # noqa: BLE001 - same reason as above
            pass


def _plain(values: dict[str, Any]) -> dict[str, Any]:
    """Whatever a caller passed, in something JSON will accept."""
    plain = {}
    for key, value in values.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            plain[key] = value
        elif isinstance(value, (list, tuple)):
            plain[key] = [str(item) for item in value]
        else:
            plain[key] = str(value)
    return plain


def recent_runs(corpus: Any, *, command: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """The last runs, newest first, with their JSON fields already parsed."""
    try:
        corpus.connection.executescript(SCHEMA)
        rows = corpus.connection.execute(
            f"SELECT * FROM {RUNS_TABLE}"
            + (" WHERE command = ?" if command else "")
            + " ORDER BY started_at DESC LIMIT ?",
            ((command, limit) if command else (limit,)),
        ).fetchall()
    except Exception:  # noqa: BLE001 - an unreadable ledger reads as empty
        return []
    return [_row_to_dict(row) for row in rows]


def last_run_per_command(corpus: Any) -> dict[str, dict[str, Any]]:
    """The most recent run of each command, for a health check to read."""
    latest: dict[str, dict[str, Any]] = {}
    for run in recent_runs(corpus, limit=500):
        latest.setdefault(run["command"], run)
    return latest


def _row_to_dict(row: Any) -> dict[str, Any]:
    record = {key: row[key] for key in row.keys()}
    for field_name in ("inputs", "outputs", "corpus_before", "corpus_after"):
        raw = record.pop(f"{field_name}_json", "") or "{}"
        try:
            record[field_name] = json.loads(raw)
        except ValueError:
            record[field_name] = {}
    record["ok"] = None if record["ok"] is None else bool(record["ok"])
    return record


def render_runs(runs: list[dict[str, Any]]) -> str:
    """The ledger as a table, for a person rather than a machine."""
    if not runs:
        return "  no runs recorded yet"
    lines = [f"  {'when':<20} {'command':<12} {'':<3} {'took':>8}  what it did"]
    for run in runs:
        mark = "ok " if run["ok"] else ("..." if run["ok"] is None else "FAIL")
        took = f"{run['seconds']:.1f}s" if run.get("seconds") else "-"
        detail = ", ".join(f"{k}={v}" for k, v in list(run["outputs"].items())[:4])
        if run.get("error"):
            detail = run["error"][:60]
        lines.append(
            f"  {run['started_at'][:19]:<20} {run['command']:<12} {mark:<3} {took:>8}  {detail}"
        )
    return "\n".join(lines)
