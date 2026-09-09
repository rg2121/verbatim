"""Stage events, kept apart from the output a person reads."""

import json
import logging

from verbatim import logs


def test_events_go_to_stderr_so_parseable_stdout_stays_clean() -> None:
    """`runs --json` and the chunk export write to stdout; a log line in the
    middle of either would corrupt it."""
    logs._configured = False
    logging.getLogger(logs.LOGGER_NAME).handlers.clear()
    logs.configure_logging()

    (handler,) = logging.getLogger(logs.LOGGER_NAME).handlers
    import sys

    assert handler.stream is sys.stderr


def test_the_two_formats_carry_the_same_fields() -> None:
    record = logging.LogRecord(
        "verbatim", logging.INFO, __file__, 1, "run.finish", (), None
    )
    record.fields = {"command": "embed", "ok": True, "seconds": 1.5, "error": None}

    as_json = json.loads(logs.JsonFormatter().format(record))
    as_text = logs.KeyValueFormatter().format(record)

    assert as_json["event"] == "run.finish" and as_json["command"] == "embed"
    assert as_json["ok"] is True and as_json["error"] is None
    assert "command=embed" in as_text and "ok=True" in as_text
    # An absent value reads as a dash rather than as the string "None".
    assert "error=-" in as_text


def test_a_value_with_spaces_is_quoted_so_the_line_still_parses() -> None:
    record = logging.LogRecord("verbatim", logging.ERROR, __file__, 1, "run.failed", (), None)
    record.fields = {"error": "RuntimeError: Cannot reach Ollama"}

    assert 'error="RuntimeError: Cannot reach Ollama"' in logs.KeyValueFormatter().format(record)


def test_logging_never_raises_into_the_work() -> None:
    """A broken logger must not fail a command that otherwise succeeded."""
    logs.event("run.start", command="chunk", weird=object())  # not JSON-serialisable
