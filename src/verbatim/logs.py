"""Machine-readable events for the stages, beside the prose for the person.

The CLI's printed output is for whoever is watching it: aligned columns,
counts in context, a progress line that rewrites itself. None of that survives
being parsed, and an ingest that runs unattended is read by something other
than a person - a cron mail, a log shipper, a dashboard, a future session
asking what happened at 03:00.

So the two are separated rather than one being bent into the other. Results
keep going to stdout unchanged. Events go to stderr as one line each, either
`key=value` for a terminal or JSON for anything else, and carry the same
fields the run ledger stores. The ledger is the durable record and these are
the stream; they agree because both are written from the same place.

    VERBATIM_LOG_FORMAT=json   one JSON object per line
    VERBATIM_LOG_FORMAT=text   key=value, the default
    VERBATIM_LOG_LEVEL=debug   default info
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

LOGGER_NAME = "verbatim"
_configured = False


class KeyValueFormatter(logging.Formatter):
    """`2026-09-09T12:00:00 info run.finish command=embed ok=True`."""

    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", {})
        rendered = " ".join(f"{key}={_terse(value)}" for key, value in fields.items())
        stamp = self.formatTime(record, "%Y-%m-%dT%H:%M:%S")
        return f"{stamp} {record.levelname.lower()} {record.getMessage()} {rendered}".rstrip()


class JsonFormatter(logging.Formatter):
    """One object per line, for anything that parses rather than reads."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "at": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        return json.dumps(payload, default=str)


def _terse(value: Any) -> str:
    text = "-" if value is None else str(value)
    return f'"{text}"' if " " in text else text


def configure_logging(fmt: str | None = None, level: str | None = None) -> None:
    """Attach one stderr handler. Safe to call more than once.

    stderr rather than stdout on purpose: `runs --json` and the chunk export
    write parseable output to stdout, and a log line landing in the middle of
    either would corrupt it.
    """
    global _configured
    from verbatim import config

    logger = logging.getLogger(LOGGER_NAME)
    chosen = (fmt or config.LOG_FORMAT).lower()
    logger.setLevel(getattr(logging, (level or config.LOG_LEVEL).upper(), logging.INFO))
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if chosen == "json" else KeyValueFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    _configured = True


def event(name: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one structured event. Never raises: logging is not the work."""
    try:
        logging.getLogger(LOGGER_NAME).log(level, name, extra={"fields": fields})
    except Exception:  # noqa: BLE001 - a broken log must not fail a command
        pass
