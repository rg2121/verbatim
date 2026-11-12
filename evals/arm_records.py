"""What both arms of the v2 evaluation read and write, so neither can drift.

The v1 worker, the v2 worker and the scorer all go through this module, which is
pinned with v1 before any v2 code exists. A set file is used only if its bytes
still match the SHA-256 the manifest records for it, wherever the file lives,
so sets frozen after the pin are checked the same way. The record carries
everything a scorer needs, and the scorer recomputes what matters from it
rather than trusting the arm that wrote it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from verbatim.answers import Answer

MANIFEST = Path(__file__).parent / "v2" / "manifest.json"

# Which arm reads which kind of set.
KINDS = {
    "questions": "ask",
    "abstention": "ask",
    "breadth": "team",
    "topics": "report",
    "claims": "gaps",
}

COMPLETE = "complete"
ERROR = "error"

# Pre-registered: these stop the driver, and the question is rerun for both arms.
# Anything else an arm raises counts against that arm.
INFRASTRUCTURE = "infrastructure"
ARM = "arm"


class SetChanged(ValueError):
    """A set file no longer matches the hash it was frozen with."""


@dataclass(frozen=True)
class SetEntry:
    """One frozen set: where it is, what it hashes to, and what it holds."""

    name: str
    path: Path
    sha256: str
    kind: str

    @property
    def arm(self) -> str:
        return KINDS[self.kind]


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: Path) -> dict[str, SetEntry]:
    """Every set the manifest names; paths are relative to the manifest's folder."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: dict[str, SetEntry] = {}
    for name, item in payload["sets"].items():
        if item["kind"] not in KINDS:
            raise ValueError(f"set {name!r} has unknown kind {item['kind']!r}")
        entries[name] = SetEntry(
            name=name,
            path=(path.parent / item["path"]).resolve(),
            sha256=item["sha256"],
            kind=item["kind"],
        )
    return entries


def read_set(entry: SetEntry) -> list[dict[str, Any]]:
    """The set's rows, refusing a file whose bytes differ from its frozen hash."""
    data = entry.path.read_bytes()
    found = sha256_of(data)
    if found != entry.sha256:
        raise SetChanged(
            f"{entry.path} hashes to {found}, but set {entry.name!r} was frozen "
            f"as {entry.sha256}. A changed set is a different experiment."
        )
    rows = []
    for line in data.decode("utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            rows.append(json.loads(stripped))
    return rows


def identity(row: dict[str, Any]) -> str:
    """The text a row is known by: its question, claim or topic.

    A claim comes before a topic because a frozen claim also carries its
    paper's title as the topic the Falsifier is given.
    """
    for key in ("question", "claim", "topic"):
        if key in row:
            return str(row[key])
    raise ValueError(f"row has no question, topic or claim: {sorted(row)}")


def case_id(row: dict[str, Any]) -> str:
    """A stable id from the row's own text, so both arms name a case alike."""
    return sha256_of(identity(row).encode("utf-8"))[:16]


def answer_fields(answer: Answer | None) -> dict[str, Any] | None:
    """An answer as the scorer reads it: marker n names the nth citation."""
    if answer is None:
        return None
    return {
        "question": answer.question,
        "text": answer.text,
        "declined": answer.declined,
        "citations": [
            {
                "marker": marker,
                "chunk_id": citation.chunk_id,
                "arxiv_id": citation.arxiv_id,
                "quote": citation.quoted_text,
            }
            for marker, citation in enumerate(answer.citations, start=1)
        ],
        "rejected": [
            {"chunk_id": item.chunk_id, "quote": item.quoted_text, "reason": item.reason}
            for item in answer.rejected
        ],
    }


# Refusals about the account or the server, never about what an arm asked:
# sign-in (401), an exhausted cloud quota (402) and rate limits (429).
ACCOUNT_REFUSALS = (401, 402, 429)


def classify_error(error: BaseException) -> str:
    """Whether a failure belongs to the infrastructure or to the arm that raised it.

    The infrastructure's: connection failures, 5xx (Ollama's client retries them,
    then reports it cannot reach the server), a model that is not pulled, and
    the account refusals above. Whichever arm happens to be running when one
    occurs is not at fault, so the question is rerun for both.
    """
    code = getattr(error, "code", None)
    if isinstance(code, int) and (code in ACCOUNT_REFUSALS or code >= 500):
        return INFRASTRUCTURE
    if isinstance(error, (ConnectionError, TimeoutError)):
        return INFRASTRUCTURE
    message = str(error)
    if message.startswith("Cannot reach Ollama") or " has no model named " in message:
        return INFRASTRUCTURE
    if any(f"refused the request ({code}" in message for code in ACCOUNT_REFUSALS):
        return INFRASTRUCTURE
    return ARM


def record(
    *,
    set_name: str,
    case: str,
    arm: str,
    order: int,
    status: str,
    spent: dict[str, Any],
    model: str | None,
    digest: str | None,
    error: BaseException | None = None,
    **payload: Any,
) -> dict[str, Any]:
    """One arm's result for one case, in the shape every scorer reads."""
    return {
        "set": set_name,
        "case": case,
        "arm": arm,
        "order": order,
        "status": status,
        "error_kind": classify_error(error) if error is not None else None,
        "error": f"{type(error).__name__}: {error}" if error is not None else None,
        "model": model,
        "digest": digest,
        **spent,
        **payload,
    }
