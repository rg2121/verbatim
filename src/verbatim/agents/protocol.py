"""The flat line protocol agents reply in.

Native tool calling and structured JSON both failed on the local model, so an
agent replies in lines of the form `KEY: value`, as v1's ANSWER/CITE, TASK,
QUERY and GAP lines are. Anything else in a reply, such as the model's
reasoning, is ignored. A reply that cannot be read raises ProtocolError, whose
message goes back once with the reply (retry_message); a second failure ends
that step as a protocol failure. Whether a readable decision is legal is the
contract's to say, not the parser's.
"""

from __future__ import annotations

import re

from verbatim.agents.state import Abstain, Final, Inspect, Next
from verbatim.answers import CHUNK_ID_PATTERN

# A key in capitals at the start of a line, however the model dresses it:
# indented, after a bullet or a number, or in bold. One line per key.
KEY_LINE = re.compile(
    r"^[ \t]*(?:[-*>]+[ \t]*|\d+[.)][ \t]*)?\**([A-Z][A-Z_]*)\**[ \t]*:\**[ \t]*(.*?)[ \t]*$",
    re.MULTILINE)
STEP_SPLIT = re.compile(r"\s*(?:,|;|->|→|\bthen\b|\band\b)\s*|\s+")
LEAD_DECISIONS = ("NEXT", "INSPECT", "FINAL", "ABSTAIN")


class ProtocolError(ValueError):
    """A reply that cannot be read; the message says how to write it."""


def fields(reply: str) -> list[tuple[str, str]]:
    """Every `KEY: value` line in a reply, in order."""
    return [(match.group(1), match.group(2)) for match in KEY_LINE.finditer(reply or "")]


def retry_message(error: ProtocolError) -> str:
    """What goes back, once, with a reply that could not be read."""
    return f"Your reply could not be used: {error}. Reply again, in the format asked for."


def parse_lead(reply: str) -> Next | Inspect | Final | Abstain:
    """The Lead's one decision: NEXT (with BRIEF and PASSAGES), INSPECT, FINAL or ABSTAIN."""
    found = fields(reply)
    decisions = [(key, value) for key, value in found if key in LEAD_DECISIONS]
    if len(decisions) != 1:
        raise ProtocolError("reply with exactly one line starting NEXT:, INSPECT:, FINAL: or "
                            f"ABSTAIN: (this reply had {len(decisions)})")
    key, value = decisions[0]
    if key == "NEXT":
        steps = tuple(step for step in (part.strip(" .!:") for part in
                                        STEP_SPLIT.split(value.lower())) if step)
        if not steps:
            raise ProtocolError("NEXT: names the steps to take, in order, "
                                "e.g. NEXT: researcher, writer")
        return Next(steps=steps, brief=_first(found, "BRIEF"),
                    passages=_ids(found, "PASSAGES"))
    if key == "INSPECT":
        ids = tuple(CHUNK_ID_PATTERN.findall(value))
        if not ids:
            raise ProtocolError("INSPECT: takes passage ids from the evidence index")
        return Inspect(ids=ids)
    if key == "FINAL":
        number = re.search(r"\d+", value)
        if number is None:
            raise ProtocolError("FINAL: takes the number of the attempt to return, e.g. FINAL: 2")
        return Final(attempt=int(number.group()))
    if not value:
        raise ProtocolError("ABSTAIN: needs the reason the passages do not answer the question")
    return Abstain(reason=value)


def _first(found: list[tuple[str, str]], key: str) -> str:
    return next((value for name, value in found if name == key), "")


def _ids(found: list[tuple[str, str]], key: str) -> tuple[str, ...]:
    values = [value for name, value in found if name == key]
    ids = tuple(CHUNK_ID_PATTERN.findall(" ".join(values)))
    if values and not ids:
        raise ProtocolError(f"{key}: takes passage ids from the evidence index")
    return ids
