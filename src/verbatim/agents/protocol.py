"""The flat line protocol agents reply in.

Native tool calling and structured JSON both failed on the local model, so an
agent replies in lines of the form `KEY: value`, as v1's ANSWER/CITE, TASK,
QUERY and GAP lines are. Anything else in a reply, such as the model's
reasoning, is ignored, and so is a bulleted key line beside a bare one: a model
weighing options lists them. A reply that cannot be read raises ProtocolError,
whose message goes back once with the reply (retry_message); a second failure
ends that step as a protocol failure. Whether a readable decision is legal is
the contract's to say, not the parser's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import TypeVar

from verbatim.agents.state import Abstain, CitationDecision, Final, Inspect, Next, Verification
from verbatim.answers import CHUNK_ID_PATTERN

T = TypeVar("T")

# A key in capitals at the start of a line, however the model dresses it:
# indented, after a bullet or a number, in bold, quotes or backticks. One line per key.
KEY_LINE = re.compile(
    r"^[ \t]*(?P<bullet>(?:[-*>]|\d+[.)])[ \t]+)?[*`\"']*(?P<key>[A-Z][A-Z_]*)[*`\"']*[ \t]*:"
    r"(?P<value>.*)$", re.MULTILINE)
DRESSING = " \t*`\"'"
STEP_WORD = re.compile(r"[a-z]+")
FILLER = frozenset({"then", "and"})
HEX_RUN = re.compile(r"\b[0-9a-f]{8,}\b")  # anything that looks like a passage id, whole or cut
NAMED_ATTEMPT = re.compile(r"(?:attempt|draft|#)\s*#?\s*(\d+)", re.IGNORECASE)
NO_PASSAGES = frozenset({"", "none", "default", "n/a", "-"})
LEAD_DECISIONS = ("NEXT", "INSPECT", "FINAL", "ABSTAIN")
# An ACTION line however it is written: any case, plural, with or without the colon.
ACTION_LINE = re.compile(
    r"^[ \t]*(?:(?:[-*>]|\d+[.)])[ \t]+)?[*`\"']*actions?\b[*`\"']*[ \t]*:?(?P<value>.*)$",
    re.IGNORECASE | re.MULTILINE)
# In one decision: a bracketed marker anywhere, or a bare number leading it ("2) drop").
ACTION_MARKER = re.compile(r"\[\s*(\d+)\s*\]|^[^\w\[]*(\d+)(?![\w.])")
ACTION_DECISION = re.compile(r"\b(?:keep[\W_]*(?:as[\W_]*)?unverified|keep|repair|drop)\b",
                             re.IGNORECASE)
VERIFIER_NEXT = {"writer": True, "repair": True, "done": False, "accept": False}
NO_DECISIONS = frozenset({"", "none", "nothing", "n/a", "-"})
RETRY_SHOWN = 1500  # characters of an unusable reply shown back with what was wrong


class ProtocolError(ValueError):
    """A reply that cannot be read; the message says how to write it."""


@dataclass(frozen=True)
class Line:
    key: str
    value: str
    bulleted: bool


def lines(reply: str) -> list[Line]:
    """Every `KEY: value` line in a reply, in order, its value undressed."""
    text = (reply or "").replace("\r\n", "\n").replace("\r", "\n")
    return [Line(match["key"], match["value"].strip(DRESSING), match["bullet"] is not None)
            for match in KEY_LINE.finditer(text)]


def fields(reply: str) -> list[tuple[str, str]]:
    """Every `KEY: value` line in a reply, as pairs."""
    return [(line.key, line.value) for line in lines(reply)]


def retry_message(error: ProtocolError) -> str:
    """What goes back, once, with a reply that could not be read."""
    return f"Your reply could not be used: {error}. Reply again, in the format asked for."


def ask_twice(ask: Callable[[str, str], str], system: str, user: str,
              read: Callable[[str], T]) -> tuple[T | None, tuple[str, ...]]:
    """One decision from a model. `read` turns a reply into it, raising
    ProtocolError on one it cannot use, an illegal one included; that reply goes
    back once with what was wrong. None after two, with every reply."""
    replies: list[str] = []
    prompt = user
    for _ in range(2):
        reply = ask(system, prompt)
        replies.append(reply)
        try:
            return read(reply), tuple(replies)
        except ProtocolError as error:
            prompt = (f"{user}\n\nYour reply was:\n{reply[:RETRY_SHOWN]}\n\n"
                      f"{retry_message(error)}")
    return None, tuple(replies)


def parse_lead(reply: str) -> Next | Inspect | Final | Abstain:
    """The Lead's one decision: NEXT (with BRIEF and PASSAGES), INSPECT, FINAL or ABSTAIN."""
    found = lines(reply)
    chosen = _decision(found, LEAD_DECISIONS, "reply with exactly one line starting NEXT:, "
                                              "INSPECT:, FINAL: or ABSTAIN:")
    key, value = chosen.key, chosen.value
    if key == "NEXT":
        steps = tuple(word for word in STEP_WORD.findall(value.lower()) if word not in FILLER)
        if not steps:
            raise ProtocolError("NEXT: names the steps to take, in order, "
                                "e.g. NEXT: researcher, writer")
        return Next(steps=steps, brief=" ".join(_values(found, "BRIEF")[:1]),
                    passages=_writer_passages(_values(found, "PASSAGES")))
    if key == "INSPECT":
        ids = passage_ids(value)
        if not ids:
            raise ProtocolError("INSPECT: takes passage ids from the evidence index")
        return Inspect(ids=ids)
    if key == "FINAL":
        return Final(attempt=_attempt(value))
    if not value:
        raise ProtocolError("ABSTAIN: needs the reason the passages do not answer the question")
    return Abstain(reason=value)


def parse_verification(reply: str) -> Verification:
    """The Verifier's decisions: an ACTION line per citation it decides on, and
    NEXT: writer, to send the draft back, or NEXT: done; RESEARCH asks the Lead
    for more evidence."""
    found = lines(reply)
    value = _decision(found, ("NEXT",), "reply with exactly one line NEXT: writer, to send "
                                        "the draft back, or NEXT: done").value
    # The first of writer or done decides: "done, no need to send it to the writer" is done.
    said = [word for word in STEP_WORD.findall(value.lower()) if word in VERIFIER_NEXT]
    if not said:
        raise ProtocolError("NEXT: takes writer or done")
    repair = VERIFIER_NEXT[said[0]]
    decisions: dict[int, CitationDecision] = {}
    text = (reply or "").replace("\r\n", "\n").replace("\r", "\n")
    for line in ACTION_LINE.finditer(text):
        for decided in _actions(line["value"]):
            if decided.marker in decisions:
                raise ProtocolError(f"two decisions for [{decided.marker}]: give each citation one")
            decisions[decided.marker] = decided
    return Verification(decisions=tuple(decisions.values()), repair=repair,
                        research=" ".join(_values(found, "RESEARCH")[:1]))


def passage_ids(value: str) -> tuple[str, ...]:
    """The passage ids in a value, each once, in order. A run of hex digits too
    short to be an id is one cut short: an error, never a passage dropped."""
    ids: list[str] = []
    for run in HEX_RUN.findall(value):
        if not CHUNK_ID_PATTERN.fullmatch(run):
            raise ProtocolError(f"passage id {run} is cut short; copy ids whole from the "
                                "evidence index")
        if run not in ids:
            ids.append(run)
    return tuple(ids)


def _actions(value: str) -> list[CitationDecision]:
    """The decisions on one ACTION line, separated by commas or semicolons. A part
    with neither a marker nor a decision is prose and is skipped; one with only
    one of them, or with more than one of either, cannot be read whole; and a line
    must decide something, unless it says none."""
    decided = []
    for part in re.split(r"[,;]", value):
        markers = [int(bracketed or bare) for bracketed, bare in ACTION_MARKER.findall(part)]
        words = ACTION_DECISION.findall(part)
        if not markers and not words:
            continue
        if len(markers) != 1 or len(words) != 1:
            raise ProtocolError("ACTION: takes one citation's marker and one decision, e.g. "
                                "ACTION: [2] repair; the decisions are keep, repair, drop and "
                                "keep_unverified")
        word = words[0].lower()
        decided.append(CitationDecision(markers[0], "keep_unverified" if "unverified" in word else word))
    if not decided and value.strip(DRESSING + ".:").lower() not in NO_DECISIONS:
        raise ProtocolError("ACTION: takes one citation's marker and one decision, e.g. ACTION: "
                            "[2] repair, or ACTION: none")
    return decided


def _decision(found: list[Line], keys: tuple[str, ...], error: str) -> Line:
    """The one decision line among `keys`: a bare line over bulleted ones, which
    a model weighing options writes."""
    decisions = [line for line in found if line.key in keys]
    bare = [line for line in decisions if not line.bulleted]
    chosen = bare if len(bare) == 1 else decisions
    if len(chosen) != 1:
        raise ProtocolError(f"{error} (this reply had {len(decisions)})")
    return chosen[0]


def _values(found: list[Line], key: str) -> list[str]:
    """A key's values: the bare lines' if any, else the bulleted lines'."""
    bare = [line.value for line in found if line.key == key and not line.bulleted]
    return bare or [line.value for line in found if line.key == key]


def _writer_passages(values: list[str]) -> tuple[str, ...]:
    ids = passage_ids(" ".join(values))
    if ids or all(value.strip(DRESSING + ".").lower() in NO_PASSAGES for value in values):
        return ids
    raise ProtocolError("PASSAGES: takes passage ids from the evidence index; "
                        "leave the line out to give the writer the default passages")


def _attempt(value: str) -> int:
    named = NAMED_ATTEMPT.search(value)
    if named is not None:
        return int(named.group(1))
    numbers = re.findall(r"\d+", value)
    if not numbers:
        raise ProtocolError("FINAL: takes the number of the attempt to return, e.g. FINAL: 2")
    if len(numbers) > 1:
        raise ProtocolError("FINAL: takes one attempt number; write it as FINAL: 2")
    return int(numbers[0])
