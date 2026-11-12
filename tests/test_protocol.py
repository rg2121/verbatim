"""M2a: the flat line protocol the Lead replies in, and how a bad reply is sent back."""

from __future__ import annotations

import pytest

from verbatim.agents.protocol import ProtocolError, fields, parse_lead, retry_message
from verbatim.agents.state import Abstain, Final, Inspect, Next

ID1, ID2 = "0123456789abcdef", "fedcba9876543210"


def test_key_lines_are_read_however_the_model_dresses_them() -> None:
    reply = (
        "Let me think. The evidence is thin, so: research first.\n"
        "  - **NEXT:** researcher, writer\n"
        "* BRIEF: find the training time\n"
        "Note: prose like this is not a key line\n"
        f"1. PASSAGES: {ID1}\n"
    )
    assert fields(reply) == [("NEXT", "researcher, writer"), ("BRIEF", "find the training time"),
                             ("PASSAGES", ID1)]


@pytest.mark.parametrize("steps", ["researcher, writer", "researcher then writer",
                                   "researcher -> writer", "Researcher and Writer"])
def test_the_lead_queues_steps_in_order(steps: str) -> None:
    assert parse_lead(f"NEXT: {steps}\nBRIEF: find the training time") == Next(
        steps=("researcher", "writer"), brief="find the training time")


def test_the_lead_can_give_the_writer_its_passages() -> None:
    assert parse_lead(f"NEXT: writer\nPASSAGES: {ID1}, {ID2}") == Next(
        steps=("writer",), passages=(ID1, ID2))


def test_the_lead_inspects_finishes_or_abstains() -> None:
    assert parse_lead(f"INSPECT: {ID1} and {ID2} (the Smith paper)") == Inspect(ids=(ID1, ID2))
    assert parse_lead("FINAL: attempt 2") == Final(attempt=2)
    assert parse_lead("ABSTAIN: the passages discuss another subject") == Abstain(
        reason="the passages discuss another subject")


@pytest.mark.parametrize(("reply", "says"), [
    ("I think we should write now.", "exactly one"),
    ("NEXT: writer\nFINAL: 1", "exactly one"),
    ("NEXT:", "names the steps"),
    ("INSPECT: the first passage", "passage ids"),
    ("FINAL: the best one", "number of the attempt"),
    ("ABSTAIN:", "reason"),
    ("NEXT: writer\nPASSAGES: the first three", "PASSAGES"),
])
def test_a_reply_that_cannot_be_read_says_how_to_fix_it(reply: str, says: str) -> None:
    with pytest.raises(ProtocolError, match=says):
        parse_lead(reply)


def test_a_retry_sends_back_what_was_wrong() -> None:
    error = ProtocolError("FINAL: takes the number of the attempt to return, e.g. FINAL: 2")
    message = retry_message(error)
    assert str(error) in message and "again" in message


# -- what the review of M2a found the parser misread ------------------------------------


@pytest.mark.parametrize("line", [
    "**NEXT: researcher, writer**", "NEXT: `researcher`, `writer`", 'NEXT: "researcher", "writer"',
    "NEXT: [researcher, writer]", "`NEXT: researcher, writer`", "NEXT: researcher, writer.",
])
def test_steps_are_read_as_words_however_they_are_dressed(line: str) -> None:
    assert parse_lead(f"{line}\nBRIEF: find the training time").steps == ("researcher", "writer")


def test_windows_line_endings_leave_nothing_behind() -> None:
    assert fields("BRIEF: find it\r\nNEXT: researcher\r\n") == [("BRIEF", "find it"),
                                                               ("NEXT", "researcher")]
    with pytest.raises(ProtocolError, match="reason"):
        parse_lead("ABSTAIN:\r\n")


@pytest.mark.parametrize("value", ["", "none", "None.", "default"])
def test_an_empty_passages_line_gives_the_writer_its_default_passages(value: str) -> None:
    assert parse_lead(f"NEXT: writer\nPASSAGES: {value}") == Next(steps=("writer",))


def test_passages_that_are_not_ids_say_the_line_can_be_left_out() -> None:
    with pytest.raises(ProtocolError, match="leave the line out"):
        parse_lead("NEXT: writer\nPASSAGES: the first three")


def test_bulleted_reasoning_does_not_count_as_the_decision() -> None:
    reply = ("Options:\n- NEXT: researcher would cost 2 calls\n- INSPECT: not needed yet\n"
             "NEXT: writer\n")
    assert parse_lead(reply) == Next(steps=("writer",))
    with pytest.raises(ProtocolError, match="exactly one"):
        parse_lead("NEXT: writer\nNEXT: researcher\nBRIEF: x")


def test_final_reads_the_attempt_it_names() -> None:
    assert parse_lead("FINAL: of the 3 attempts, attempt 2") == Final(attempt=2)
    assert parse_lead("FINAL: draft #3") == Final(attempt=3)
    with pytest.raises(ProtocolError, match="one attempt number"):
        parse_lead("FINAL: 2 of 3")


def test_an_id_cut_short_is_an_error_not_a_dropped_passage() -> None:
    with pytest.raises(ProtocolError, match="0123456789ab"):
        parse_lead(f"NEXT: writer\nPASSAGES: {ID1}, 0123456789ab")
    with pytest.raises(ProtocolError, match="whole"):
        parse_lead(f"INSPECT: {ID2}, fedcba98765")


def test_a_passage_named_twice_is_read_once() -> None:
    assert parse_lead(f"INSPECT: {ID1}, {ID1}, {ID2}, {ID2}, {ID1}") == Inspect(ids=(ID1, ID2))
    assert parse_lead(f"NEXT: writer\nPASSAGES: {ID1}, {ID1}").passages == (ID1,)
