"""Carrying a conversation without letting it become evidence.

A follow-up question is rarely self-contained. "Why does it clip?" means
nothing on its own, and retrieval given those four words returns nothing
useful, so a single-turn assistant forces the reader to restate the subject
every time.

The obvious fix - hand the conversation to the model that writes the answer -
would break the guarantee this project exists for. That model may only state
what a retrieved passage supports, because every claim it makes is checked
against the passage it cites. Text from earlier turns cannot be checked
against anything: it is the model's own prose, and letting it back in would
mean a claim could enter an answer without ever having come from a paper.

So the conversation is used for exactly one thing: turning a follow-up into a
question that can stand alone, before retrieval sees it. The answering path is
untouched and still sees only passages. History decides what is looked up; it
never decides what may be said.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re

# Enough for the subject to survive several turns, short enough that a local
# model rewrites in about a second rather than re-reading an essay.
MAX_TURNS = 6
MAX_TURN_CHARS = 400

# A question that already stands on its own needs no rewrite. The call costs a
# second, and a model asked to rewrite a complete question can narrow it or
# drop the term that mattered - so the history is only consulted for a
# question that actually points back at something. The test is deliberately
# generous: anything that might refer back is rewritten, because an
# unnecessary rewrite is shown to the reader and a missed one is not.
BACK_REFERENCE = re.compile(
    r"\b(it|its|it's|they|them|their|theirs|this|that|these|those|he|him|his|she|her)\b"
    r"|\bthe\s+(paper|study|work|method|approach|model|system|authors?|result|results|"
    r"dataset|experiment|finding|findings)\b"
    r"|\b(same|above|previous|earlier|latter|former)\b",
    re.IGNORECASE,
)
# Fewer words than this and a question cannot carry its own subject: "why?",
# "and the cost?" - both need what came before.
MIN_STANDALONE_WORDS = 5

REWRITE_SYSTEM = (
    "Rewrite the user's latest question so it can be understood on its own, "
    "using the conversation only to resolve what it refers to. "
    "Reply with the rewritten question and nothing else. "
    "Keep the wording and the intent; replace pronouns and references with "
    "the thing they name. If the question already stands alone, repeat it "
    "unchanged."
)


@dataclass(frozen=True)
class Turn:
    """One exchange: what was asked, and the answer that was given."""

    question: str
    answer: str = ""

    def render(self) -> str:
        answer = self.answer.strip()[:MAX_TURN_CHARS]
        return f"Q: {self.question.strip()}\nA: {answer}" if answer else f"Q: {self.question.strip()}"


def refers_back(question: str) -> bool:
    """Whether this question needs the conversation to be understood."""
    if len(question.split()) < MIN_STANDALONE_WORDS:
        return True
    return BACK_REFERENCE.search(question) is not None


def standalone_question(
    complete: Callable[[str, str], str],
    history: Sequence[Turn],
    question: str,
) -> str:
    """Resolve a follow-up against the conversation, for retrieval only.

    Returns the original question whenever there is no history, the question
    already stands on its own, the model is unreachable, or it replies with
    nothing usable. A rewrite is an improvement to retrieval, and an
    improvement must not be able to stop a question being asked at all.
    """
    if not history or not refers_back(question):
        return question
    recent = list(history)[-MAX_TURNS:]
    transcript = "\n\n".join(turn.render() for turn in recent)
    try:
        reply = complete(REWRITE_SYSTEM, f"{transcript}\n\nQ: {question.strip()}")
    except (RuntimeError, OSError):
        return question
    # Take the first non-empty line before collapsing whitespace: a model that
    # starts explaining itself has stopped rewriting, and the commentary comes
    # after a newline. Collapsing first would splice it onto the question.
    for line in reply.splitlines():
        candidate = " ".join(line.split()).strip().strip('"')
        if candidate:
            return candidate
    return question
