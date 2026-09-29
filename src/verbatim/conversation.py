"""Follow-up questions: the conversation decides what is looked up, never what is said.

A follow-up such as "why does it clip?" is rewritten into a standalone question
before retrieval. The answering model still sees only passages, because text
from earlier turns cannot be checked against a source.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re

# Enough history to keep the subject; short enough for a fast rewrite.
MAX_TURNS = 6
MAX_TURN_CHARS = 400

# Only a question that may point back is rewritten; a complete question can be
# narrowed by a rewrite. The test is generous, since an unneeded rewrite is
# shown to the reader and a missed one is not.
BACK_REFERENCE = re.compile(
    r"\b(it|its|it's|they|them|their|theirs|this|that|these|those|he|him|his|she|her)\b"
    r"|\bthe\s+(paper|study|work|method|approach|model|system|authors?|result|results|"
    r"dataset|experiment|finding|findings)\b"
    r"|\b(same|above|previous|earlier|latter|former)\b",
    re.IGNORECASE,
)
# Shorter questions, such as "and the cost?", cannot carry their own subject.
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

    Returns the question unchanged when there is no history, it already stands
    alone, or the model fails or replies with nothing usable.
    """
    if not history or not refers_back(question):
        return question
    recent = list(history)[-MAX_TURNS:]
    transcript = "\n\n".join(turn.render() for turn in recent)
    try:
        reply = complete(REWRITE_SYSTEM, f"{transcript}\n\nQ: {question.strip()}")
    except (RuntimeError, OSError):
        return question
    # The first non-empty line: anything after it is the model explaining itself.
    for line in reply.splitlines():
        candidate = " ".join(line.split()).strip().strip('"')
        if candidate:
            return candidate
    return question
