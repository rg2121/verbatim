"""What a written section leaves unsupported or unaddressed, checked by code.

Models asked whether passages support an answer say yes, so this critic never
asks for a judgement. It counts sentences with no citation marker, and asks
the model only to list the question's subjects and to flag conflicting cited
passages; each suggestion is checked against the passages before it counts.
The recommended action is derived by code.
"""

from __future__ import annotations

import re

from verbatim.answers import Answer
from verbatim.contracts import Critique, Gathered
from verbatim.expansion import strip_list_marker

# Sentence boundaries: imperfect on "e.g.", good enough to count markers by.
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
MARKER = re.compile(r"\[(?:\d+|unverified)\]")
# Shorter sentences, such as "In summary.", are connectives rather than claims.
MIN_CLAIM_WORDS = 5

SUBJECTS_SYSTEM = (
    "You list the distinct subjects a research question asks about, one per "
    "line and nothing else. A subject is a noun phrase of one to four words "
    "that a paper could be about. Do not answer the question, do not judge "
    "anything, and do not add commentary. Write at most five lines."
)
CONFLICT_SYSTEM = (
    "You are given passages that were cited together. Name only pairs that "
    "state different values, findings or conclusions about the same thing. "
    "Reply with one pair per line as 'id_a || id_b' using the passage ids "
    "given, and nothing else. If no pair disagrees, reply with the single "
    "word NONE."
)


def uncited_sentences(text: str) -> tuple[str, ...]:
    """Sentences making a claim with no citation marker. Needs no model."""
    uncited = []
    for sentence in SENTENCE.split(text.strip()):
        cleaned = sentence.strip()
        if len(cleaned.split()) < MIN_CLAIM_WORDS:
            continue
        if not MARKER.search(cleaned):
            uncited.append(cleaned)
    return tuple(uncited)


def cited_sentences(text: str) -> tuple[str, ...]:
    return tuple(
        sentence.strip()
        for sentence in SENTENCE.split(text.strip())
        if MARKER.search(sentence) and "[unverified]" not in sentence
    )


def unaddressed_subjects(
    complete_text, question: str, gathered: Gathered
) -> tuple[str, ...]:
    """Subjects the question raises that no retrieved passage mentions.

    The model lists the question's subjects; any found in the passages is
    dropped, so an invented gap cannot reach the caller.
    """
    try:
        reply = complete_text(SUBJECTS_SYSTEM, question)
    except (RuntimeError, OSError):
        return ()
    haystack = " ".join(match.chunk.text.lower() for match in gathered.matches)
    missing = []
    for line in (reply or "").splitlines():
        subject = strip_list_marker(line).strip(".").strip()
        if not subject or len(subject.split()) > 4:
            continue
        words = [w for w in re.findall(r"[a-z0-9']+", subject.lower()) if len(w) > 2]
        if not words:
            continue
        # Generous on purpose: one content word present counts as covered.
        if not any(word in haystack for word in words):
            missing.append(subject)
    return tuple(dict.fromkeys(missing))[:5]


def conflicting_passages(
    complete_text, answer: Answer, gathered: Gathered
) -> tuple[tuple[str, str], ...]:
    """Pairs of cited passages a model flags as disagreeing, kept only if both were cited."""
    cited = [citation.chunk_id for citation in answer.citations]
    if len(cited) < 2:
        return ()
    by_id = {match.chunk.chunk_id: match for match in gathered.matches}
    passages = "\n\n".join(
        f"[{chunk_id}] {by_id[chunk_id].chunk.text[:400]}"
        for chunk_id in cited if chunk_id in by_id
    )
    try:
        reply = complete_text(CONFLICT_SYSTEM, passages)
    except (RuntimeError, OSError):
        return ()
    pairs = []
    for line in (reply or "").splitlines():
        if "||" not in line:
            continue
        left, _, right = line.partition("||")
        one, two = left.strip().strip("[]"), right.strip().strip("[]")
        if one in by_id and two in by_id and one in cited and two in cited and one != two:
            pairs.append((one, two))
    return tuple(dict.fromkeys(pairs))[:3]


def critique(
    question: str,
    answer: Answer,
    gathered: Gathered,
    *,
    complete_text=None,
) -> Critique:
    """What is unsupported or unaddressed here, and what to do about it.

    Without `complete_text` only the uncited-sentence count runs.
    """
    unsupported = uncited_sentences(answer.text) if not answer.declined else ()
    unaddressed = (
        unaddressed_subjects(complete_text, question, gathered)
        if complete_text is not None and gathered.matches and not answer.declined
        else ()
    )
    conflicts = (
        conflicting_passages(complete_text, answer, gathered)
        if complete_text is not None and not answer.declined
        else ()
    )
    # Derived by code, never asked of a model.
    if unaddressed:
        action = "retrieve again: the question raises subjects no passage covers"
    elif unsupported:
        action = "rewrite: sentences carry claims with no citation"
    elif conflicts:
        action = "report the disagreement rather than choosing a side"
    else:
        action = "accept"
    return Critique(
        supported=cited_sentences(answer.text),
        unsupported=unsupported,
        unaddressed=unaddressed,
        conflicts=conflicts,
        action=action,
    )
