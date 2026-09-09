"""Saying what a section leaves unsupported or unaddressed, checkably.

A critique step is the one role in the usual multi-agent design this project
had refused to build, and for a measured reason: asked whether retrieved
passages support an answer, every model tested says yes - 2/6, 1/6, 2/6, and a
13.8GB model at 0/6, worse than no check at all. Passages are retrieved
*because* they match, and a model reads that as evidence they suffice.

So this critic is built the way the subject check is, which is the one thing
that did work. It never judges. It asks the model only to *describe* - what
does this question ask about? - and then everything the model says is checked
against the passages mechanically before it counts. What survives is a fact
about the text, not an opinion about the answer:

- **unsupported**: sentences carrying no citation marker. Pure counting. The
  answering prompt has always instructed that every claim carry a marker, and
  nothing until now checked whether it did.
- **unaddressed**: subjects the question raises that appear in no retrieved
  passage. The model proposes the subjects; a substring check over the
  passages removes every one that is actually present, so a hallucinated gap
  cannot survive.
- **conflicts**: pairs of cited passages the model flags as disagreeing, kept
  only if both ids were really retrieved and really cited.

The recommended action is then derived by code from those three, never asked
for. That is the whole design: the model widens the search for problems, the
code decides which are real.
"""

from __future__ import annotations

import re

from verbatim.answers import Answer, Completion
from verbatim.contracts import Critique, Gathered

# A sentence ends at punctuation followed by a space and a capital, which is
# imperfect on "e.g." and exact enough to count markers by.
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
MARKER = re.compile(r"\[(?:\d+|unverified)\]")
SUBJECT_LINE = re.compile(r"^[-*\d.)\s]*(.+?)\s*$", re.M)
# A sentence too short to carry a claim - "In summary.", "This has two
# consequences." - is not evidence of anything missing. Five words keeps
# "It was published in 2014" and drops the connectives; the cost of going
# lower is a critic that fires on transitions, and a warning that frequent is
# one readers learn to skip.
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
    """Sentences making a claim with no citation marker attached.

    The prompt has always required a marker per claim; this is the first thing
    that checks. Deterministic, and the reason the critic can say something
    true even when no model is available.
    """
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

    The model is asked what the question is about - never whether the passages
    are adequate - and every subject it returns is then looked for in the
    passages. One that is present is dropped, so a subject invented or
    misjudged by the model cannot reach the caller.
    """
    try:
        reply = complete_text(SUBJECTS_SYSTEM, question)
    except (RuntimeError, OSError):
        return ()
    haystack = " ".join(match.chunk.text.lower() for match in gathered.matches)
    missing = []
    for line in SUBJECT_LINE.findall(reply or ""):
        subject = line.strip().strip(".").strip()
        if not subject or len(subject.split()) > 4:
            continue
        words = [w for w in re.findall(r"[a-z0-9']+", subject.lower()) if len(w) > 2]
        if not words:
            continue
        # Present if any content word of the subject occurs at all: the test is
        # deliberately generous, because the claim being made is absence.
        if not any(word in haystack for word in words):
            missing.append(subject)
    return tuple(dict.fromkeys(missing))[:5]


def conflicting_passages(
    complete_text, answer: Answer, gathered: Gathered
) -> tuple[tuple[str, str], ...]:
    """Pairs of cited passages a model flags as disagreeing, id-checked.

    A pair survives only if both ids were retrieved and both were cited, so
    the model can widen attention across the evidence but cannot invent a
    disagreement between passages that are not there.
    """
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

    Runs without a model: the uncited-sentence half is arithmetic and is the
    part that fires most. `complete_text` adds the two halves that need a
    model to propose candidates, both of which are checked before they count.
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
    # Derived, never asked for. A model that recommends its own next step is
    # judging again by another name.
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
