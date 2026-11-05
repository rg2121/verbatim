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
from verbatim.quotations import mask_quoted_brackets

# Sentence boundaries: imperfect on "e.g.", good enough to count markers by.
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
MARKER = re.compile(r"\[(?:\d+|unverified)\]")
CITED = re.compile(r"\[(\d+)\]")
# Shorter sentences, such as "In summary.", are connectives rather than claims.
MIN_CLAIM_WORDS = 5
# Under this share of its content words in the passages it cites, a marked
# sentence is not resting on them. Read one by one, a higher floor mostly catches
# sound sentences; first drafts fall under this one for 1% of what they mark.
MIN_BACKED_SHARE = 0.5
CONTENT_WORD = re.compile(r"[a-z0-9]+")
# Words any sentence has, and those an answer uses to talk about its passages.
COMMON_WORDS = frozenset(
    """a an the of in on at to for from by with and or but not no is are was were be
    been being this that these those it its as such which who whom whose where when
    how what why can could may might must shall should will would do does did has have
    had than then there here their they them we our you your he she his her also both
    each more most some any all into over under between through about against during
    within without per via using used use based however furthermore specifically
    additionally therefore thus while passages passage state states stated indicate
    indicates describe describes described noted note provided show shows shown one
    two""".split()
)

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


def sentences(text: str) -> list[tuple[str, str]]:
    """Each sentence as written, and as its markers read.

    A bracket inside a quotation is the quoted text's own, such as a paper's
    reference number, so the second copy has it masked. Both split alike.
    """
    written = text.strip()
    read = mask_quoted_brackets(written)
    pieces, start = [], 0
    for boundary in SENTENCE.finditer(read):
        pieces.append((written[start:boundary.start()], read[start:boundary.start()]))
        start = boundary.end()
    pieces.append((written[start:], read[start:]))
    return pieces


def uncited_sentences(text: str) -> tuple[str, ...]:
    """Sentences making a claim with no citation marker. Needs no model."""
    uncited = []
    for sentence, read in sentences(text):
        cleaned = sentence.strip()
        if len(cleaned.split()) < MIN_CLAIM_WORDS:
            continue
        if not MARKER.search(read):
            uncited.append(cleaned)
    return tuple(uncited)


def _cited(text: str) -> list[tuple[str, str]]:
    return [
        (sentence.strip(), read.strip()) for sentence, read in sentences(text)
        if MARKER.search(read) and "[unverified]" not in read
    ]


def cited_sentences(text: str) -> tuple[str, ...]:
    return tuple(sentence for sentence, _read in _cited(text))


def unbacked_sentences(
    answer: Answer, texts: dict[str, str], floor: float = MIN_BACKED_SHARE
) -> tuple[str, ...]:
    """Marked sentences with under `floor` of their content words in the passages they cite.

    Needs no model. The quote check proves a quote's words are in its passage, not
    that they say what the sentence says. Shared words cannot prove that either,
    but too few of them show that the sentence is not resting on the passage.
    """
    unbacked = []
    for sentence, read in _cited(answer.text):
        plain = re.sub(rf"\s*{MARKER.pattern}", "", read)
        words = _content_words(plain)
        if len(plain.split()) < MIN_CLAIM_WORDS or not words:
            continue
        cited = " ".join(
            texts.get(answer.citations[int(number) - 1].chunk_id, "")
            for number in CITED.findall(read)
            if 0 < int(number) <= len(answer.citations)
        )
        if len(words & _content_words(cited)) / len(words) < floor:
            unbacked.append(sentence)
    return tuple(unbacked)


def _content_words(text: str) -> set[str]:
    return {
        word for word in CONTENT_WORD.findall(text.lower())
        if word not in COMMON_WORDS and len(word) > 1
    }


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
