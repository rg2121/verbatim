"""Repairing an answer the quote check flagged, with what the check found.

Asked whether its answer is right, a model says yes; told which quote is not in
which passage, and shown the closest words that are, it has something to act
on. Only what code found goes back, and only once. The revision is verified
like the draft, and replaces it only if it still cites every passage the draft
cited with a verified quote, adds no marker to a sentence its passage does not
share, and has fewer problems; otherwise the draft stands. A draft that cites
nothing is an abstention and is never sent back: pressing it to cite would press
it to answer what the passages do not cover.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import difflib
import re
from typing import Any

from verbatim.answers import (
    ANSWER_SCHEMA,
    CHUNK_ID_PATTERN,
    MARKER_PATTERN,
    SYSTEM_PROMPT,
    Answer,
    Completion,
    FormatError,
    RejectedCitation,
    answer_from_payload,
    build_prompt,
    verify_quote,
)
from verbatim.corpus import ChunkMatch
from verbatim.critique import unbacked_sentences, uncited_sentences
from verbatim.logs import event

UNVERIFIED = "[unverified]"
# A hint holds at least half the quote's words, in order: an unrelated sentence
# offered as a hint invites a quote that does not say the claim.
HINT_MIN_SHARE = 0.5
HINT_MAX_WORDS = 60
# Words kept either side of the match when a long sentence is cut.
HINT_CONTEXT = 5

REPAIR_RULES = """
A checker compared your draft with the passages and found the problems listed
after it. Fix only those problems:
- A quote that is not in its passage: copy the words exactly from the passage.
  Where the closest words are shown, use them only if they say what your
  sentence claims; otherwise cite another passage that says it, or remove the
  claim.
- A marker with no CITE line: add the CITE line, or remove the marker and its
  claim.
- A sentence with no marker: add a marker and a CITE line for a passage that
  says it, or remove the sentence.
Keep every other sentence, and every CITE line the checker did not mention,
exactly as it is. Add nothing new.
"""

NO_ANSWER_LINE = (
    'Your reply had no line starting with "ANSWER:", so nothing in it could be '
    "checked. Reply again in exactly the format."
)


@dataclass(frozen=True)
class Reflection:
    """A draft, what the check found in it, the revision asked for, and the answer kept.

    `draft` is None when the reply had no ANSWER line. `revision` is None when
    nothing was found, or when the revision could not be read.
    """

    draft: Answer | None
    findings: tuple[str, ...]
    revision: Answer | None
    answer: Answer


def problem_count(answer: Answer) -> int:
    """Rejected quotes, markers left unverified and unmarked claims; none in an abstention."""
    if not answer.citations and not answer.rejected and UNVERIFIED not in answer.text:
        return 0
    return (
        len(answer.rejected)
        + answer.text.count(UNVERIFIED)
        + len(uncited_sentences(answer.text))
    )


def passage_texts(matches: Sequence[ChunkMatch]) -> dict[str, str]:
    """Each passage's text by chunk id, whitespace normalised as the quote check reads it."""
    return {match.chunk.chunk_id: " ".join(match.chunk.text.split()) for match in matches}


def better(draft: Answer, revision: Answer, texts: dict[str, str]) -> Answer:
    """The revision, if it has fewer problems, every verified passage and no new unbacked sentence.

    A repair may not drop evidence that passed. An unmarked sentence at least
    looks unsupported, so a marker its passage does not back is no fix, however
    much else the repair put right. A tie keeps the draft.
    """
    cited = {citation.chunk_id for citation in revision.citations}
    if any(citation.chunk_id not in cited for citation in draft.citations):
        return draft
    if len(unbacked_sentences(revision, texts)) > len(unbacked_sentences(draft, texts)):
        return draft
    return revision if problem_count(revision) < problem_count(draft) else draft


def findings(
    payload: dict[str, Any], draft: Answer, matches: Sequence[ChunkMatch]
) -> list[str]:
    """What the check found in a draft, each as a fact the model can act on."""
    if problem_count(draft) == 0:
        return []
    texts = passage_texts(matches)
    markers = {
        (raw.get("chunk_id", ""), raw.get("quoted_text", "")): raw.get("marker")
        for raw in payload.get("citations", [])
    }
    found = [
        _rejected(item, markers.get((item.chunk_id, item.quoted_text)), texts)
        for item in draft.rejected
    ]
    found.extend(
        f"The answer uses [{marker}], but there is no CITE [{marker}] line."
        for marker in _dangling(payload)
    )
    found.extend(
        f'This sentence makes a claim but has no marker: "{sentence}"'
        for sentence in uncited_sentences(draft.text)
    )
    return found


def closest_words(quote: str, passage: str) -> str | None:
    """The sentence of `passage`, or pair of sentences, holding most of the quote's words in order.

    Returned exactly as it stands in the whitespace-normalised passage, so that,
    copied, it passes the quote check. None when under HINT_MIN_SHARE of the
    quote's words are there.
    """
    found = _closest(quote, passage)
    return None if found is None else found[1]


def reflect(
    completion: Completion,
    question: str,
    matches: Sequence[ChunkMatch],
    papers: dict[str, Any],
    payload: dict[str, Any] | None,
    reply: str = "",
) -> Reflection:
    """Verify a draft and, if the check finds problems, ask once for a repair.

    `payload` is None when the reply had no ANSWER line: the model is told so
    and asked once more, and a second such reply raises FormatError.
    """
    passages = list(matches)
    prompt = build_prompt(question, passages)
    if payload is None:
        retried = completion.revise(
            SYSTEM_PROMPT,
            f"{prompt}\n\nYour reply began:\n{reply[:600]}\n\n{NO_ANSWER_LINE}",
            ANSWER_SCHEMA,
        )
        answer = answer_from_payload(question, passages, retried, papers)
        event("answer.reflect", found=1, kept=True, problems_after=problem_count(answer))
        return Reflection(draft=None, findings=(NO_ANSWER_LINE,), revision=answer, answer=answer)

    draft = answer_from_payload(question, passages, payload, papers)
    found = tuple(findings(payload, draft, passages))
    if not found:
        return Reflection(draft=draft, findings=(), revision=None, answer=draft)
    listed = "\n".join(f"{number}. {item}" for number, item in enumerate(found, start=1))
    try:
        revised = completion.revise(
            SYSTEM_PROMPT + REPAIR_RULES,
            f"{prompt}\n\nYour draft:\n{_as_written(payload)}\n\n"
            f"What the checker found:\n{listed}",
            ANSWER_SCHEMA,
        )
    except FormatError:
        # An unreadable revision costs the call, not the draft.
        revision = None
    else:
        revision = answer_from_payload(question, passages, revised, papers)
    answer = draft if revision is None else better(draft, revision, passage_texts(passages))
    event(
        "answer.reflect", found=len(found), kept=answer is not draft,
        problems_before=problem_count(draft), problems_after=problem_count(answer),
    )
    return Reflection(draft=draft, findings=found, revision=revision, answer=answer)


def _as_written(payload: dict[str, Any]) -> str:
    """The draft in the line format the model wrote it in."""
    lines = [f"ANSWER: {payload.get('answer', '')}"]
    lines.extend(
        f"CITE [{raw.get('marker')}] {raw.get('chunk_id', '')} :: {raw.get('quoted_text', '')}"
        for raw in payload.get("citations", [])
    )
    return "\n".join(lines)


def _rejected(item: RejectedCitation, marker: object, texts: dict[str, str]) -> str:
    """One rejected citation, and where its words are, or the nearest words that are."""
    label = "A CITE line" if marker is None else f"CITE [{marker}]"
    if item.reason == "empty quote":
        return f"{label} quotes nothing."
    elsewhere = next(
        (chunk_id for chunk_id, text in texts.items()
         if chunk_id != item.chunk_id and verify_quote(item.quoted_text, text) is None),
        None,
    )
    if item.chunk_id not in texts:
        found = f"{label} names passage {item.chunk_id}, which is not one of the passages given."
        if elsewhere is not None:
            return f"{found} Its words are word for word in passage {elsewhere}."
        nearest = max(
            ((chunk_id, near) for chunk_id, text in texts.items()
             if (near := _closest(item.quoted_text, text)) is not None),
            key=lambda pair: pair[1][0],
            default=None,
        )
        if nearest is None:
            return f"{found} No passage has words close to its quote."
        return (
            f"{found} The closest words to its quote are in passage "
            f'{nearest[0]}: "{nearest[1][1]}"'
        )
    found = f'{label} quotes words that are not in passage {item.chunk_id}: "{item.quoted_text}".'
    if elsewhere is not None:
        return f"{found} They are word for word in passage {elsewhere}."
    words = closest_words(item.quoted_text, texts[item.chunk_id])
    if words is None:
        return f"{found} No sentence in that passage is close to them."
    return f'{found} The closest words in that passage are: "{words}"'


def _dangling(payload: dict[str, Any]) -> list[str]:
    """Markers in the answer with no CITE line, read as renumbering reads them."""
    citations = payload.get("citations", [])
    numbers = {raw.get("marker") for raw in citations}
    chunk_ids = {raw.get("chunk_id") for raw in citations}

    def is_marker(token: str) -> bool:
        return token.isdigit() or CHUNK_ID_PATTERN.fullmatch(token) is not None

    found: list[str] = []
    for match in MARKER_PATTERN.finditer(payload.get("answer", "")):
        parts = [part.strip() for part in match.group(1).split(",")]
        # "[see also, page 3]" is prose, not a group of markers.
        if not all(is_marker(part) for part in parts):
            continue
        found.extend(
            part for part in parts
            if (int(part) not in numbers if part.isdigit() else part not in chunk_ids)
        )
    return list(dict.fromkeys(found))


def _closest(quote: str, passage: str) -> tuple[tuple[float, int], str] | None:
    """The best span's (share of the quote's words, minus its length) and its words."""
    wanted = [word for word in map(_word, quote.split()) if word]
    tokens = passage.split()
    if not wanted or not tokens:
        return None
    words = [_word(token) for token in tokens]
    spans = _sentences(tokens)
    candidates = spans + [(start, end) for (start, _), (_, end) in zip(spans, spans[1:])]
    # The most of the quote's words, then the shortest span.
    rank, start, end = max(
        ((_shared(wanted, words[start:end]) / len(wanted), start - end), start, end)
        for start, end in candidates
    )
    if rank[0] < HINT_MIN_SHARE:
        return None
    if end - start > HINT_MAX_WORDS:
        start, end = _around_match(wanted, words, start, end)
    return rank, " ".join(tokens[start:end])


def _word(token: str) -> str:
    return re.sub(r"\W+", "", token.lower())


def _shared(wanted: list[str], words: list[str]) -> int:
    """How many of `wanted` appear in `words`, in order."""
    blocks = difflib.SequenceMatcher(None, wanted, words, autojunk=False).get_matching_blocks()
    return sum(block.size for block in blocks)


def _sentences(tokens: list[str]) -> list[tuple[int, int]]:
    """Token spans of a passage's sentences, cut after a token that ends one."""
    spans: list[tuple[int, int]] = []
    start = 0
    for index, token in enumerate(tokens):
        if token.rstrip("\"')]”’").endswith((".", "!", "?")):
            spans.append((start, index + 1))
            start = index + 1
    if start < len(tokens):
        spans.append((start, len(tokens)))
    return spans


def _around_match(wanted: list[str], words: list[str], start: int, end: int) -> tuple[int, int]:
    """A long span cut to its matched words, a few either side, at most HINT_MAX_WORDS."""
    blocks = [
        block for block in difflib.SequenceMatcher(
            None, wanted, words[start:end], autojunk=False
        ).get_matching_blocks() if block.size
    ]
    low = max(start, start + blocks[0].b - HINT_CONTEXT)
    high = min(end, start + blocks[-1].b + blocks[-1].size + HINT_CONTEXT, low + HINT_MAX_WORDS)
    return low, high
