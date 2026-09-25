"""Rewriting a question before it reaches retrieval.

Retrieval here fuses BM25 and vector search, and both are given the user's
words exactly as typed. That is a weak point for two kinds of question: one
phrased differently from the literature ("how do you stop a model memorising
its training data" against papers that say "regularisation"), and one so short
that it carries little signal ("entropy?").

Two standard treatments, both kept behind flags and off until measured on this
corpus, which is the same discipline the reranker and the contextual embedding
prefix are held to. Retrieval here already puts the right passage first most
of the time, so a change that is not measured is as likely to cost as to gain.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import re

from verbatim.corpus import ChunkMatch

# The same constant the retrieval fusion uses, for the same reason: large
# enough that the top few ranks do not dominate.
RRF_SMOOTHING = 60
MAX_VARIANTS = 5
# What a model puts in front of a list item: a bullet, or a number followed by
# "." or ")" - bolded or not - and then a space. Only that. A line can itself
# begin with a number, and stripping every leading digit turned "4-bit
# quantization" into "bit quantization" and "3D" into "D".
LIST_MARKER = re.compile(r"^\s*(?:[-*•+]|\**\d+[.)]\**)\s+")

REWRITE_SYSTEM = (
    "You rewrite a search query for a corpus of arXiv papers. "
    "Reply with one rewrite per line and nothing else. "
    "Use the vocabulary a paper would use, not the vocabulary of a question."
)
HYDE_SYSTEM = (
    "You write the single sentence an arXiv paper's abstract would contain if "
    "it answered the question. Reply with that sentence and nothing else. "
    "Do not hedge, do not say you are uncertain, do not add commentary."
)


def rewrite_query(
    complete: Callable[[str, str], str], question: str, *, variants: int = 3
) -> list[str]:
    """Ask a model for alternative phrasings of a question.

    The original is always kept and always first: a rewrite can drop the term
    that actually mattered, and fusion should never be able to lose the words
    the user chose.
    """
    if variants < 1:
        raise ValueError("variants must be at least 1")
    try:
        reply = complete(REWRITE_SYSTEM, question)
    except (RuntimeError, OSError):
        # A rewrite is an optimisation; retrieval must still answer without it.
        return [question]
    return dedupe_queries([question, *_lines(reply)])[: variants + 1]


def hypothetical_answer(complete: Callable[[str, str], str], question: str) -> str:
    """Write the sentence a paper answering this question would contain.

    Embedding an answer rather than a question puts the query in the same
    register as the passages being searched, which is what HyDE is for. The
    question is returned unchanged if the model gives nothing usable, so this
    can only add signal.
    """
    try:
        reply = complete(HYDE_SYSTEM, question)
    except (RuntimeError, OSError):
        return question
    sentence = " ".join(reply.split())
    return sentence or question


def fuse_rankings(
    rankings: Sequence[Sequence[ChunkMatch]], *, smoothing: int = RRF_SMOOTHING
) -> list[ChunkMatch]:
    """Combine several rankings of the same corpus by position.

    Reciprocal Rank Fusion again, for the same reason as in hybrid search: the
    rankings come from different queries and their scores are not comparable,
    but their positions are.
    """
    fused: dict[str, float] = defaultdict(float)
    best: dict[str, ChunkMatch] = {}
    for ranking in rankings:
        for rank, match in enumerate(ranking, start=1):
            fused[match.chunk.chunk_id] += 1.0 / (smoothing + rank)
            best.setdefault(match.chunk.chunk_id, match)
    return [
        ChunkMatch(chunk=best[chunk_id].chunk, score=score, snippet=best[chunk_id].snippet)
        for chunk_id, score in sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    ]


def strip_list_marker(line: str) -> str:
    """One line of a model's list, without the bullet or number in front.

    Shared by every reader of one-item-per-line replies - rewrites, plans,
    report sections, the critic's subjects - so none of them can strip
    differently from the others.
    """
    return LIST_MARKER.sub("", line, count=1).strip()


def _lines(reply: str) -> list[str]:
    """Read one rewrite per line, dropping list markers and empty lines."""
    found = []
    for line in reply.splitlines():
        cleaned = strip_list_marker(line)
        if cleaned:
            found.append(cleaned)
    return found[:MAX_VARIANTS]


def dedupe_queries(queries: Sequence[str]) -> list[str]:
    """Drop repeated phrasings, keeping the first of each and its order."""
    seen: set[str] = set()
    unique = []
    for query in queries:
        key = " ".join(query.lower().split())
        if key and key not in seen:
            seen.add(key)
            unique.append(query)
    return unique
