"""Query rewriting before retrieval: model rephrasings fused by rank, and HyDE.

Both are opt-in, for questions phrased unlike the literature or too short to
carry much signal.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import re

from verbatim.corpus import ChunkMatch

# Reciprocal Rank Fusion constant: the top few ranks do not dominate, and deep
# results still fade out.
RRF_SMOOTHING = 60
MAX_VARIANTS = 5
# A bullet or list number (possibly bold) and a space. Not every leading digit:
# "4-bit quantization" must keep its 4.
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
    """Alternative phrasings from a model, with the original always kept first."""
    if variants < 1:
        raise ValueError("variants must be at least 1")
    try:
        reply = complete(REWRITE_SYSTEM, question)
    except (RuntimeError, OSError):
        # A rewrite is an optimisation; retrieval must still answer without it.
        return [question]
    return dedupe_queries([question, *_lines(reply)])[: variants + 1]


def hypothetical_answer(complete: Callable[[str, str], str], question: str) -> str:
    """The sentence a paper answering this would contain (HyDE), or the question on failure."""
    try:
        reply = complete(HYDE_SYSTEM, question)
    except (RuntimeError, OSError):
        return question
    sentence = " ".join(reply.split())
    return sentence or question


def fuse_rankings(
    rankings: Sequence[Sequence[ChunkMatch]], *, smoothing: int = RRF_SMOOTHING
) -> list[ChunkMatch]:
    """Combine rankings from different queries by position (Reciprocal Rank Fusion)."""
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
    """One line of a model's list without its bullet or number; shared by every list reader."""
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
