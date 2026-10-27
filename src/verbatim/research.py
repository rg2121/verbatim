"""Answering a compound question in parts, behind `ask --plan`.

A model splits the question, each part is retrieved separately, and one answer
is written over the union through the verified-citation path. Off by default:
splitting can also dilute a question that never needed it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from verbatim.answers import Answer, Completion, answer_from_matches
from verbatim.corpus import ChunkMatch, CorpusStore
from verbatim.embedding import Embedder
from verbatim.expansion import strip_list_marker
from verbatim.retrieval import hybrid_search

MAX_PARTS = 4

PLAN_SYSTEM = (
    "You split a research question into the smallest number of independent "
    "sub-questions needed to answer it, one per line and nothing else. "
    "A question about a single subject needs no splitting: reply with it "
    "unchanged. Never write more than four lines."
)


def plan_subquestions(
    complete: Callable[[str, str], str], question: str, *, max_parts: int = 3
) -> list[str]:
    """Split a question into parts, or return it unchanged if planning fails."""
    if max_parts < 1:
        raise ValueError("max_parts must be at least 1")
    try:
        reply = complete(PLAN_SYSTEM, question)
    except (RuntimeError, OSError):
        return [question]
    return parse_plan_lines(reply, min(max_parts, MAX_PARTS)) or [question]


def parse_plan_lines(reply: str, limit: int) -> list[str]:
    """Read a one-per-line plan, dropping list markers and repeats; shared with the report planner."""
    lines: list[str] = []
    seen: set[str] = set()
    for line in reply.splitlines():
        cleaned = strip_list_marker(line)
        key = " ".join(cleaned.lower().split())
        if cleaned and key not in seen:
            seen.add(key)
            lines.append(cleaned)
    return lines[:limit]


def gather_for_parts(
    corpus: CorpusStore,
    embedder: Embedder,
    parts: list[str],
    *,
    per_part: int = 4,
    max_per_paper: int | None = 2,
    **search_options: Any,
) -> list[ChunkMatch]:
    """Retrieve for each part with its own quota, so one part cannot crowd out another.

    `search_options` are passed to hybrid_search for every part.
    """
    collected: list[ChunkMatch] = []
    seen: set[str] = set()
    for part in parts:
        for match in hybrid_search(
            corpus, embedder, part, limit=per_part, max_per_paper=max_per_paper,
            **search_options,
        ):
            if match.chunk.chunk_id in seen:
                continue
            seen.add(match.chunk.chunk_id)
            collected.append(match)
    return collected


def gather_within_budget(
    corpus: CorpusStore,
    embedder: Embedder,
    parts: list[str],
    *,
    passages: int,
    max_per_paper: int | None = 2,
    **search_options: Any,
) -> list[ChunkMatch]:
    """Retrieve for every part from one shared budget of `passages`.

    Each part gets an even share, at least two, and the union is cut to the
    budget. `ask --plan` and the evaluation's hybrid+plan mode both call this,
    so what ships is what was measured.
    """
    return gather_for_parts(
        corpus, embedder, parts,
        per_part=max(passages // max(len(parts), 1), 2),
        max_per_paper=max_per_paper, **search_options,
    )[:passages]


def research(
    corpus: CorpusStore,
    embedder: Embedder,
    completion: Completion,
    complete_text: Callable[[str, str], str],
    question: str,
    *,
    max_parts: int = 3,
    passages: int = 8,
    max_per_paper: int | None = 2,
    **search_options: Any,
) -> tuple[Answer, list[str], list[ChunkMatch]]:
    """Answer a question by parts, returning the answer, the plan and the passages.

    The plan is shown to the reader; the passages let the caller run the same
    checks as the single-step path.
    """
    parts = plan_subquestions(complete_text, question, max_parts=max_parts)
    # The budget is shared by the parts actually planned, so a question left
    # whole gets every passage, as plain `ask` does.
    matches = gather_within_budget(
        corpus, embedder, parts, passages=passages,
        max_per_paper=max_per_paper, **search_options,
    )
    return answer_from_matches(corpus, completion, question, matches), parts, matches
