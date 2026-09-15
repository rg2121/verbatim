"""Answering a question in parts, when one retrieval cannot reach all of it.

A single retrieval serves a single-subject question well. It serves a compound
one badly: "how do MOND and dark matter differ in galaxy clusters" is two
subjects, and the passages that best match the whole sentence tend to come from
whichever half the wording leans towards. Splitting the question, retrieving for
each part, and answering over the union puts both halves in front of the model.

This is a multi-step loop, not a swarm of agents. Each step is a retrieval, and
the answer is still written once by the existing verified-citation path, so a
claim still cannot appear unless its quoted words are in the passage it names.
It is off by default and reported beside the single-step path rather than
replacing it, because decomposition can also dilute a question that never
needed splitting: three mediocre retrievals in place of one good one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from verbatim.answers import Answer, Completion, answer_from_matches
from verbatim.corpus import ChunkMatch, CorpusStore
from verbatim.embedding import Embedder
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
    """Break a question into the parts that need answering separately.

    The original question is returned whenever planning fails, is unavailable,
    or produces nothing usable, so this can only add retrievals, never lose the
    question that was actually asked.
    """
    if max_parts < 1:
        raise ValueError("max_parts must be at least 1")
    try:
        reply = complete(PLAN_SYSTEM, question)
    except (RuntimeError, OSError):
        return [question]
    return parse_plan_lines(reply, min(max_parts, MAX_PARTS)) or [question]


def parse_plan_lines(reply: str, limit: int) -> list[str]:
    """Read a plan the model wrote one line at a time, tolerating decoration.

    Models prepend bullets and numbers however often the prompt says not to,
    so the markers are stripped rather than fought. Shared with the report
    planner, which asks a different question but reads the same shape of
    reply.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for line in reply.splitlines():
        cleaned = line.strip().lstrip("-*0123456789.) ").strip()
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
    """Retrieve for each part and keep every part's share of the passages.

    A quota per part rather than one fused ranking, because fusion lets the
    better-matching half of a compound question crowd out the other, which is
    the failure this exists to avoid.

    `search_options` are hybrid_search's own - scope, filters, the gated legs -
    and reach every part unchanged, so a question scoped to one paper stays
    scoped when it is split.
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


def research(
    corpus: CorpusStore,
    embedder: Embedder,
    completion: Completion,
    complete_text: Callable[[str, str], str],
    question: str,
    *,
    max_parts: int = 3,
    per_part: int = 4,
    max_per_paper: int | None = 2,
    **search_options: Any,
) -> tuple[Answer, list[str], list[ChunkMatch]]:
    """Answer a question by parts: the answer, the plan, and the passages used.

    The plan is returned rather than hidden, because a reader who disagrees
    with how a question was split needs to see the split to say so. The
    passages are returned because every check the single-step path runs over
    them - the subject warning above all - must run here too, and returning an
    answer without them would quietly disable the one abstention safeguard
    that measured any use.
    """
    parts = plan_subquestions(complete_text, question, max_parts=max_parts)
    matches = gather_for_parts(
        corpus, embedder, parts,
        per_part=per_part, max_per_paper=max_per_paper, **search_options,
    )
    return answer_from_matches(corpus, completion, question, matches), parts, matches
