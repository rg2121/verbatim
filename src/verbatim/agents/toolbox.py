"""The tools agents call, all reading through ScopedCorpus (I7, I11).

They are v1's tools (tools.TOOL_SCHEMAS, tools.call_tool), read-only, with one
difference: search runs through the scope, since v1's search_chunks hands the
store itself to hybrid_search. INSPECT reads passages' exact text for the
Lead, bounded so a look stays cheap.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verbatim import tools
from verbatim.agents import contract
from verbatim.agents.scoped import ScopedCorpus
from verbatim.retrieval import default_search_options

INSPECT_CHARACTERS = 6000
CUT = " [cut]"


def call(name: str, arguments: dict[str, Any], corpus: ScopedCorpus, embedder: Any) -> Any:
    """Run one of v1's agent tools on what this run may read."""
    if name == "search_chunks":
        return search_chunks(corpus, embedder, str(arguments.get("query", "")),
                             limit=int(arguments.get("limit", 8)),
                             arxiv_id=arguments.get("arxiv_id"))
    return tools.call_tool(name, arguments, corpus, embedder)  # type: ignore[arg-type]


def search_chunks(corpus: ScopedCorpus, embedder: Any, query: str, *, limit: int = 8,
                  arxiv_id: str | None = None) -> list[dict[str, Any]]:
    """tools.search_chunks, searching through the scope; test_scoped holds the two equal."""
    if not query.strip():
        raise ValueError("query is empty: pass the words to search for")
    bounded = tools.bounded_limit(limit)  # checked before the paper, as v1 checks them
    scope = (tools.held_version(corpus, arxiv_id)  # type: ignore[arg-type]
             if arxiv_id and str(arxiv_id).strip() else None)
    matches = corpus.search(embedder, query, limit=bounded, arxiv_id=scope,
                            max_per_paper=3 if scope is None else None, **default_search_options())
    return tools.passage_rows(corpus, [match.chunk for match in matches])  # type: ignore[arg-type]


def inspect(corpus: ScopedCorpus, ids: Sequence[str]) -> list[dict[str, Any]]:
    """The exact text of up to contract.MAX_INSPECT passages, in the order asked
    and each once, sharing INSPECT_CHARACTERS between them; a passage cut short
    ends with CUT. A passage the scope does not show is not returned."""
    wanted = list(dict.fromkeys(ids))
    if len(wanted) > contract.MAX_INSPECT:
        raise ValueError(f"INSPECT reads at most {contract.MAX_INSPECT} passages at a time")
    found = {chunk.chunk_id: chunk for chunk in corpus.chunks_by_id(wanted)}
    rows = tools.passage_rows(corpus, [found[chunk_id] for chunk_id in wanted  # type: ignore[arg-type]
                                       if chunk_id in found])
    share = INSPECT_CHARACTERS // max(len(rows), 1)
    for row in rows:
        if len(row["text"]) > share:
            row["text"] = row["text"][:share - len(CUT)] + CUT
    return rows
