"""The operations a research assistant can perform, as callable tools.

Three kinds of question reach a literature assistant, and only one of them is a
retrieval problem:

- "What do papers say about X?" — similarity search over passages.
- "What does paper X say about Y?" — resolve the paper, then search within it.
- "What came out this week?" — a SQL question. Similarity search returns things
  that *resemble* a query; it has no notion of recency and cannot promise
  completeness, so answering this with a vector index would quietly lie.

Each tool is a plain function over the corpus, so it is useful directly and
testable without a model. The schemas below describe them to one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from verbatim.corpus import CorpusStore
from verbatim.embedding import Embedder
from verbatim.retrieval import hybrid_search

MAX_RESULTS = 25


def search_chunks(
    corpus: CorpusStore,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 8,
    arxiv_id: str | None = None,
) -> list[dict[str, Any]]:
    """Find passages about a topic, optionally within one paper."""
    matches = hybrid_search(
        corpus,
        embedder,
        query,
        limit=min(limit, MAX_RESULTS),
        arxiv_id=arxiv_id,
        max_per_paper=3 if arxiv_id is None else None,
    )
    return [
        {
            "chunk_id": m.chunk.chunk_id,
            "arxiv_id": m.chunk.arxiv_id,
            "page": m.chunk.page_number,
            "section": m.chunk.section_path.split(" > ")[-1],
            "text": m.chunk.text,
        }
        for m in matches
    ]


def get_paper(corpus: CorpusStore, identifier: str) -> dict[str, Any] | None:
    """Look up one paper by arXiv ID, with or without a version, or by title.

    Tried in order of confidence: the exact versioned id, then the base id
    resolved to its current version, then an approximate title match.
    """
    wanted = identifier.strip().removeprefix("arXiv:").removeprefix("arxiv:").strip()
    paper = corpus.get_paper(wanted) or corpus.latest_version_of(wanted)
    if paper is None:
        candidates = corpus.find_papers_by_title(wanted, limit=1)
        paper = candidates[0] if candidates else None
    if paper is None:
        return None
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": paper.authors,
        "categories": paper.categories,
        "published": paper.published,
        "updated": paper.updated,
        "abstract": paper.summary,
        "url": paper.source_url,
        "license_url": paper.license_url,
    }


def list_papers(
    corpus: CorpusStore,
    *,
    days: int | None = None,
    since: str | None = None,
    category: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List papers by recency, newest first.

    `days` is relative to now, which is how the question is usually asked.
    """
    if days is not None and since is None:
        since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    papers = corpus.papers_updated_since(
        since, category=category, limit=min(limit, MAX_RESULTS)
    )
    return [
        {
            "arxiv_id": p.arxiv_id,
            "title": p.title,
            "updated": p.updated,
            "categories": p.categories,
            "url": p.source_url,
        }
        for p in papers
    ]


def read_pages(
    corpus: CorpusStore, arxiv_id: str, pages: list[int]
) -> list[dict[str, Any]]:
    """Read specific pages of a paper, for following up on a hit."""
    return [
        {
            "chunk_id": c.chunk_id,
            "arxiv_id": c.arxiv_id,
            "page": c.page_number,
            "section": c.section_path.split(" > ")[-1],
            "text": c.text,
        }
        for c in corpus.chunks_on_pages(arxiv_id, pages[:20])
    ]


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_chunks",
        "description": (
            "Search the corpus for passages about a topic. Use for questions about "
            "what the literature says. Pass arxiv_id to search within one paper."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "description": "Passages to return, default 8"},
                "arxiv_id": {"type": "string", "description": "Restrict to one paper version"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_paper",
        "description": (
            "Look up one paper's metadata by arXiv ID or approximate title. Use this "
            "first when a question names a specific paper."
        ),
        "parameters": {
            "type": "object",
            "properties": {"identifier": {"type": "string"}},
            "required": ["identifier"],
        },
    },
    {
        "name": "list_papers",
        "description": (
            "List papers by recency, newest first. Use for questions about what is "
            "new or recent; passage search cannot answer those."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "How far back, in days"},
                "since": {"type": "string", "description": "ISO timestamp lower bound"},
                "category": {"type": "string", "description": "e.g. cs.CL"},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "read_pages",
        "description": "Read specific pages of one paper, to follow up on a search hit.",
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string"},
                "pages": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["arxiv_id", "pages"],
        },
    },
]


def call_tool(
    name: str,
    arguments: dict[str, Any],
    corpus: CorpusStore,
    embedder: Embedder,
) -> Any:
    """Dispatch one tool call, rejecting names that were never offered."""
    if name == "search_chunks":
        return search_chunks(
            corpus,
            embedder,
            str(arguments.get("query", "")),
            limit=int(arguments.get("limit", 8)),
            arxiv_id=arguments.get("arxiv_id"),
        )
    if name == "get_paper":
        return get_paper(corpus, str(arguments.get("identifier", "")))
    if name == "list_papers":
        days = arguments.get("days")
        return list_papers(
            corpus,
            days=int(days) if days is not None else None,
            # Passed through rather than dropped: silently ignoring it would
            # return an unfiltered list that looks like a filtered one.
            since=arguments.get("since"),
            category=arguments.get("category"),
            limit=int(arguments.get("limit", 20)),
        )
    if name == "read_pages":
        return read_pages(
            corpus,
            str(arguments.get("arxiv_id", "")),
            [int(p) for p in arguments.get("pages", [])],
        )
    raise ValueError(f"Unknown tool: {name}")
