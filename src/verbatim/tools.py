"""The corpus operations offered as tools to `ask --tools` and MCP clients.

Passage search answers what the literature says; a recency question ("what
came out this week") is a catalogue query, which similarity search cannot
answer. Each tool is a plain function, testable without a model.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from verbatim.corpus import CorpusStore, split_arxiv_version
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
        # The default graph leg; it only fires on definition-shaped questions.
        use_graph=True,
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
    """Look up one paper by versioned id, base id (current version) or approximate title."""
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
    """List papers by recency, newest first; `days` counts back from now."""
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


def get_citations(
    corpus: CorpusStore,
    identifier: str,
    *,
    direction: str = "cited_by",
    limit: int = 20,
) -> dict[str, Any] | None:
    """What cites a paper, or what it builds on, among the papers held here."""
    if direction not in ("cited_by", "references"):
        raise ValueError("direction must be 'cited_by' or 'references'")
    found = get_paper(corpus, identifier)
    if found is None:
        return None
    bounded = min(limit, MAX_RESULTS)
    if direction == "references":
        related = corpus.papers_cited_by(found["arxiv_id"], limit=bounded)
    else:
        base_arxiv_id, _ = split_arxiv_version(found["arxiv_id"])
        related = corpus.papers_citing(base_arxiv_id, limit=bounded)
    return {
        "arxiv_id": found["arxiv_id"],
        "title": found["title"],
        "direction": direction,
        "papers": [
            {"arxiv_id": p.arxiv_id, "title": p.title, "published": p.published}
            for p in related
        ],
    }


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
        "name": "get_citations",
        "description": (
            "List the held papers that cite one paper (direction 'cited_by'), or "
            "the held papers it builds on (direction 'references'). Use for "
            "questions about influence, lineage, or what builds on what; passage "
            "search cannot answer those."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "identifier": {"type": "string", "description": "arXiv ID or approximate title"},
                "direction": {
                    "type": "string",
                    "enum": ["cited_by", "references"],
                    "description": "cited_by: who cites it. references: what it cites",
                },
                "limit": {"type": "integer"},
            },
            "required": ["identifier"],
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
            since=arguments.get("since"),
            category=arguments.get("category"),
            limit=int(arguments.get("limit", 20)),
        )
    if name == "get_citations":
        return get_citations(
            corpus,
            str(arguments.get("identifier", "")),
            direction=str(arguments.get("direction", "cited_by")),
            limit=int(arguments.get("limit", 20)),
        )
    if name == "read_pages":
        return read_pages(
            corpus,
            str(arguments.get("arxiv_id", "")),
            [int(p) for p in arguments.get("pages", [])],
        )
    raise ValueError(f"Unknown tool: {name}")
