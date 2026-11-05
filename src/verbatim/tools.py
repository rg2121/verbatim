"""The corpus operations offered as tools to `ask --tools` and MCP clients.

Passage search answers what the literature says; a recency question ("what
came out this week") is a catalogue query, which similarity search cannot
answer. Each tool is a plain function, testable without a model. A bad argument
raises ValueError saying what was expected: a model can correct itself from
that, where an empty result would read as an answer.

`verify_citations` and `export_citations` are for MCP clients only, and so are
not in TOOL_SCHEMAS: the local agent's answers are checked as they are written.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
import json
from typing import Any

from verbatim.answers import answer_from_payload, verify_quote
from verbatim.arxiv import Paper
from verbatim.chunking import Chunk
from verbatim.citations import (
    EXPORT_FORMATS,
    Citation,
    citation_from_chunk,
    export_references,
    locator,
    render_locator,
)
from verbatim.corpus import ChunkMatch, CorpusStore, split_arxiv_version
from verbatim.critique import unbacked_sentences, uncited_sentences
from verbatim.embedding import Embedder
from verbatim.reflection import closest_words, passage_texts
from verbatim.retrieval import default_search_options, hybrid_search

MAX_RESULTS = 25
MAX_READ_PAGES = 20
# Below where a client starts to warn: Claude Code does at 10,000 tokens.
READ_CHARACTER_BUDGET = 30_000
MAX_QUOTES = 50
VERIFIED_MEANS = (
    "Verified means these exact words are in that passage. It does not mean "
    "the passage supports the claim made from them."
)


def bounded_limit(limit: int) -> int:
    """A result count within what any tool returns. SQLite reads a negative LIMIT as none."""
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    return min(limit, MAX_RESULTS)


def _clean_id(identifier: object) -> str:
    return str(identifier).strip().removeprefix("arXiv:").removeprefix("arxiv:").strip()


def held_version(corpus: CorpusStore, arxiv_id: object) -> str:
    """The version of a paper held here, from an arXiv ID with or without a version."""
    wanted = _clean_id(arxiv_id)
    paper = corpus.get_paper(wanted) or corpus.latest_version_of(wanted)
    if paper is None:
        raise ValueError(f"no paper {wanted!r} in this corpus: get_paper finds one by ID or title")
    return paper.arxiv_id


def _passage_row(chunk: Chunk, paper: Paper | None) -> dict[str, Any]:
    """A passage with what a reader needs to cite it and find it again."""
    return {
        "chunk_id": chunk.chunk_id,
        "arxiv_id": chunk.arxiv_id,
        "title": paper.title if paper else None,
        "locator": render_locator(chunk.source_kind, chunk.page_number, chunk.section_path),
        "source_kind": chunk.source_kind,
        # HTML has no pages: page_number only counts its sections.
        **({} if chunk.source_kind == "html" else {"page": chunk.page_number}),
        "position": chunk.page_number,
        "section": chunk.section_path.split(" > ")[-1],
        "url": chunk.source_url,
        # The paper's record is the authority, as in a citation.
        "license_url": (paper.license_url if paper else None) or chunk.license_url,
        "text": chunk.text,
    }


def _passage_rows(corpus: CorpusStore, chunks: Sequence[Chunk]) -> list[dict[str, Any]]:
    if not chunks:
        return []
    papers = corpus.papers_by_id(sorted({chunk.arxiv_id for chunk in chunks}))
    return [_passage_row(chunk, papers.get(chunk.arxiv_id)) for chunk in chunks]


def search_chunks(
    corpus: CorpusStore,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 8,
    arxiv_id: str | None = None,
) -> list[dict[str, Any]]:
    """Find passages about a topic, optionally within one paper."""
    if not query.strip():
        raise ValueError("query is empty: pass the words to search for")
    bounded = bounded_limit(limit)
    # A model passing "" for an optional argument means no scope, not a paper called "".
    scope = held_version(corpus, arxiv_id) if arxiv_id and str(arxiv_id).strip() else None
    matches = hybrid_search(
        corpus,
        embedder,
        query,
        limit=bounded,
        arxiv_id=scope,
        max_per_paper=3 if scope is None else None,
        **default_search_options(),
    )
    return _passage_rows(corpus, [match.chunk for match in matches])


def get_paper(corpus: CorpusStore, identifier: str) -> dict[str, Any] | None:
    """Look up one paper by versioned id, base id (current version) or approximate title."""
    wanted = _clean_id(identifier)
    # A title search for nothing matches every paper, and the shortest title would win.
    if not any(character.isalnum() for character in wanted):
        raise ValueError("identifier is empty: pass an arXiv ID or part of a paper's title")
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


def _stamp(moment: datetime) -> str:
    """A moment as the catalogue writes one, so that comparing the text compares the time."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _since_stamp(since: object) -> str:
    try:
        moment = datetime.fromisoformat(str(since).strip())
    except ValueError as error:
        raise ValueError(
            f"since must be an ISO date such as 2026-08-01 or 2026-08-01T00:00:00Z, got {since!r}"
        ) from error
    return _stamp(moment if moment.tzinfo else moment.replace(tzinfo=UTC))


def list_papers(
    corpus: CorpusStore,
    *,
    days: int | None = None,
    since: str | None = None,
    category: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List papers by recency, newest first; `days` counts back from now."""
    bounded = bounded_limit(limit)
    if days is not None and days < 1:
        raise ValueError(f"days must be at least 1, got {days}")
    if since is not None:
        since = _since_stamp(since)
    elif days is not None:
        since = _stamp(datetime.now(UTC) - timedelta(days=days))
    papers = corpus.papers_updated_since(since, category=category, limit=bounded)
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
    """What cites a paper, or what it builds on, among the papers held here.

    `total` is the true count: a list that stops at MAX_RESULTS reads as that many.
    """
    if direction not in ("cited_by", "references"):
        raise ValueError("direction must be 'cited_by' or 'references'")
    bounded = bounded_limit(limit)
    found = get_paper(corpus, identifier)
    if found is None:
        return None
    if direction == "references":
        # A paper's own bibliography is short enough to fetch whole and count.
        every = corpus.papers_cited_by(found["arxiv_id"], limit=100_000)
        related, total = every[:bounded], len(every)
    else:
        base_arxiv_id, _ = split_arxiv_version(found["arxiv_id"])
        related = corpus.papers_citing(base_arxiv_id, limit=bounded)
        total = corpus.citation_counts([base_arxiv_id]).get(base_arxiv_id, 0)
    return {
        "arxiv_id": found["arxiv_id"],
        "title": found["title"],
        "direction": direction,
        "total": total,
        "papers": [
            {"arxiv_id": p.arxiv_id, "title": p.title, "published": p.published}
            for p in related
        ],
    }


def read_pages(
    corpus: CorpusStore, arxiv_id: str, pages: list[int], *, skip: int = 0
) -> dict[str, Any]:
    """Read specific pages of a paper, for following up on a hit.

    Stops near READ_CHARACTER_BUDGET, and says what it left out: one HTML section
    in the corpus runs to 1.8 million characters. `next_skip` passed back as `skip`
    reads on. The budget counts each row as a client receives it, with its fields.
    """
    if not pages:
        raise ValueError("pages is empty: pass the position of a search result")
    if skip < 0:
        raise ValueError(f"skip must be at least 0, got {skip}")
    held = held_version(corpus, arxiv_id)
    asked = pages[:MAX_READ_PAGES]
    chunks = corpus.chunks_on_pages(held, asked)
    paper = corpus.get_paper(held)
    kept: list[dict[str, Any]] = []
    used = 0
    for chunk in chunks[skip:]:
        row = _passage_row(chunk, paper)
        size = len(json.dumps(row, indent=2, ensure_ascii=False))
        # The first always fits, or a long one could never be read.
        if kept and used + size > READ_CHARACTER_BUDGET:
            break
        kept.append(row)
        used += size
    reached = skip + len(kept)
    return {
        "arxiv_id": held,
        "passages": kept,
        "passages_total": len(chunks),
        "next_skip": reached if reached < len(chunks) else None,
        "positions_not_read": pages[MAX_READ_PAGES:],
        "positions_without_passages": sorted(set(asked) - {chunk.page_number for chunk in chunks}),
    }


def _citation_record(citation: Citation) -> dict[str, Any]:
    """What a reader needs to find these words again and check them."""
    return {
        "arxiv_id": citation.arxiv_id,
        "title": citation.title,
        "authors": citation.authors,
        "locator": locator(citation),
        "source_sha256": citation.source_sha256,
        "source_url": citation.source_url,
        "license_url": citation.license_url,
        "retrieved_at": citation.retrieved_at,
    }


def _verdict(
    corpus: CorpusStore, chunk_id: str, quote: str, chunks: dict[str, Chunk], papers: dict[str, Any]
) -> dict[str, Any]:
    """One quote's result, and on failure where its words are, or the nearest that are."""
    verdict: dict[str, Any] = {
        "chunk_id": chunk_id, "quote": quote, "verified": False, "reason": None,
        "did_you_mean": None, "found_in": None, "citation": None,
    }
    chunk = chunks.get(chunk_id)
    paper = papers.get(chunk.arxiv_id) if chunk else None
    if chunk is None or paper is None:
        verdict["reason"] = (
            "no passage with this chunk_id in the corpus: "
            "use a chunk_id from search_chunks or read_pages"
        )
        return verdict
    failure = verify_quote(quote, chunk.text)
    if failure is None:
        verdict["verified"] = True
        verdict["citation"] = _citation_record(citation_from_chunk(chunk, paper, quote))
        return verdict
    verdict["reason"] = failure
    verdict["did_you_mean"] = closest_words(quote, " ".join(chunk.text.split()))
    verdict["found_in"] = next(
        (
            other.chunk_id
            for other in corpus.chunks_for_paper(chunk.arxiv_id)
            if other.chunk_id != chunk_id and verify_quote(quote, other.text) is None
        ),
        None,
    )
    return verdict


def verify_citations(
    corpus: CorpusStore, citations: list[dict[str, str]], *, answer: str | None = None
) -> dict[str, Any]:
    """Check a client's quotes with the check `ask` uses, and read its answer as `ask` would.

    It goes through answer_from_payload, so the two cannot drift apart. In
    `answer`, [n] marks the nth citation. Only words are proved here: a marked
    sentence that shares too few words with its passage is reported, but no
    check can say a passage supports a claim.
    """
    if not citations:
        raise ValueError("citations is empty: pass at least one {chunk_id, quote}")
    if len(citations) > MAX_QUOTES:
        raise ValueError(f"at most {MAX_QUOTES} quotes per call, got {len(citations)}: split them")
    wanted = list(dict.fromkeys(item["chunk_id"] for item in citations))
    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks_by_id(wanted)}
    papers = corpus.papers_by_id(sorted({chunk.arxiv_id for chunk in chunks.values()}))
    matches = [ChunkMatch(chunk=chunk, score=1.0, snippet="") for chunk in chunks.values()]
    checked = answer_from_payload(
        "",
        matches,
        {
            "answer": answer or "",
            "citations": [
                {"marker": marker, "chunk_id": item["chunk_id"], "quoted_text": item["quote"]}
                for marker, item in enumerate(citations, start=1)
            ],
        },
        papers,
    )
    verdicts = [
        _verdict(corpus, item["chunk_id"], item["quote"], chunks, papers) for item in citations
    ]
    verified = sum(verdict["verified"] for verdict in verdicts)
    return {
        "all_verified": verified == len(verdicts),
        "verified": verified,
        "rejected": len(verdicts) - verified,
        "verdicts": verdicts,
        "bibliography": export_references(checked.citations, papers),
        "note": VERIFIED_MEANS,
        "checked_answer": None if answer is None else checked.text,
        "unmarked_sentences": None if answer is None else list(uncited_sentences(checked.text)),
        "unbacked_sentences": (
            None if answer is None else list(unbacked_sentences(checked, passage_texts(matches)))
        ),
    }


def export_citations(
    corpus: CorpusStore, chunk_ids: list[str], *, format: str = "markdown"
) -> dict[str, Any]:
    """Render a reference list in code, so that no model retypes a reference."""
    if not chunk_ids:
        raise ValueError("chunk_ids is empty: pass the chunk_id of each passage to cite")
    if len(chunk_ids) > MAX_QUOTES:
        raise ValueError(f"at most {MAX_QUOTES} passages per call, got {len(chunk_ids)}: split them")
    if format not in EXPORT_FORMATS:
        raise ValueError(f"format must be one of {', '.join(EXPORT_FORMATS)}, got {format!r}")
    wanted = list(dict.fromkeys(chunk_ids))
    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks_by_id(wanted)}
    missing = [chunk_id for chunk_id in wanted if chunk_id not in chunks]
    if missing:
        raise ValueError(
            f"no passages with these chunk_ids: {', '.join(missing)}: "
            "use chunk_ids from search_chunks or read_pages"
        )
    papers = corpus.papers_by_id(sorted({chunk.arxiv_id for chunk in chunks.values()}))
    citations = [
        citation_from_chunk(chunks[chunk_id], papers[chunks[chunk_id].arxiv_id], "")
        for chunk_id in wanted
    ]
    return {
        "format": format,
        # BibTeX and CSL hold one record per paper, with every locator in it.
        "entries": len(citations) if format == "markdown" else len(papers),
        "references": export_references(citations, papers, format),
    }


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
            skip=int(arguments.get("skip", 0)),
        )
    raise ValueError(f"Unknown tool: {name}")
