"""Ways of turning a question into ranked, citable passages."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import re
from typing import Any

from verbatim.chunking import REFERENCES
from verbatim.corpus import ChunkMatch, CorpusStore, PaperFilter
from verbatim.embedding import Embedder
from verbatim.expansion import (
    RRF_SMOOTHING,
    dedupe_queries,
    fuse_rankings,
    hypothetical_answer,
    rewrite_query,
)
from verbatim.graph import anchored_matches, definitional_subject
from verbatim.reranking import Reranker, default_reranker, rerank
from verbatim.vectors import SqliteVecIndex, VectorIndex, ensure_current

# Questions setting two named things against each other, recognised by vocabulary.
# "different" counts only with "from": "the different approaches to X" is one subject.
COMPARISON_SHAPE = re.compile(
    r"\b(?:compare|comparison|contrast|versus|vs\.?|differs?|difference)\b"
    r"|\bdifferent\s+from\b",
    re.IGNORECASE,
)


def comparison_shaped(question: str) -> bool:
    """Whether a question names things to set against each other; gates the title leg."""
    return COMPARISON_SHAPE.search(question) is not None


# How much wider to search when a paper filter will discard results afterwards.
FILTER_OVERFETCH = 8


def vector_search(
    corpus: CorpusStore,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 10,
    arxiv_id: str | None = None,
    include_references: bool = False,
    index: VectorIndex | None = None,
    papers: PaperFilter | None = None,
) -> list[ChunkMatch]:
    """Rank passages by embedding similarity; higher scores are better.

    A paper filter is applied after the index returns, so the search over-fetches
    and may return fewer than `limit` results.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    model = embedder.model
    stored = corpus.embedding_count(model.model_id)
    if stored == 0:
        return []
    index = index or SqliteVecIndex(corpus.connection, model.dimensions)
    # Over-fetch when filtering, since matches are discarded after ranking.
    fetch = limit * FILTER_OVERFETCH if papers else limit
    ensure_current(index, stored, model.model_id, corpus.chunk_generation())

    # Filters go into the index query; a list of chunk ids would exceed SQLite's
    # variable limit.
    hits = index.search(
        embedder.embed_query(query),
        limit=fetch,
        arxiv_id=arxiv_id,
        exclude_kind=None if include_references else REFERENCES,
    )
    if not hits:
        return []

    by_id = {chunk.chunk_id: chunk for chunk in corpus.chunks_by_id([cid for cid, _ in hits])}
    allowed = corpus.papers_matching(papers, [chunk.arxiv_id for chunk in by_id.values()]) if papers else None
    matches: list[ChunkMatch] = []
    for chunk_id, distance in hits:
        chunk = by_id.get(chunk_id)
        if chunk is None:  # index drifted mid-query; skip rather than fabricate
            continue
        if allowed is not None and chunk.arxiv_id not in allowed:
            continue
        matches.append(
            ChunkMatch(chunk=chunk, score=_similarity(distance), snippet=_preview(chunk.text))
        )
        if len(matches) == limit:
            break
    return matches


def hybrid_search(
    corpus: CorpusStore,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 10,
    arxiv_id: str | None = None,
    candidates: int = 50,
    smoothing: int = RRF_SMOOTHING,
    max_per_paper: int | None = None,
    include_references: bool = False,
    index: VectorIndex | None = None,
    use_titles: bool = False,
    use_graph: bool = False,
    use_compare: bool = False,
    papers: PaperFilter | None = None,
    reranker: Reranker | None = None,
    rerank_pool: int = 50,
    rerank_when_legs_disagree: bool = False,
    rerank_skips_definitions: bool = False,
) -> list[ChunkMatch]:
    """Fuse keyword and vector rankings with Reciprocal Rank Fusion.

    RRF combines rankings by position, so BM25 and cosine scores never need a
    common scale. The keyword leg matches any term. `use_graph` adds passages
    from the paper a definitional question is named after; `use_compare` adds
    title matches on comparison-shaped questions, `use_titles` on every
    question. Those legs stand down when the search is scoped or filtered.
    A `reranker` reorders the top `rerank_pool` fused passages, or, gated,
    only when the keyword and vector legs disagree on the top paper. It can
    skip definition-shaped questions, and never reorders a title leg's choice.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    keyword = corpus.search_chunks(
        query,
        limit=candidates,
        arxiv_id=arxiv_id,
        match="any",
        include_references=include_references,
        papers=papers,
    )
    vector = vector_search(
        corpus,
        embedder,
        query,
        limit=candidates,
        arxiv_id=arxiv_id,
        include_references=include_references,
        index=index,
        papers=papers,
    )

    # The title and graph legs name papers of their own, so they stand down when
    # the caller has restricted which papers are allowed.
    unscoped = arxiv_id is None and not papers
    titled = use_titles or (use_compare and comparison_shaped(query))
    titles = (
        title_search(corpus, query, include_references=include_references)
        if titled and unscoped
        else []
    )
    anchors = (
        anchored_matches(corpus, query, include_references=include_references)
        if use_graph and unscoped
        else []
    )

    legs = [keyword, vector]
    for extra in (titles, anchors):
        if extra:
            legs.append(extra)

    fused: dict[str, float] = defaultdict(float)
    for ranking in legs:
        for rank, match in enumerate(ranking, start=1):
            fused[match.chunk.chunk_id] += 1.0 / (smoothing + rank)

    # Prefer the keyword leg's snippet: it highlights the matched terms.
    best: dict[str, ChunkMatch] = {match.chunk.chunk_id: match for match in vector}
    best.update({match.chunk.chunk_id: match for match in titles})
    best.update({match.chunk.chunk_id: match for match in anchors})
    best.update({match.chunk.chunk_id: match for match in keyword})

    ordered = [
        ChunkMatch(chunk=best[chunk_id].chunk, score=score, snippet=best[chunk_id].snippet)
        for chunk_id, score in sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    ]
    # Rerankers bury the paper a term is named after under passages that use it,
    # so a reranker leaves alone what a title leg chose and, if asked, definitions.
    if (
        reranker is not None
        and not titles
        and not (rerank_skips_definitions and definitional_subject(query))
        and (not rerank_when_legs_disagree or legs_disagree(keyword, vector))
    ):
        ordered = rerank(reranker, query, ordered[:rerank_pool]) + ordered[rerank_pool:]
    # Capped before truncating, so dropping a paper's surplus promotes another
    # paper into the results rather than shortening them.
    if max_per_paper is not None:
        ordered = cap_per_paper(ordered, max_per_paper)
    return ordered[:limit]


def legs_disagree(keyword: list[ChunkMatch], vector: list[ChunkMatch]) -> bool:
    """Whether the two legs name different top papers, or one found nothing.

    When both put the same paper first, fusion has nothing to settle.
    """
    if not keyword or not vector:
        return True
    return keyword[0].chunk.arxiv_id != vector[0].chunk.arxiv_id


def default_search_options() -> dict[str, Any]:
    """How every front end searches: the graph leg, and the reranker where it was measured to help.

    The reranker (VERBATIM_RERANKER, ColBERT unless set) runs only where the
    keyword and vector legs disagree on the top paper, and never on a
    definitional question (docs/findings.md).
    """
    return {
        "use_graph": True,
        "reranker": default_reranker(),
        "rerank_when_legs_disagree": True,
        "rerank_skips_definitions": True,
    }


def title_search(
    corpus: CorpusStore,
    query: str,
    *,
    papers: int = 5,
    per_paper: int = 3,
    include_references: bool = False,
) -> list[ChunkMatch]:
    """Passages from the papers whose titles match the question, chosen within each by the query."""
    matches: list[ChunkMatch] = []
    for arxiv_id, _ in corpus.search_titles(query, limit=papers):
        matches.extend(
            corpus.search_chunks(
                query,
                limit=per_paper,
                arxiv_id=arxiv_id,
                match="any",
                include_references=include_references,
            )
        )
    return matches


def expanded_search(
    corpus: CorpusStore,
    embedder: Embedder,
    query: str,
    *,
    complete: Callable[[str, str], str],
    rewrites: int = 0,
    hyde: bool = False,
    limit: int = 10,
    **options: object,
) -> list[ChunkMatch]:
    """Hybrid search over several phrasings, fused by rank; the original wording is always one.

    With `hyde`, a hypothetical answer sentence is searched too.
    """
    queries = (
        rewrite_query(complete, query, variants=rewrites) if rewrites else [query]
    )
    if hyde:
        queries.append(hypothetical_answer(complete, query))
    rankings = [
        hybrid_search(corpus, embedder, phrasing, limit=limit, **options)  # type: ignore[arg-type]
        for phrasing in dedupe_queries(queries)
    ]
    if len(rankings) == 1:
        return rankings[0]
    fused = fuse_rankings(rankings)
    max_per_paper = options.get("max_per_paper")
    if isinstance(max_per_paper, int):
        fused = cap_per_paper(fused, max_per_paper)
    return fused[:limit]


def cap_per_paper(matches: list[ChunkMatch], max_per_paper: int) -> list[ChunkMatch]:
    """Keep at most `max_per_paper` passages per paper, preserving rank order."""
    if max_per_paper < 1:
        raise ValueError("max_per_paper must be at least 1")
    seen: dict[str, int] = defaultdict(int)
    kept: list[ChunkMatch] = []
    for match in matches:
        arxiv_id = match.chunk.arxiv_id
        if seen[arxiv_id] >= max_per_paper:
            continue
        seen[arxiv_id] += 1
        kept.append(match)
    return kept


def _similarity(distance: float) -> float:
    """Cosine similarity from the L2 distance between unit vectors (exact: |a-b|^2 = 2 - 2cos)."""
    return 1.0 - (distance * distance) / 2.0


def _preview(text: str, width: int = 160) -> str:
    return text if len(text) <= width else text[:width].rstrip() + "…"
