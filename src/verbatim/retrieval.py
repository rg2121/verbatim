"""Ways of turning a question into ranked, citable passages."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from verbatim.chunking import BODY
from verbatim.corpus import ChunkMatch, CorpusStore
from verbatim.embedding import Embedder
from verbatim.expansion import (
    dedupe_queries,
    fuse_rankings,
    hypothetical_answer,
    rewrite_query,
)
from verbatim.vectors import SqliteVecIndex, VectorIndex, ensure_current

# Standard RRF constant: large enough that the top few ranks are not dominated
# by a single leg, small enough that deep results still fade out.
RRF_SMOOTHING = 60


def vector_search(
    corpus: CorpusStore,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 10,
    arxiv_id: str | None = None,
    include_references: bool = False,
    index: VectorIndex | None = None,
) -> list[ChunkMatch]:
    """Rank passages by embedding similarity rather than shared words.

    Scores are similarity, so higher is better, matching the keyword path.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    model = embedder.model
    stored = corpus.embedding_count(model.model_id)
    if stored == 0:
        return []
    index = index or SqliteVecIndex(corpus.connection, model.dimensions)
    ensure_current(index, stored, model.model_id)

    # Filters are pushed into the index rather than applied to a list of chunk
    # ids: SQLite caps a statement at 32766 variables, so passing every body
    # chunk id would fail at roughly 500 papers.
    hits = index.search(
        embedder.embed_query(query),
        limit=limit,
        arxiv_id=arxiv_id,
        kind=None if include_references else BODY,
    )
    if not hits:
        return []

    by_id = {chunk.chunk_id: chunk for chunk in corpus.chunks_by_id([cid for cid, _ in hits])}
    matches: list[ChunkMatch] = []
    for chunk_id, distance in hits:
        chunk = by_id.get(chunk_id)
        if chunk is None:  # index drifted mid-query; skip rather than fabricate
            continue
        matches.append(
            ChunkMatch(chunk=chunk, score=_similarity(distance), snippet=_preview(chunk.text))
        )
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
) -> list[ChunkMatch]:
    """Fuse keyword and vector rankings with Reciprocal Rank Fusion.

    RRF combines rankings by position rather than by score, so BM25 relevance
    and cosine similarity — which share no scale — never have to be reconciled.

    The keyword leg matches any term rather than all: requiring every word of a
    natural-language question matches nothing, which would silently reduce
    hybrid search to vector-only.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    keyword = corpus.search_chunks(
        query,
        limit=candidates,
        arxiv_id=arxiv_id,
        match="any",
        include_references=include_references,
    )
    vector = vector_search(
        corpus,
        embedder,
        query,
        limit=candidates,
        arxiv_id=arxiv_id,
        include_references=include_references,
        index=index,
    )

    fused: dict[str, float] = defaultdict(float)
    for ranking in (keyword, vector):
        for rank, match in enumerate(ranking, start=1):
            fused[match.chunk.chunk_id] += 1.0 / (smoothing + rank)

    # Prefer the keyword leg's snippet: it highlights the matched terms.
    best: dict[str, ChunkMatch] = {match.chunk.chunk_id: match for match in vector}
    best.update({match.chunk.chunk_id: match for match in keyword})

    ordered = [
        ChunkMatch(chunk=best[chunk_id].chunk, score=score, snippet=best[chunk_id].snippet)
        for chunk_id, score in sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    ]
    # Capped before truncating, so dropping a paper's surplus promotes another
    # paper into the results rather than shortening them.
    if max_per_paper is not None:
        ordered = cap_per_paper(ordered, max_per_paper)
    return ordered[:limit]


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
    """Hybrid search over several phrasings of the same question.

    Each phrasing is retrieved for separately and the rankings are fused, so a
    passage found by more than one phrasing rises. The user's own words are
    always among the phrasings and are never displaced.

    With `hyde`, the vector leg is additionally given the sentence a paper
    answering the question would contain, which puts the query in the register
    of the passages rather than of the question.
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
    """Keep at most N passages per paper, preserving rank order.

    Without this, one verbose paper with a well-matching section fills every
    slot and a corpus-wide question is answered from a single source.
    """
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
    """Map L2 distance between unit vectors to cosine similarity.

    For unit-length vectors ||a-b||^2 = 2 - 2*cos, so this is exact rather than
    a monotonic stand-in.
    """
    return 1.0 - (distance * distance) / 2.0


def _preview(text: str, width: int = 160) -> str:
    return text if len(text) <= width else text[:width].rstrip() + "…"
