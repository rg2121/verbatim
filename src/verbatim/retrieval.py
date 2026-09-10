"""Ways of turning a question into ranked, citable passages."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import re

from verbatim.chunking import BODY
from verbatim.corpus import ChunkMatch, CorpusStore, PaperFilter
from verbatim.embedding import Embedder
from verbatim.graph import anchored_matches
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

# Questions that set two named things against each other. Deliberately about
# vocabulary, not structure: "compare A and B" and "how does A differ from B"
# announce themselves, and a parse of which words are A and which are B is not
# needed because the title leg searches the whole question anyway. "different"
# is only comparative with "from" - "the different approaches to X" is a
# single-subject phrasing.
COMPARISON_SHAPE = re.compile(
    r"\b(?:compare|comparison|contrast|versus|vs\.?|differs?|difference)\b"
    r"|\bdifferent\s+from\b",
    re.IGNORECASE,
)


def comparison_shaped(question: str) -> bool:
    """Whether a question names things to set against each other.

    The gate for the compound title leg, in the mould of the definitional
    gate: the always-on title leg was measured worse on single-subject
    questions and best-in-class on compound ones, so the shape of the question
    decides whether it runs.
    """
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
    """Rank passages by embedding similarity rather than shared words.

    Scores are similarity, so higher is better, matching the keyword path.

    A paper filter is applied after the index returns, because the vector
    index carries only chunk_id, arxiv_id and kind as metadata: filtering on
    publication date or subject would mean rebuilding it with those columns.
    The search over-fetches instead, so a narrow filter can still come back
    with fewer than `limit` results rather than silently ranking the wrong
    papers.
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

    # Filters are pushed into the index rather than applied to a list of chunk
    # ids: SQLite caps a statement at 32766 variables, so passing every body
    # chunk id would fail at roughly 500 papers.
    hits = index.search(
        embedder.embed_query(query),
        limit=fetch,
        arxiv_id=arxiv_id,
        kind=None if include_references else BODY,
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
) -> list[ChunkMatch]:
    """Fuse keyword and vector rankings with Reciprocal Rank Fusion.

    RRF combines rankings by position rather than by score, so BM25 relevance
    and cosine similarity — which share no scale — never have to be reconciled.

    The keyword leg matches any term rather than all: requiring every word of a
    natural-language question matches nothing, which would silently reduce
    hybrid search to vector-only.

    `use_graph` adds passages from the paper a definitional question is named
    after, chosen by title match backed by in-corpus citations. It contributes
    nothing to any other kind of question - that gate is what separates it
    from `use_titles`, which runs everywhere and was measured worse overall.

    `use_compare` runs the same title leg as `use_titles`, but only on
    comparison-shaped questions ("compare A and B", "how does A differ from
    B") - the shape that names its papers almost by title, where the leg was
    measured best-in-class. Like the graph leg, the title leg stands down
    whenever the caller scoped the search to one paper or filtered the
    papers, because it names papers of its own.
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

    # The title and anchor legs name papers of their own, so they stand down
    # whenever the caller has already said which papers are allowed: a search
    # scoped to one paper or filtered by date or category must not be handed
    # passages from outside that scope, however well-shaped the question.
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
    # Capped before truncating, so dropping a paper's surplus promotes another
    # paper into the results rather than shortening them.
    if max_per_paper is not None:
        ordered = cap_per_paper(ordered, max_per_paper)
    return ordered[:limit]


def title_search(
    corpus: CorpusStore,
    query: str,
    *,
    papers: int = 5,
    per_paper: int = 3,
    include_references: bool = False,
) -> list[ChunkMatch]:
    """Passages from the papers whose *titles* match the question.

    A third kind of evidence, and the only one that can answer a definitional
    question. "What is the Adam optimizer" is drowned in passage space - 3,152
    papers here mention Adam and the one that defines it holds 53 passages -
    but the paper called "Adam: A Method for Stochastic Optimization" is the
    first title match. Within each paper the query still chooses the passages,
    so a title match contributes evidence rather than an arbitrary excerpt.
    """
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
