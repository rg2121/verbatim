"""Ways of turning a question into ranked, citable passages."""

from __future__ import annotations

from verbatim.corpus import ChunkMatch, CorpusStore
from verbatim.embedding import Embedder
from verbatim.vectors import SqliteVecIndex, VectorIndex, ensure_current


def vector_search(
    corpus: CorpusStore,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 10,
    arxiv_id: str | None = None,
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

    scope = None
    if arxiv_id is not None:
        scope = [chunk.chunk_id for chunk in corpus.chunks_for_paper(arxiv_id)]
        if not scope:
            return []
    hits = index.search(embedder.embed_query(query), limit=limit, chunk_ids=scope)
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


def _similarity(distance: float) -> float:
    """Map L2 distance between unit vectors to cosine similarity.

    For unit-length vectors ||a-b||^2 = 2 - 2*cos, so this is exact rather than
    a monotonic stand-in.
    """
    return 1.0 - (distance * distance) / 2.0


def _preview(text: str, width: int = 160) -> str:
    return text if len(text) <= width else text[:width].rstrip() + "…"
