"""Re-scoring retrieved passages against the query, one pair at a time.

Retrieval compares a query to passages it has already turned into vectors, so
it never reads the two together. A cross-encoder does, which is the usual next
step after hybrid retrieval.

**Measured worse here, on all three models tried, and therefore off by default**
(over the 42-case development set of the time; the finding did not depend on it):

    hybrid (no rerank)          r@1 0.90   mrr 0.940
    ms-marco-MiniLM-L-6-v2      r@1 0.67   mrr 0.752
    bge-reranker-base           r@1 0.60   mrr 0.701
    jina-reranker-v1-turbo-en   r@1 0.67   mrr 0.741

Reranking pays off when retrieval is mediocre and the right passage is sitting
at rank 8. Here it is already first 90% of the time, so there is almost nothing
to promote and a great deal to demote - any reshuffle is more likely to move a
correct hit down than a wrong one up. Kept behind --rerank because that changes
if the corpus grows harder or the queries get longer, and this is the code that
would measure it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from verbatim.corpus import ChunkMatch

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    """Scores query-passage pairs; higher means more relevant."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


@dataclass
class CrossEncoderReranker:
    """ONNX cross-encoder, ~80MB, loaded on first use like the embedder."""

    model_name: str = DEFAULT_RERANK_MODEL
    _encoder: object | None = None

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        self._load()
        return [float(s) for s in self._encoder.rerank(query, list(passages))]  # type: ignore[union-attr]

    def _load(self) -> None:
        if self._encoder is not None:
            return
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as error:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "Reranking needs fastembed's cross-encoder support; upgrade fastembed"
            ) from error
        self._encoder = TextCrossEncoder(model_name=self.model_name)


def rerank(
    reranker: Reranker,
    query: str,
    matches: Sequence[ChunkMatch],
    *,
    limit: int | None = None,
) -> list[ChunkMatch]:
    """Reorder matches by cross-encoder score, keeping everything else intact.

    The returned score is the cross-encoder's, not the fused rank score, so a
    caller can tell which stage ranked what.
    """
    if not matches:
        return []
    scores = reranker.score(query, [m.chunk.text for m in matches])
    if len(scores) != len(matches):
        raise RuntimeError(
            f"Reranker returned {len(scores)} scores for {len(matches)} passages"
        )
    ordered = sorted(
        (
            ChunkMatch(chunk=m.chunk, score=score, snippet=m.snippet)
            for m, score in zip(matches, scores, strict=True)
        ),
        key=lambda m: -m.score,
    )
    return ordered[:limit] if limit else ordered
