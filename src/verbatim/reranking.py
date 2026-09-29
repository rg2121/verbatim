"""Cross-encoder reranking of retrieved passages, behind `--rerank`.

Off by default: it made retrieval worse on all three models tried, because the
right passage is usually already first and a reshuffle mostly demotes it.
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
    """ONNX cross-encoder (about 80 MB), loaded on first use."""

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
    """Reorder matches by cross-encoder score, which replaces the fused score."""
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
