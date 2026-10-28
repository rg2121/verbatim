"""Reranking retrieved passages with a second model.

Two kinds: a cross-encoder reads question and passage together; ColBERT's late
interaction encodes each token separately and matches them. Both raise mrr on
the held-out key, ColBERT more, but both bury definitional questions, ranking
the passages that use a term above the paper that introduced it
(docs/findings.md). So the default search reranks with ColBERT only where the
keyword and vector legs disagree, and never on a definition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import functools
from typing import Protocol

from verbatim import config
from verbatim.corpus import ChunkMatch

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
# 96 numbers per token, about 130 MB.
DEFAULT_LATE_INTERACTION_MODEL = "answerdotai/answerai-colbert-small-v1"


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
        self._encoder = TextCrossEncoder(
            model_name=self.model_name, cache_dir=str(config.MODEL_CACHE)
        )


def maxsim(query_tokens, passage_tokens) -> float:
    """Late interaction's score: each query token's best match in the passage, summed."""
    return float((query_tokens @ passage_tokens.T).max(axis=1).sum())


@dataclass
class LateInteractionReranker:
    """ColBERT: one vector per token, so a rare term keeps its own match, loaded on first use."""

    model_name: str = DEFAULT_LATE_INTERACTION_MODEL
    _encoder: object | None = None

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        self._load()
        question = next(iter(self._encoder.query_embed([query])))  # type: ignore[union-attr]
        # One at a time: a batch pads each passage to the longest, which on a CPU
        # doubled the time and evicted the corpus from the page cache.
        return [maxsim(question, tokens)
                for tokens in self._encoder.embed(list(passages), batch_size=1)]  # type: ignore[union-attr]

    def _load(self) -> None:
        if self._encoder is not None:
            return
        from fastembed import LateInteractionTextEmbedding

        self._encoder = LateInteractionTextEmbedding(
            model_name=self.model_name, cache_dir=str(config.MODEL_CACHE)
        )


RERANKERS = {"cross": CrossEncoderReranker, "colbert": LateInteractionReranker}


def make_reranker(name: str) -> Reranker:
    if name not in RERANKERS:
        raise ValueError(f"unknown reranker {name!r}; choose {' or '.join(RERANKERS)}")
    return RERANKERS[name]()


@functools.cache
def shared_reranker(name: str) -> Reranker:
    """One reranker per name and process, so a server loads its model once."""
    return make_reranker(name)


def default_reranker() -> Reranker | None:
    """The reranker VERBATIM_RERANKER names, or None when it is off."""
    name = config.RERANKER
    if name == "off":
        return None
    if name not in RERANKERS:
        raise ValueError(f"VERBATIM_RERANKER must be {', '.join(RERANKERS)} or off, got {name!r}")
    return shared_reranker(name)


def rerank(
    reranker: Reranker,
    query: str,
    matches: Sequence[ChunkMatch],
    *,
    limit: int | None = None,
) -> list[ChunkMatch]:
    """Reorder matches by the reranker's score, which replaces the fused score."""
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
