"""Turning chunk text into vectors, behind an interface we can swap."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from verbatim.corpus import CorpusStore, EmbeddingModel

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge asks for this prefix on the query side only; passages are embedded bare.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    """Anything that can turn text into vectors of a fixed width."""

    @property
    def model(self) -> EmbeddingModel: ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class FastEmbedEmbedder:
    """ONNX-backed embedder: no torch, and the weights are ~130MB.

    Loading is deferred until the first call so that importing this module -
    which the CLI does for every command - stays cheap.
    """

    model_name: str = DEFAULT_MODEL_NAME
    _encoder: object | None = None
    _model: EmbeddingModel | None = None

    @property
    def model(self) -> EmbeddingModel:
        if self._model is None:
            self._load()
        assert self._model is not None
        return self._model

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        return _as_lists(self._encoder.embed(list(texts)))  # type: ignore[union-attr]

    def embed_query(self, text: str) -> list[float]:
        self._load()
        return _as_lists(self._encoder.embed([QUERY_PREFIX + text]))[0]  # type: ignore[union-attr]

    def _load(self) -> None:
        if self._encoder is not None:
            return
        try:
            from fastembed import TextEmbedding
        except ImportError as error:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "Embedding needs the 'fastembed' package; install it with 'uv add fastembed'"
            ) from error
        self._encoder = TextEmbedding(model_name=self.model_name)
        description = next(
            item
            for item in TextEmbedding.list_supported_models()
            if item["model"] == self.model_name
        )
        self._model = EmbeddingModel(
            model_id=self.model_name,
            dimensions=int(description["dim"]),
            normalized=True,
            revision=str(description.get("sources", {}).get("hf", self.model_name)),
        )


def embed_pending(
    corpus: CorpusStore,
    embedder: Embedder,
    *,
    batch_size: int = 64,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Embed every chunk that has no vector from this model yet.

    Returns the number embedded, so a caller can tell a fresh run from a no-op.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    model = embedder.model
    corpus.register_embedding_model(model)
    pending = corpus.pending_embeddings(model.model_id, limit=limit)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedder.embed_passages([chunk.text for chunk in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Embedder returned {len(vectors)} vectors for {len(batch)} chunks"
            )
        corpus.record_embeddings(
            model.model_id, [(chunk.chunk_id, vector) for chunk, vector in zip(batch, vectors)]
        )
        if on_progress is not None:
            on_progress(start + len(batch), len(pending))
    return len(pending)


def _as_lists(vectors: Iterable[object]) -> list[list[float]]:
    return [[float(value) for value in vector] for vector in vectors]  # type: ignore[union-attr]
