"""Turning chunk text into vectors, behind an interface we can swap."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from verbatim import config
from verbatim.arxiv import Paper
from verbatim.chunking import Chunk
from verbatim.corpus import CorpusStore, EmbeddingModel

DEFAULT_MODEL_NAME = config.EMBEDDING_MODEL
# bge asks for this prefix on the query side only; passages are embedded bare.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    """Anything that can turn text into vectors of a fixed width."""

    @property
    def model_id(self) -> str:
        """The identity of the vectors this embedder makes, known without loading it."""
        ...

    @property
    def model(self) -> EmbeddingModel: ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class FastEmbedEmbedder:
    """ONNX-backed embedder (no torch, about 130 MB of weights), loaded on first use."""

    model_name: str = DEFAULT_MODEL_NAME
    # Part of the model id: vectors made with and without context are stored
    # apart, so they never mix and can be compared.
    with_context: bool = config.EMBED_WITH_CONTEXT
    _encoder: object | None = None
    _model: EmbeddingModel | None = None

    @property
    def model_id(self) -> str:
        """The identity of the vectors this embedder makes, known without loading it."""
        return f"{self.model_name}+context" if self.with_context else self.model_name

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
            model_id=self.model_id,
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
    chunker_version: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Embed every chunk with no vector from this model yet, returning how many.

    `chunker_version` skips chunks from an older chunker, which are due to be replaced.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    model = embedder.model
    corpus.register_embedding_model(model)
    pending = corpus.pending_embeddings(
        model.model_id, limit=limit, chunker_version=chunker_version
    )
    with_context = getattr(embedder, "with_context", False)
    titles = _titles_for(corpus, pending) if with_context else {}
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedder.embed_passages(
            [
                passage_text(chunk, titles.get(chunk.arxiv_id, ""))
                if with_context
                else chunk.text
                for chunk in batch
            ]
        )
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Embedder returned {len(vectors)} vectors for {len(batch)} chunks"
            )
        corpus.record_embeddings(
            model.model_id, [(chunk.chunk_id, vector) for chunk, vector in zip(batch, vectors, strict=True)]
        )
        if on_progress is not None:
            on_progress(start + len(batch), len(pending))
    return len(pending)


def passage_text(chunk: Chunk, title: str = "") -> str:
    """One passage prefixed by its paper title and section heading.

    Used only with `with_context`, off by default because measuring it was inconclusive.
    """
    heading = chunk.section_path.split(" > ")[-1].strip()
    context = " ".join(part for part in (title.strip(), heading) if part)
    return f"{context}\n\n{chunk.text}" if context else chunk.text


def paper_subject_text(paper: Paper) -> str:
    """A paper's title and abstract, embedded together as what the paper is about."""
    return f"{paper.title.strip()}\n\n{paper.summary.strip()}"


def embed_pending_papers(
    corpus: CorpusStore,
    embedder: Embedder,
    *,
    batch_size: int = 64,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Embed the title and abstract of every current paper with no vector yet."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    model = embedder.model
    corpus.register_embedding_model(model)
    pending = corpus.pending_paper_embeddings(model.model_id, limit=limit)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedder.embed_passages([paper_subject_text(paper) for paper in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Embedder returned {len(vectors)} vectors for {len(batch)} papers"
            )
        corpus.record_paper_embeddings(
            model.model_id,
            [(paper.arxiv_id, vector) for paper, vector in zip(batch, vectors, strict=True)],
        )
        if on_progress is not None:
            on_progress(start + len(batch), len(pending))
    return len(pending)


def _titles_for(corpus: CorpusStore, chunks: Sequence[Chunk]) -> dict[str, str]:
    """Paper titles for a batch of chunks, fetched once rather than per chunk."""
    papers = corpus.papers_by_id(sorted({chunk.arxiv_id for chunk in chunks}))
    return {arxiv_id: paper.title for arxiv_id, paper in papers.items()}


def _as_lists(vectors: Iterable[object]) -> list[list[float]]:
    return [[float(value) for value in vector] for vector in vectors]  # type: ignore[union-attr]
