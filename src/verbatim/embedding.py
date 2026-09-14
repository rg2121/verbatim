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
        """The identity of the vectors this embedder makes, without loading it.

        Callers that only need to name the vectors - counting them, syncing the
        index - should not pay seconds of ONNX startup to ask.
        """
        return f"{self.model_name}+context" if self.with_context else self.model_name

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
    # Whether passages were embedded with their title and section in front.
    # Part of the model's identity, not a switch beside it: vectors made with
    # and without context are not interchangeable, and `pending_embeddings`
    # keys on the model id, so a shared id would let a flipped flag leave the
    # corpus holding a silent mixture of the two. Separate ids also mean the
    # two can be measured side by side without either being thrown away.
    with_context: bool = config.EMBED_WITH_CONTEXT
    _encoder: object | None = None
    _model: EmbeddingModel | None = None

    @property
    def model_id(self) -> str:
        """The identity of the vectors this embedder makes, without loading it.

        Callers that only need to name the vectors - counting them, syncing the
        index - should not pay seconds of ONNX startup to ask.
        """
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
    """Embed every chunk that has no vector from this model yet.

    Returns the number embedded, so a caller can tell a fresh run from a no-op.
    `chunker_version` skips passages an older chunker produced, which are due
    to be replaced and whose vectors would be discarded with them.
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
    """The text embedded for one passage, with the context it sits in.

    A passage is embedded alone by default, which loses what it is part of: a
    paragraph reading "we improve on this by 4 points" carries no sign of the
    subject, and a section heading and paper title are exactly the missing
    words. This is the cheap form of contextual retrieval - the published
    version writes a context sentence per chunk with a model, which is one
    inference call per chunk and not affordable over a corpus this size, while
    the title and section path are already stored and cost nothing.

    Off by default until measured on this corpus, because prefixing every
    passage with the same title also makes a paper's passages resemble each
    other, which could cost precision within a paper.
    """
    heading = chunk.section_path.split(" > ")[-1].strip()
    context = " ".join(part for part in (title.strip(), heading) if part)
    return f"{context}\n\n{chunk.text}" if context else chunk.text


def paper_subject_text(paper: Paper) -> str:
    """The text that states what a paper is about.

    Title and abstract together: the title alone is too short to embed well,
    and the abstract alone loses terms that only appear in the title.
    """
    return f"{paper.title.strip()}\n\n{paper.summary.strip()}"


def embed_pending_papers(
    corpus: CorpusStore,
    embedder: Embedder,
    *,
    batch_size: int = 64,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Embed the title and abstract of every paper that has no vector yet.

    Mirrors embed_pending, and is incremental on the same terms: a paper is
    embedded only if it has no vector for this model.
    """
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
