from dataclasses import replace
from pathlib import Path

import pytest

from verbatim.arxiv import Paper
from verbatim.chunking import extract_pdf_chunks
from verbatim.corpus import (
    CorpusStore,
    EmbeddingModel,
    pack_vector,
    unpack_vector,
)
from verbatim.downloads import DownloadedPaper, file_sha256

MODEL = EmbeddingModel(
    model_id="bge-small-en-v1.5", dimensions=4, normalized=True, revision="abc123"
)


def paper(arxiv_id: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        summary="A test abstract.",
        authors=["Ada Lovelace"],
        categories=["physics.gen-ph"],
        published="2026-08-01T10:00:00Z",
        updated="2026-08-08T10:00:00Z",
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        license_url=None,
    )


def chunked_corpus(tmp_path: Path, pages: list[str], arxiv_id: str = "2608.11111v1") -> CorpusStore:
    pdf = tmp_path / f"{arxiv_id}.pdf"
    pdf.write_bytes(f"%PDF-1.7 {arxiv_id}\n%%EOF\n".encode())
    record = DownloadedPaper(
        arxiv_id=arxiv_id,
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        license_url=None,
        local_path=str(pdf),
        sha256=file_sha256(pdf),
        byte_count=pdf.stat().st_size,
    )
    corpus = CorpusStore(tmp_path / "corpus.db")
    corpus.upsert_papers([paper(arxiv_id)])
    corpus.record_downloads([record])
    corpus.record_chunks(extract_pdf_chunks([record], extract_pages=lambda _: pages))
    return corpus


def test_vector_round_trips_through_storage() -> None:
    original = [0.5, -0.25, 0.125, 0.0]

    assert unpack_vector(pack_vector(original)) == pytest.approx(original)


def test_embedding_backlog_shrinks_as_vectors_are_recorded(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, ["alpha beta", "gamma delta"]) as corpus:
        corpus.register_embedding_model(MODEL)
        before = corpus.pending_embeddings(MODEL.model_id)
        corpus.record_embeddings(
            MODEL.model_id, [(before[0].chunk_id, [1.0, 0.0, 0.0, 0.0])]
        )
        after = corpus.pending_embeddings(MODEL.model_id)

        assert len(before) == 2
        assert [chunk.chunk_id for chunk in after] == [before[1].chunk_id]
        assert corpus.embedding_count(MODEL.model_id) == 1


def test_rechunking_discards_vectors_for_the_replaced_chunks(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, ["original wording"]) as corpus:
        corpus.register_embedding_model(MODEL)
        pending = corpus.pending_embeddings(MODEL.model_id)
        corpus.record_embeddings(MODEL.model_id, [(pending[0].chunk_id, [1.0, 0.0, 0.0, 0.0])])
        assert corpus.embedding_count(MODEL.model_id) == 1

        record = corpus.all_downloads()[0]
        corpus.record_chunks(
            extract_pdf_chunks([record], extract_pages=lambda _: ["replacement wording"])
        )

        assert corpus.embedding_count(MODEL.model_id) == 0
        assert len(corpus.pending_embeddings(MODEL.model_id)) == 1


def test_registering_a_model_with_a_different_shape_is_rejected(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, ["alpha"]) as corpus:
        corpus.register_embedding_model(MODEL)
        corpus.register_embedding_model(MODEL)  # idempotent

        with pytest.raises(ValueError, match="already stored"):
            corpus.register_embedding_model(
                EmbeddingModel(
                    model_id=MODEL.model_id, dimensions=768, normalized=True, revision="abc123"
                )
            )
        with pytest.raises(ValueError, match="already stored"):
            corpus.register_embedding_model(
                EmbeddingModel(
                    model_id=MODEL.model_id, dimensions=4, normalized=True, revision="def456"
                )
            )


class StubEmbedder:
    """Deterministic stand-in so tests need no model download."""

    def __init__(self, dimensions: int = 4) -> None:
        self.model = EmbeddingModel(
            model_id="stub", dimensions=dimensions, normalized=True, revision="v0"
        )
        self.batches: list[int] = []
        self.seen: list[str] = []

    def embed_passages(self, texts):
        self.batches.append(len(texts))
        self.seen.extend(texts)
        return [[float(len(text)), 0.0, 0.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 0.0, 0.0, 0.0]


def test_embed_pending_is_incremental_and_batches(tmp_path: Path) -> None:
    from verbatim.embedding import embed_pending

    with chunked_corpus(tmp_path, ["alpha", "beta", "gamma", "delta", "epsilon"]) as corpus:
        embedder = StubEmbedder()
        first = embed_pending(corpus, embedder, batch_size=2)
        second = embed_pending(corpus, embedder, batch_size=2)

        assert first == 5
        assert second == 0
        assert embedder.batches == [2, 2, 1]
        assert corpus.embedding_count("stub") == 5


def test_embed_pending_respects_a_limit(tmp_path: Path) -> None:
    from verbatim.embedding import embed_pending

    with chunked_corpus(tmp_path, ["alpha", "beta", "gamma"]) as corpus:
        assert embed_pending(corpus, StubEmbedder(), limit=2) == 2
        assert corpus.embedding_count("stub") == 2
        assert len(corpus.pending_embeddings("stub")) == 1


def test_embed_pending_rejects_a_mismatched_vector_count(tmp_path: Path) -> None:
    from verbatim.embedding import embed_pending

    class Broken(StubEmbedder):
        def embed_passages(self, texts):
            return []

    with chunked_corpus(tmp_path, ["alpha"]) as corpus:
        with pytest.raises(RuntimeError, match="returned 0 vectors"):
            embed_pending(corpus, Broken())


def test_abstracts_are_embedded_once_per_paper(tmp_path: Path) -> None:
    """One vector per paper, and re-running embeds nothing new."""
    from helpers import paper
    from verbatim.corpus import CorpusStore
    from verbatim.embedding import embed_pending_papers

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        corpus.upsert_papers([paper("2608.11111v1"), paper("2608.22222v1")])
        embedder = StubEmbedder()

        first = embed_pending_papers(corpus, embedder)
        second = embed_pending_papers(corpus, embedder)
        stored = corpus.paper_embedding_count(embedder.model.model_id)

    assert (first, second, stored) == (2, 0, 2)
    # What is embedded states the paper's subject: title and abstract.
    assert embedder.seen[0].startswith("Paper 2608.11111v1")
    assert "A test abstract." in embedder.seen[0]


def test_only_the_current_version_of_a_paper_is_embedded(tmp_path: Path) -> None:
    """The abstract index says what the corpus is about, once per paper."""
    from helpers import paper
    from verbatim.corpus import CorpusStore
    from verbatim.embedding import embed_pending_papers

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        corpus.upsert_papers([paper("2608.11111v1")])
        corpus.upsert_papers([paper("2608.11111v2")])

        embedded = embed_pending_papers(corpus, StubEmbedder())

    assert embedded == 1


def test_context_is_part_of_the_model_identity(tmp_path: Path) -> None:
    """Vectors made with and without context are not interchangeable.

    pending_embeddings keys on the model id, so a shared id would let a
    flipped flag leave the corpus holding a silent mixture of the two.
    """
    from verbatim.embedding import FastEmbedEmbedder

    plain = FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5")
    contextual = FastEmbedEmbedder(
        model_name="BAAI/bge-small-en-v1.5", with_context=True
    )

    assert plain.model.model_id != contextual.model.model_id
    assert contextual.model.model_id.endswith("+context")
    assert plain.model.dimensions == contextual.model.dimensions


def test_a_passage_is_embedded_behind_its_title_and_section() -> None:
    """A paragraph reading "we improve on this" carries no sign of its subject."""
    from verbatim.chunking import Chunk
    from verbatim.embedding import passage_text

    chunk = Chunk(
        chunk_id="abc", arxiv_id="2608.05440v1", source_url="u", license_url=None,
        source_sha256="f" * 64, page_number=1, chunk_index=0, char_start=0,
        char_end=9, text="We improve on this by 4 points.",
        chunker_version="7:1200:200",
        section_path="Shape-Aware Conversion > 3 Results",
    )

    assert passage_text(chunk, "Shape-Aware Conversion") == (
        "Shape-Aware Conversion 3 Results\n\nWe improve on this by 4 points."
    )
    # Without a title or a heading there is no context to add, and the passage
    # must be embedded exactly as it is stored.
    bare = replace(chunk, section_path="")
    assert passage_text(bare, "") == chunk.text


def test_context_embedding_uses_the_paper_title_from_the_corpus(tmp_path: Path) -> None:
    from helpers import chunked_corpus
    from verbatim.embedding import embed_pending

    class Contextual(StubEmbedder):
        with_context = True

        def __init__(self) -> None:
            super().__init__()
            self.model = EmbeddingModel(
                model_id="stub+context", dimensions=4, normalized=True, revision="v0"
            )

    with chunked_corpus(tmp_path, {"2608.11111v1": ["Body text about flux."]}) as corpus:
        embedder = Contextual()
        embed_pending(corpus, embedder)

    assert embedder.seen[0].startswith("Paper 2608.11111v1")
    assert "Body text about flux." in embedder.seen[0]
