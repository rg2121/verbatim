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
