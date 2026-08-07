from pathlib import Path

import pytest

from verbatim.chunking import chunk_text, extract_pdf_chunks
from verbatim.downloads import DownloadedPaper
from verbatim.storage import save_chunks


def download() -> DownloadedPaper:
    return DownloadedPaper(
        arxiv_id="2401.12345v1",
        source_url="https://arxiv.org/abs/2401.12345v1",
        pdf_url="https://arxiv.org/pdf/2401.12345v1",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        local_path="test.pdf",
        sha256="a" * 64,
        byte_count=123,
    )


def test_chunk_text_normalizes_and_overlaps_at_word_boundaries() -> None:
    chunks = chunk_text("alpha   beta gamma delta epsilon", chunk_size=16, overlap=5)

    assert chunks == ["alpha beta gamma", "gamma delta", "delta epsilon"]


def test_extract_pdf_chunks_preserves_page_provenance(tmp_path: Path) -> None:
    chunks = extract_pdf_chunks(
        [download()],
        chunk_size=50,
        overlap=5,
        extract_pages=lambda _: ["First page text.", "Second page text."],
    )
    destination = tmp_path / "chunks.jsonl"
    save_chunks(chunks, destination)

    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert all(chunk.arxiv_id == "2401.12345v1" for chunk in chunks)
    assert all(chunk.pdf_sha256 == "a" * 64 for chunk in chunks)
    assert len(destination.read_text().splitlines()) == 2


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (20, -1), (20, 20)],
)
def test_chunk_text_rejects_invalid_chunk_settings(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=chunk_size, overlap=overlap)
