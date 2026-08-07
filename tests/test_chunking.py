from pathlib import Path
from hashlib import sha256

import pytest

from verbatim.chunking import chunk_text, extract_pdf_chunks
from verbatim.downloads import DownloadedPaper
from verbatim.storage import save_chunks


def download(pdf_path: Path) -> DownloadedPaper:
    content = b"%PDF-1.7 placeholder\n%%EOF\n"
    pdf_path.write_bytes(content)
    return DownloadedPaper(
        arxiv_id="2401.12345v1",
        source_url="https://arxiv.org/abs/2401.12345v1",
        pdf_url="https://arxiv.org/pdf/2401.12345v1",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        local_path=str(pdf_path),
        sha256=sha256(content).hexdigest(),
        byte_count=len(content),
    )


def test_chunk_text_normalizes_and_overlaps_at_word_boundaries() -> None:
    chunks = chunk_text("alpha   beta gamma delta epsilon", chunk_size=16, overlap=5)

    assert chunks == ["alpha beta gamma", "gamma delta", "delta epsilon"]


def test_chunk_text_does_not_start_a_chunk_in_the_middle_of_a_word() -> None:
    chunks = chunk_text("alpha beta gamma delta epsilon", chunk_size=25, overlap=3)

    assert chunks == ["alpha beta gamma delta", "epsilon"]


def test_extract_pdf_chunks_preserves_page_provenance(tmp_path: Path) -> None:
    paper = download(tmp_path / "test.pdf")
    chunks = extract_pdf_chunks(
        [paper],
        chunk_size=50,
        overlap=5,
        extract_pages=lambda _: ["First page text.", "Second page text."],
    )
    destination = tmp_path / "chunks.jsonl"
    save_chunks(chunks, destination)

    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert all(chunk.arxiv_id == "2401.12345v1" for chunk in chunks)
    assert all(chunk.pdf_sha256 == paper.sha256 for chunk in chunks)
    assert len(destination.read_text().splitlines()) == 2


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (20, -1), (20, 20)],
)
def test_chunk_text_rejects_invalid_chunk_settings(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=chunk_size, overlap=overlap)


def test_extract_pdf_chunks_rejects_a_file_that_does_not_match_its_manifest(tmp_path: Path) -> None:
    paper = download(tmp_path / "test.pdf")
    (tmp_path / "test.pdf").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        extract_pdf_chunks([paper], extract_pages=lambda _: ["not reached"])
