from pathlib import Path
from hashlib import sha256

import pytest

from verbatim.chunking import chunk_page, chunk_text, extract_pdf_chunks
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


def test_chunk_text_folds_typographic_ligatures_to_ascii() -> None:
    (chunk,) = chunk_text("magnetic ﬂux and eﬃciency", chunk_size=100, overlap=10)

    assert chunk == "magnetic flux and efficiency"


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


def test_chunk_text_keeps_every_character_of_a_token_longer_than_the_chunk() -> None:
    chunks = chunk_text("aa " + "b" * 24 + " cc", chunk_size=10, overlap=3)

    assert sum(chunk.count("b") for chunk in chunks) == 24
    assert chunks == ["aa", "bbbbbbbbb", "bbbbbbbbbb", "bbbbb cc"]


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


def marked(pages: list[str]) -> list[str]:
    """Chunk one paper's pages and return the kind of each chunk."""
    from verbatim.chunking import mark_references
    from verbatim.chunking import Chunk

    chunks = []
    for page_number, text in enumerate(pages, start=1):
        for index, span in enumerate(chunk_page(text, chunk_size=200, overlap=20)):
            chunks.append(
                Chunk(
                    chunk_id=f"c{page_number}{index}",
                    arxiv_id="2608.11111v1",
                    source_url="u",
                    license_url=None,
                    pdf_sha256="s",
                    page_number=page_number,
                    chunk_index=index,
                    char_start=span.char_start,
                    char_end=span.char_end,
                    text=span.text,
                    chunker_version="3:200:20",
                )
            )
    return [chunk.kind for chunk in mark_references(chunks)]


def test_bibliography_and_everything_after_it_are_marked_as_references() -> None:
    kinds = marked(
        [
            "Introduction. We study flux.",
            "Method. We measure flux carefully.",
            "Conclusion follows. References [1] A. Smith, Flux, 2020.",
            "[2] B. Jones, More flux, 2021.",
        ]
    )

    assert kinds[0] == "body" and kinds[1] == "body"
    assert kinds[2] == "references" and kinds[3] == "references"


def test_prose_merely_mentioning_references_is_not_marked() -> None:
    kinds = marked(
        [
            "Introduction to flux.",
            "We follow the References section of prior work when reporting results.",
            "Discussion of the measured flux values and their implications here.",
        ]
    )

    assert set(kinds) == {"body"}


def test_an_early_bibliography_mention_does_not_swallow_the_paper() -> None:
    kinds = marked(
        [
            "References [1] A. Smith, 2020. (a quoted reference list in the abstract)",
            "Body text about flux measurements.",
            "More body text about flux measurements and their meaning.",
        ]
    )

    assert kinds[0] == "body"


def test_a_paper_with_no_bibliography_is_left_alone() -> None:
    assert set(marked(["Only body text.", "Still only body text about flux."])) == {"body"}
