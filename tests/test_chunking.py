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
    assert all(chunk.source_sha256 == paper.sha256 for chunk in chunks)
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
    """Chunk one paper's Markdown pages and return the kind of each chunk."""
    from verbatim.downloads import DownloadedPaper
    from verbatim.chunking import extract_pdf_chunks
    import tempfile
    from hashlib import sha256 as _sha

    directory = Path(tempfile.mkdtemp())
    pdf = directory / "paper.pdf"
    content = b"%PDF-1.7 x\n%%EOF\n"
    pdf.write_bytes(content)
    record = DownloadedPaper(
        arxiv_id="2608.11111v1",
        source_url="u",
        pdf_url="p",
        license_url=None,
        local_path=str(pdf),
        sha256=_sha(content).hexdigest(),
        byte_count=len(content),
    )
    chunks = extract_pdf_chunks(
        [record], chunk_size=400, overlap=40, extract_pages=lambda _: pages
    )
    return [chunk.kind for chunk in chunks]


def test_a_bibliography_continuing_onto_the_next_page_stays_marked() -> None:
    kinds = marked(
        [
            "## Introduction\n\nWe study flux.",
            "## Method\n\nWe measure flux carefully.",
            "## Conclusion\n\nFlux moves.\n\n## References\n\n[1] A. Smith, Flux, 2020.",
            "[2] B. Jones, More flux, 2021.",
        ]
    )

    assert kinds[:3] == ["body", "body", "body"]
    assert set(kinds[3:]) == {"references"}


def test_a_section_that_merely_mentions_references_is_not_marked() -> None:
    kinds = marked(
        [
            "## Introduction\n\nFlux is studied.",
            "## Reference-Assisted Human Audit\n\nWe audit against a reference set.",
            "## Evaluated Model References\n\nModels are listed with their versions.",
        ]
    )

    assert set(kinds) == {"body"}


def test_only_the_bibliography_is_marked_not_what_follows_it() -> None:
    """Appendices commonly sit after the references and are real content."""
    kinds = marked(
        [
            "## Introduction\n\nBody text about flux measurements.",
            "## References\n\n[1] A. Smith, Flux, 2020.",
            "## Appendix A\n\nSupplementary derivation of the flux equation.",
        ]
    )

    assert kinds[0] == "body"
    assert kinds[1] == "references"
    assert kinds[2] == "body"  # the appendix must remain searchable


def test_a_bibliography_mid_document_is_still_found() -> None:
    """A paper with long appendices puts its references well before the end."""
    pages = ["## References\n\n[1] A. Smith, 2020."] + [
        f"## Appendix {n}\n\nMore supplementary material about flux." for n in range(9)
    ]
    kinds = marked(pages)

    assert kinds[0] == "references"
    assert set(kinds[1:]) == {"body"}


def test_a_paper_with_no_bibliography_is_left_alone() -> None:
    assert set(marked(["## Intro\n\nOnly body text.", "## Method\n\nStill body text."])) == {"body"}


def test_section_paths_record_the_heading_hierarchy() -> None:
    from verbatim.chunking import parse_markdown_sections

    sections, stack = parse_markdown_sections(
        "# **Title**\n\nPreamble.\n\n## **1 Intro**\n\nBody.\n\n### **1.1 Setup**\n\nDetail.", []
    )

    assert [s.path for s in sections] == [
        ("Title",),
        ("Title", "1 Intro"),
        ("Title", "1 Intro", "1.1 Setup"),
    ]
    assert [level for level, _ in stack] == [1, 2, 3]


def test_a_heading_continues_onto_the_next_page() -> None:
    from verbatim.chunking import parse_markdown_sections

    first, stack = parse_markdown_sections("## **2 Method**\n\nWe begin.", [])
    second, _ = parse_markdown_sections("The method continues here.", stack)

    assert first[0].path == ("2 Method",)
    assert second[0].path == ("2 Method",)


def test_markdown_emphasis_is_stripped_from_text_and_headings() -> None:
    from verbatim.chunking import parse_markdown_sections, strip_markdown

    assert strip_markdown("**Bold** and _italic_ and `code`") == "Bold and italic and code"
    sections, _ = parse_markdown_sections("## **3 Results**\n\nWe found **flux**.", [])
    assert sections[0].path == ("3 Results",)
    assert sections[0].text == "We found flux."


def test_lone_surrogates_are_removed_so_text_can_be_stored_and_exported() -> None:
    """PDF extraction emits them; SQLite and JSON both refuse to encode them."""
    from verbatim.chunking import normalize_page_text

    contaminated = "magnetic \ud835\udc65 flux"  # a lone surrogate pair, not a real char
    with pytest.raises(UnicodeEncodeError):
        contaminated.encode("utf-8")

    cleaned = normalize_page_text(contaminated)

    assert cleaned == "magnetic flux"
    assert cleaned.encode("utf-8")  # must not raise


def test_one_unencodable_paper_does_not_discard_a_whole_chunking_batch(tmp_path: Path) -> None:
    """A surrogate used to abort the insert, rolling back every paper with it."""
    import json
    from verbatim.storage import save_chunks

    chunks = extract_pdf_chunks(
        [download(tmp_path / "test.pdf")],
        extract_pages=lambda _: ["## S\n\nvalue \ud835\udc65 here"],
    )
    destination = tmp_path / "chunks.jsonl"
    save_chunks(chunks, destination)

    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert rows and "value here" in rows[0]["text"]
