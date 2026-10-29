from pathlib import Path
from hashlib import sha256

import pytest

from verbatim.chunking import chunk_page, extract_pdf_chunks
from verbatim.downloads import DownloadedPaper
from verbatim.storage import save_chunks


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    return [span.text for span in chunk_page(text, chunk_size=chunk_size, overlap=overlap)]


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
    from verbatim.chunking import clean_markdown_text, parse_markdown_sections

    assert clean_markdown_text("**Bold** and _italic_ and `code`") == "Bold and italic and code"
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


def test_layout_extraction_is_abandoned_when_it_will_not_finish(tmp_path: Path) -> None:
    """Layout extraction that never returns is abandoned at the timeout."""
    from verbatim.chunking import ExtractionTimeout, extract_pages_from_pdf

    pdf = tmp_path / "slow.pdf"
    pdf.write_bytes(b"%PDF-1.7 not really a pdf\n%%EOF\n")

    # A timeout small enough that even a valid parse would not beat it.
    with pytest.raises(ExtractionTimeout, match="exceeded"):
        extract_pages_from_pdf(pdf, timeout=0.001)


def test_a_paper_that_cannot_be_extracted_does_not_lose_its_batch(tmp_path: Path) -> None:
    """The batch is the unit of work; one bad file must not discard the rest."""
    from verbatim.chunking import ExtractionTimeout

    first = download(tmp_path / "a.pdf")
    second = download(tmp_path / "b.pdf")
    object.__setattr__(second, "arxiv_id", "2401.99999v1")

    def extract(path: Path) -> list[str]:
        if path.name == "a.pdf":
            raise ExtractionTimeout("Layout extraction of a.pdf exceeded 180s")
        return ["## S\n\nThe second paper still chunks."]

    chunks = extract_pdf_chunks([first, second], extract_pages=extract)

    assert [chunk.arxiv_id for chunk in chunks] == ["2401.99999v1"]
    assert "second paper" in chunks[0].text


def test_an_extraction_that_crashes_says_so_without_waiting_out_the_timeout(
    tmp_path: Path,
) -> None:
    """A child that died on a corrupt file put nothing on the queue, and the
    parent waited the full timeout - three minutes by default - before calling
    it slow. A crash is reported as a crash, as soon as it happens."""
    import time

    from verbatim.chunking import ExtractionFailed, extract_pages_from_pdf

    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a pdf at all")

    started = time.monotonic()
    with pytest.raises(ExtractionFailed, match="exited"):
        extract_pages_from_pdf(broken, timeout=120)

    assert time.monotonic() - started < 60


def test_sentences_end_at_full_stops_but_not_inside_abbreviations_or_numbers() -> None:
    from verbatim.chunking import sentence_spans

    text = (
        "Transformers use attention (Vaswani et al. 2017). See Fig. 2 for details, "
        "e.g. the decoder. Accuracy rose to 93.5 percent. A. Smith disagreed! Why? "
        "Results follow."
    )

    assert [text[start:end] for start, end in sentence_spans(text)] == [
        "Transformers use attention (Vaswani et al. 2017).",
        "See Fig. 2 for details, e.g. the decoder.",
        "Accuracy rose to 93.5 percent.",
        "A. Smith disagreed!",
        "Why?",
        "Results follow.",
    ]


TWELVE_SENTENCES = [f"Sentence number {n} says something useful." for n in range(12)]


def test_sentence_windows_end_every_passage_at_a_sentence_end() -> None:
    from verbatim.chunking import chunk_sentences

    text = " ".join(TWELVE_SENTENCES)

    spans = chunk_sentences(text, chunk_size=120, overlap=50)

    assert all(span.text.endswith(".") and len(span.text) <= 120 for span in spans)
    assert all(text[span.char_start:span.char_end] == span.text for span in spans)
    # The overlap repeats the whole last sentence, which fits in 50 characters.
    assert spans[1].text.startswith(TWELVE_SENTENCES[1])
    assert all(any(sentence in span.text for span in spans) for sentence in TWELVE_SENTENCES)


def test_a_sentence_longer_than_a_passage_is_cut_at_words_as_before() -> None:
    from verbatim.chunking import chunk_sentences

    long = "This " + " ".join(["word"] * 40) + " ends."
    text = f"Short one. {long} Short two."

    spans = chunk_sentences(text, chunk_size=100, overlap=20)

    assert all(len(span.text) <= 100 for span in spans)
    assert all(text[span.char_start:span.char_end] == span.text for span in spans)
    assert spans[0].text == "Short one." and spans[-1].text.endswith("Short two.")


def test_semantic_cuts_fall_where_neighbouring_sentences_change_subject() -> None:
    """With room for either, the cut goes at the change of subject, not as late as fits."""
    from verbatim.chunking import chunk_semantic, chunk_sentences

    qubits = ["Qubits decohere in noise.", "Qubits need cold fridges.", "Qubits are fragile."]
    proteins = ["Proteins fold quickly.", "Proteins bind ligands.", "Proteins misfold sometimes."]
    text = " ".join(qubits + proteins)

    def by_subject(texts):
        return [[1.0, 0.0] if "Qubit" in t else [0.0, 1.0] for t in texts]

    semantic = chunk_semantic(text, chunk_size=110, overlap=0, embed=by_subject)
    packed = chunk_sentences(text, chunk_size=110, overlap=0)

    assert [span.text for span in semantic] == [" ".join(qubits), " ".join(proteins)]
    assert all(text[span.char_start:span.char_end] == span.text for span in semantic)
    # Packing as many sentences as fit mixes the two subjects instead.
    assert packed[0].text == " ".join(qubits + proteins[:1])


def test_the_way_passages_are_cut_is_part_of_the_chunker_fingerprint(monkeypatch) -> None:
    """Switching it must re-chunk every paper, or two kinds of passage would mix."""
    from verbatim import config
    from verbatim.chunking import CHUNKER_VERSION, chunker_fingerprint

    default = chunker_fingerprint(1200, 200)
    monkeypatch.setattr(config, "CHUNKER", "sentences")

    # Unchanged by default, so the corpus already built is not cut again.
    assert default == f"{CHUNKER_VERSION}:1200:200"
    assert chunker_fingerprint(1200, 200) == f"{default}:sentences"


def test_the_chosen_way_of_cutting_reaches_the_pdf_extractor(tmp_path: Path, monkeypatch) -> None:
    from verbatim import config

    monkeypatch.setattr(config, "CHUNKER", "sentences")
    paper = download(tmp_path / "test.pdf")

    chunks = extract_pdf_chunks(
        [paper], chunk_size=120, overlap=50,
        extract_pages=lambda _: [" ".join(TWELVE_SENTENCES)],
    )

    assert len(chunks) == 11
    assert all(chunk.text.endswith(".") for chunk in chunks)
    assert all(chunk.chunker_version.endswith(":sentences") for chunk in chunks)


def test_an_unknown_way_of_cutting_names_the_setting(monkeypatch) -> None:
    from verbatim import config
    from verbatim.chunking import section_chunker

    monkeypatch.setattr(config, "CHUNKER", "paragraphs")

    with pytest.raises(ValueError, match="VERBATIM_CHUNKER"):
        section_chunker(1200, 200)
