from hashlib import sha256
from pathlib import Path

from verbatim.arxiv import Paper
from verbatim.chunking import chunker_fingerprint, extract_pdf_chunks
from verbatim.corpus import CorpusStore, split_arxiv_version
from verbatim.downloads import DownloadedPaper, file_sha256


def paper(arxiv_id: str, *, updated: str = "2026-08-08T10:00:00Z") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        summary="A test abstract.",
        authors=["Ada Lovelace"],
        categories=["physics.gen-ph"],
        published="2026-08-01T10:00:00Z",
        updated=updated,
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        license_url=None,
    )


def download(arxiv_id: str) -> DownloadedPaper:
    content = f"PDF {arxiv_id}".encode()
    return DownloadedPaper(
        arxiv_id=arxiv_id,
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        license_url=None,
        local_path=f"data/pdfs/{arxiv_id}.pdf",
        sha256=sha256(content).hexdigest(),
        byte_count=len(content),
    )


def test_split_arxiv_version_supports_modern_and_legacy_ids() -> None:
    assert split_arxiv_version("2608.12345v2") == ("2608.12345", 2)
    assert split_arxiv_version("math/0309136v1") == ("math/0309136", 1)


def test_corpus_accumulates_versions_and_selects_only_current_undownloaded_papers(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        first = corpus.upsert_papers([paper("2608.12345v1"), paper("2608.33333v1")])
        corpus.record_downloads([download("2608.12345v1")])
        second = corpus.upsert_papers([paper("2608.12345v2", updated="2026-08-09T10:00:00Z")])

        pending = corpus.pending_downloads(10)
        all_downloads = corpus.all_downloads()

    assert (first.inserted, first.updated) == (2, 0)
    assert (second.inserted, second.updated) == (1, 0)
    assert [item.arxiv_id for item in pending] == ["2608.12345v2", "2608.33333v1"]
    assert [item.arxiv_id for item in all_downloads] == ["2608.12345v1"]


def test_corpus_stores_chunks_and_replaces_them_when_a_pdf_is_rechunked(tmp_path: Path) -> None:
    pdf = tmp_path / "2608.12345v1.pdf"
    pdf.write_bytes(b"%PDF-1.7 body\n%%EOF\n")
    record = DownloadedPaper(
        arxiv_id="2608.12345v1",
        source_url="https://arxiv.org/abs/2608.12345v1",
        pdf_url="https://arxiv.org/pdf/2608.12345v1",
        license_url=None,
        local_path=str(pdf),
        sha256=file_sha256(pdf),
        byte_count=pdf.stat().st_size,
    )

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        corpus.upsert_papers([paper("2608.12345v1")])
        corpus.record_downloads([record])

        first = extract_pdf_chunks([record], extract_pages=lambda _: ["alpha beta gamma"])
        corpus.record_chunks(first)
        after_first = corpus.chunk_count()

        second = extract_pdf_chunks([record], extract_pages=lambda _: ["alpha beta gamma"])
        corpus.record_chunks(second)

        stored = corpus.all_chunks()

    assert after_first == 1
    assert corpus_chunk_texts(stored) == ["alpha beta gamma"]
    assert stored[0].char_start == 0 and stored[0].char_end == len("alpha beta gamma")
    assert stored[0].chunker_version == chunker_fingerprint(1_200, 200)
    assert stored[0].kind == "body"


def corpus_chunk_texts(chunks: list) -> list[str]:
    return [chunk.text for chunk in chunks]


def test_corpus_only_rechunks_pdfs_whose_chunker_or_bytes_changed(tmp_path: Path) -> None:
    pdf = tmp_path / "2608.12345v1.pdf"
    pdf.write_bytes(b"%PDF-1.7 body\n%%EOF\n")
    record = DownloadedPaper(
        arxiv_id="2608.12345v1",
        source_url="https://arxiv.org/abs/2608.12345v1",
        pdf_url="https://arxiv.org/pdf/2608.12345v1",
        license_url=None,
        local_path=str(pdf),
        sha256=file_sha256(pdf),
        byte_count=pdf.stat().st_size,
    )
    fingerprint = chunker_fingerprint(1_200, 200)

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        corpus.upsert_papers([paper("2608.12345v1")])
        corpus.record_downloads([record])

        first_pass = corpus.pending_chunking(fingerprint)
        corpus.record_chunks(extract_pdf_chunks(first_pass, extract_pages=lambda _: ["alpha beta"]))
        corpus.mark_chunked(first_pass, fingerprint)

        second_pass = corpus.pending_chunking(fingerprint)
        different_parameters = corpus.pending_chunking(chunker_fingerprint(400, 50))

    assert [item.arxiv_id for item in first_pass] == ["2608.12345v1"]
    assert second_pass == []
    assert [item.arxiv_id for item in different_parameters] == ["2608.12345v1"]


def test_corpus_marks_chunking_done_even_when_a_pdf_yields_no_text(tmp_path: Path) -> None:
    pdf = tmp_path / "2608.99999v1.pdf"
    pdf.write_bytes(b"%PDF-1.7 scanned\n%%EOF\n")
    record = DownloadedPaper(
        arxiv_id="2608.99999v1",
        source_url="https://arxiv.org/abs/2608.99999v1",
        pdf_url="https://arxiv.org/pdf/2608.99999v1",
        license_url=None,
        local_path=str(pdf),
        sha256=file_sha256(pdf),
        byte_count=pdf.stat().st_size,
    )
    fingerprint = chunker_fingerprint(1_200, 200)

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        corpus.upsert_papers([paper("2608.99999v1")])
        corpus.record_downloads([record])
        pending = corpus.pending_chunking(fingerprint)
        corpus.record_chunks(extract_pdf_chunks(pending, extract_pages=lambda _: ["   "]))
        corpus.mark_chunked(pending, fingerprint)

        retried = corpus.pending_chunking(fingerprint)
        stored = corpus.chunk_count()

    assert stored == 0
    assert retried == []


def test_corpus_refreshes_existing_metadata_without_creating_a_duplicate(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        corpus.upsert_papers([paper("2608.12345v1")])
        summary = corpus.upsert_papers([paper("2608.12345v1", updated="2026-08-10T10:00:00Z")])
        pending = corpus.pending_downloads(1)

    assert (summary.inserted, summary.updated) == (0, 1)
    assert pending[0].updated == "2026-08-10T10:00:00Z"


def test_license_backfill_records_absence_so_it_is_not_retried(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        corpus.upsert_papers([paper("2608.12345v1"), paper("2608.12345v2"), paper("2608.99999v1")])

        pending = corpus.base_ids_missing_license(10)
        corpus.record_license("2608.12345", "http://arxiv.org/licenses/nonexclusive-distrib/1.0/")
        corpus.record_license("2608.99999", None)  # genuinely has none

        remaining = corpus.base_ids_missing_license(10)
        both_versions = [corpus.get_paper(f"2608.12345v{n}") for n in (1, 2)]

    assert sorted(pending) == ["2608.12345", "2608.99999"]
    assert remaining == []  # absence recorded, so neither is looked up again
    assert all(p is not None and p.license_url is not None for p in both_versions)


def test_downloads_can_be_gated_on_what_a_licence_permits(tmp_path: Path) -> None:
    """Having a licence is not the same as being allowed to reuse the paper."""
    from verbatim.arxiv import Paper

    def licensed(arxiv_id: str, url: str | None) -> Paper:
        return Paper(**{**paper(arxiv_id).to_dict(), "license_url": url})

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        corpus.upsert_papers([
            licensed("2608.00001v1", "http://creativecommons.org/licenses/by/4.0/"),
            licensed("2608.00002v1", "http://creativecommons.org/licenses/by-nc-nd/4.0/"),
            licensed("2608.00003v1", "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"),
            licensed("2608.00004v1", None),
        ])

        ungated = corpus.pending_downloads(10)
        reported = corpus.pending_downloads(10, require_license=True)
        open_only = corpus.pending_downloads(10, license_classes=["open"])

    assert len(ungated) == 4
    assert len(reported) == 3            # any licence at all, including nonexclusive
    assert [p.arxiv_id for p in open_only] == ["2608.00001v1"]


def downloaded(corpus: CorpusStore, tmp_path: Path, arxiv_id: str) -> DownloadedPaper:
    """Store one paper with a PDF, so it is eligible for HTML fetching."""
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
    corpus.upsert_papers([paper(arxiv_id)])
    corpus.record_downloads([record])
    return record


def test_a_paper_with_no_html_rendering_is_not_asked_for_twice(tmp_path: Path) -> None:
    """Most papers have a rendering; the rest must not be re-fetched forever."""
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        downloaded(corpus, tmp_path, "2608.11111v1")
        assert corpus.papers_without_html(10) == ["2608.11111v1"]

        corpus.record_html_absent("2608.11111v1", "https://arxiv.org/html/2608.11111v1")

        assert corpus.papers_without_html(10) == []


def test_a_paper_ruled_out_is_never_offered_for_chunking(tmp_path: Path) -> None:
    """It has no bytes to chunk, so it must not reach the HTML chunker."""
    from verbatim.html_source import HtmlSource

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        downloaded(corpus, tmp_path, "2608.11111v1")
        downloaded(corpus, tmp_path, "2608.22222v1")
        corpus.record_html_absent("2608.11111v1", "https://arxiv.org/html/2608.11111v1")
        html = tmp_path / "2608.22222v1.html"
        html.write_text("<html><body><p>Body.</p></body></html>", encoding="utf-8")
        corpus.record_html_source(
            HtmlSource(
                arxiv_id="2608.22222v1",
                url="https://arxiv.org/html/2608.22222v1",
                local_path=str(html),
                sha256=file_sha256(html),
                byte_count=html.stat().st_size,
            )
        )

        pending = corpus.html_sources_pending_chunking(chunker_fingerprint(1200, 200))

    assert [source.arxiv_id for source in pending] == ["2608.22222v1"]
