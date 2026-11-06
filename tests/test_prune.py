"""Deleting PDFs a fully ingested paper no longer needs, and fetching them back."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from helpers import downloaded, paper
from verbatim.chunking import extract_pdf_chunks
from verbatim.corpus import CorpusStore, EmbeddingModel
from verbatim.html_source import HtmlSource
from verbatim.pruning import prune_pdfs, restore_pdfs

FINGERPRINT = "8:1200:200"
MODEL = EmbeddingModel(model_id="stub-model", dimensions=4, normalized=True, revision="r1")
PAGES = ["Attention weighs every token against every other.", "Then it sums them."]


def ingest(
    corpus: CorpusStore,
    tmp_path: Path,
    arxiv_id: str = "2611.00001v1",
    *,
    skip: str | None = None,
    fingerprint: str = FINGERPRINT,
    model: EmbeddingModel = MODEL,
):
    """Store one paper as the pipeline leaves it, without the step `skip` names."""
    corpus.upsert_papers([paper(arxiv_id)])
    record = downloaded(tmp_path, arxiv_id)
    corpus.record_downloads([record])
    chunks = extract_pdf_chunks([record], extract_pages=lambda _: PAGES)
    assert len(chunks) == 2
    if skip != "passages":
        corpus.record_chunks(chunks)
    if skip == "checked":
        return record
    html = tmp_path / f"{arxiv_id}.html"
    html.write_text("<html><body><p>Attention.</p></body></html>")
    source = HtmlSource(
        arxiv_id=arxiv_id, url=f"https://arxiv.org/html/{arxiv_id}",
        local_path=str(html), sha256="h" * 64, byte_count=html.stat().st_size,
    )
    truncated = skip == "usable"
    corpus.record_html_source(
        source, usable=not truncated,
        note="rendering covers 40% of the PDF text" if truncated else None,
    )
    corpus.mark_html_chunked([source], "7:1200:200" if skip == "chunked" else fingerprint)
    corpus.register_embedding_model(model)
    if skip != "passages":
        # One passage short of a full set when the step skipped is embedding.
        embedded = chunks[:1] if skip == "embedded" else chunks
        corpus.record_embeddings(
            model.model_id, [(chunk.chunk_id, [1.0, 0.0, 0.0, 0.0]) for chunk in embedded]
        )
    if skip != "read":
        corpus.record_references([], source.sha256, for_papers=[arxiv_id])
    return record


def counts(corpus: CorpusStore) -> dict[str, int]:
    return corpus.pipeline_stats(FINGERPRINT, MODEL.model_id)


def test_a_fully_ingested_papers_pdf_is_deleted_and_its_record_kept(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        record = ingest(corpus, tmp_path)
        report = prune_pdfs(corpus, FINGERPRINT, MODEL.model_id)
        stats = counts(corpus)
        kept = corpus.download_for(record.arxiv_id)

    assert not Path(record.local_path).exists()
    assert report.deleted == 1 and report.bytes_freed == record.byte_count
    # The URL and checksum a restore needs stay behind.
    assert kept is not None and kept.sha256 == record.sha256
    assert stats["pdfs_pruned"] == 1 and stats["bytes_pruned"] == record.byte_count
    assert stats["bytes_on_disk"] == 0
    # A deleted PDF is not fetched again.
    assert stats["downloads_pending"] == 0


@pytest.mark.parametrize(
    ("skip", "why"),
    [
        ("checked", "its rendering has not been compared with the PDF yet"),
        ("usable", "its rendering was truncated, so the PDF is the source"),
        ("chunked", "its passages were cut by an older chunker"),
        ("embedded", "a passage has no vector yet"),
        ("read", "its bibliography has not been read"),
        ("passages", "it has no passages at all"),
    ],
)
def test_a_pdf_stays_until_its_paper_is_fully_ingested(tmp_path: Path, skip, why) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        record = ingest(corpus, tmp_path, skip=skip)
        report = prune_pdfs(corpus, FINGERPRINT, MODEL.model_id)
        prunable = corpus.prunable_pdf_count(FINGERPRINT, MODEL.model_id)

    assert Path(record.local_path).exists(), why
    assert report.deleted == 0 and prunable == 0, why


def test_the_rendering_passages_and_vectors_stay(tmp_path: Path) -> None:
    """Answers read only the database, and re-chunking reads the rendering."""
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        record = ingest(corpus, tmp_path)
        before = counts(corpus)
        prune_pdfs(corpus, FINGERPRINT, MODEL.model_id)
        after = counts(corpus)

    assert (tmp_path / f"{record.arxiv_id}.html").exists()
    assert after["chunks"] == before["chunks"] == 2
    assert after["vectors"] == before["vectors"] == 2


def test_the_count_of_deletable_pdfs_matches_what_a_prune_deletes(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        ingest(corpus, tmp_path, "2611.00001v1")
        ingest(corpus, tmp_path, "2611.00002v1", skip="read")
        counted = corpus.prunable_pdf_count(FINGERPRINT, MODEL.model_id)
        report = prune_pdfs(corpus, FINGERPRINT, MODEL.model_id)
        left = corpus.prunable_pdf_count(FINGERPRINT, MODEL.model_id)

    assert counted == report.deleted == 1
    assert left == 0


def test_a_dry_run_deletes_nothing_and_says_what_it_would_free(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        record = ingest(corpus, tmp_path)
        report = prune_pdfs(corpus, FINGERPRINT, MODEL.model_id, dry_run=True)
        stats = counts(corpus)

    assert Path(record.local_path).exists()
    assert report.deleted == 1 and report.bytes_freed == record.byte_count
    assert stats["pdfs_pruned"] == 0


def test_a_pdf_already_deleted_by_hand_is_recorded_as_gone(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        record = ingest(corpus, tmp_path)
        Path(record.local_path).unlink()
        report = prune_pdfs(corpus, FINGERPRINT, MODEL.model_id)
        stats = counts(corpus)

    assert report.deleted == 0 and report.already_gone == 1
    assert stats["pdfs_pruned"] == 1


def test_only_a_pdf_is_ever_deleted(tmp_path: Path) -> None:
    """The path comes from the database; nothing but a .pdf is removed through it."""
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        record = ingest(corpus, tmp_path)
        notes = tmp_path / "notes.txt"
        notes.write_text("keep me")
        corpus.record_downloads([replace(record, local_path=str(notes))])
        report = prune_pdfs(corpus, FINGERPRINT, MODEL.model_id)
        stats = counts(corpus)

    assert notes.exists()
    assert report.refused == [record.arxiv_id] and report.deleted == 0
    assert stats["pdfs_pruned"] == 0


def test_a_second_prune_has_nothing_to_do(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        ingest(corpus, tmp_path)
        prune_pdfs(corpus, FINGERPRINT, MODEL.model_id)
        again = prune_pdfs(corpus, FINGERPRINT, MODEL.model_id)

    assert again.deleted == again.already_gone == 0


def test_a_limit_bounds_how_many_are_deleted(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        for number in range(1, 4):
            ingest(corpus, tmp_path, f"2611.0000{number}v1")
        report = prune_pdfs(corpus, FINGERPRINT, MODEL.model_id, limit=2)
        left = corpus.prunable_pdf_count(FINGERPRINT, MODEL.model_id)

    assert report.deleted == 2 and left == 1


def test_restore_fetches_the_pdf_back_and_checks_it_against_the_deleted_one(
    tmp_path: Path,
) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        record = ingest(corpus, tmp_path)
        original = Path(record.local_path).read_bytes()
        prune_pdfs(corpus, FINGERPRINT, MODEL.model_id)
        report = restore_pdfs(corpus, fetch_pdf=lambda url: original, sleep=lambda _: None)
        stats = counts(corpus)

    assert Path(record.local_path).read_bytes() == original
    assert report.restored == 1 and report.changed == 0 and report.failed == []
    assert stats["pdfs_pruned"] == 0 and stats["bytes_on_disk"] == record.byte_count


def test_restore_says_when_arxiv_serves_different_bytes(tmp_path: Path) -> None:
    """arXiv can rebuild a version's PDF; the new file is kept and the change reported."""
    regenerated = b"%PDF-1.7 rebuilt by a newer TeX\n%%EOF\n"
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        record = ingest(corpus, tmp_path)
        prune_pdfs(corpus, FINGERPRINT, MODEL.model_id)
        report = restore_pdfs(corpus, fetch_pdf=lambda url: regenerated, sleep=lambda _: None)
        now = corpus.download_for(record.arxiv_id)

    assert report.restored == 1 and report.changed == 1
    assert Path(record.local_path).read_bytes() == regenerated
    assert now is not None and now.sha256 != record.sha256


def test_a_failed_restore_keeps_the_pdf_recorded_as_deleted(tmp_path: Path) -> None:
    def offline(url: str) -> bytes:
        raise OSError("network is unreachable")

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        record = ingest(corpus, tmp_path)
        prune_pdfs(corpus, FINGERPRINT, MODEL.model_id)
        report = restore_pdfs(corpus, fetch_pdf=offline, sleep=lambda _: None)
        stats = counts(corpus)

    assert report.restored == 0
    assert [arxiv_id for arxiv_id, _ in report.failed] == [record.arxiv_id]
    assert stats["pdfs_pruned"] == 1


def test_restore_paces_its_requests(tmp_path: Path) -> None:
    pauses: list[float] = []
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        records = [ingest(corpus, tmp_path, f"2611.0000{n}v1") for n in (1, 2)]
        contents = {r.pdf_url: Path(r.local_path).read_bytes() for r in records}
        prune_pdfs(corpus, FINGERPRINT, MODEL.model_id)
        restore_pdfs(corpus, fetch_pdf=contents.__getitem__, delay=3.0, sleep=pauses.append)

    assert pauses == [3.0]
