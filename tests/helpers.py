"""Shared builders for tests that need a populated corpus."""

from __future__ import annotations

from pathlib import Path

from verbatim.arxiv import Paper
from verbatim.chunking import extract_pdf_chunks
from verbatim.corpus import CorpusStore
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


def downloaded(tmp_path: Path, arxiv_id: str) -> DownloadedPaper:
    pdf = tmp_path / f"{arxiv_id.replace('/', '-')}.pdf"
    pdf.write_bytes(f"%PDF-1.7 {arxiv_id}\n%%EOF\n".encode())
    return DownloadedPaper(
        arxiv_id=arxiv_id,
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        license_url=None,
        local_path=str(pdf),
        sha256=file_sha256(pdf),
        byte_count=pdf.stat().st_size,
    )


def chunked_corpus(tmp_path: Path, pages_by_paper: dict[str, list[str]]) -> CorpusStore:
    """Open a corpus already populated with papers, PDFs, and chunks."""
    corpus = CorpusStore(tmp_path / "corpus.db")
    for arxiv_id, pages in pages_by_paper.items():
        record = downloaded(tmp_path, arxiv_id)
        corpus.upsert_papers([paper(arxiv_id)])
        corpus.record_downloads([record])
        corpus.record_chunks(
            extract_pdf_chunks([record], extract_pages=lambda _, captured=pages: captured)
        )
    return corpus


def store_paper(corpus: CorpusStore, tmp_path: Path, arxiv_id: str) -> DownloadedPaper:
    """Store one paper and its PDF, so it is eligible for HTML fetching."""
    record = downloaded(tmp_path, arxiv_id)
    corpus.upsert_papers([paper(arxiv_id)])
    corpus.record_downloads([record])
    return record
