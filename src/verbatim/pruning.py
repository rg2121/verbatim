"""Deleting the PDFs fully ingested papers no longer need, and fetching them back.

A paper with a usable HTML rendering is chunked from the rendering; its PDF is
read once, to check the rendering is not truncated. After that check, and once
the passages are cut by the current chunker, embedded and the bibliography read,
nothing reads the PDF again: answers read only the database. The download
record stays, with the URL and checksum, so a restore can fetch the PDF again
and say whether arXiv served the same bytes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import time

from verbatim import config
from verbatim.corpus import CorpusStore
from verbatim.downloads import download_papers

# Deletions are recorded in batches, so an interrupted prune loses little.
RECORD_EVERY = 500


@dataclass
class PruneReport:
    deleted: int = 0
    bytes_freed: int = 0
    # Gone before the prune reached them; recorded all the same.
    already_gone: int = 0
    # Recorded paths that are not a .pdf, which a prune never deletes.
    refused: list[str] = field(default_factory=list)


@dataclass
class RestoreReport:
    restored: int = 0
    # arXiv can rebuild a version's PDF, so the bytes may differ from the deleted ones.
    changed: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)


def prune_pdfs(
    corpus: CorpusStore,
    chunker_fingerprint: str,
    model_id: str,
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> PruneReport:
    """Delete every PDF nothing reads again (`CorpusStore.prunable_pdfs`) and record it."""
    report = PruneReport()
    settled: list[str] = []
    for download in corpus.prunable_pdfs(chunker_fingerprint, model_id, limit=limit):
        path = Path(download.local_path)
        if path.suffix.lower() != ".pdf":
            report.refused.append(download.arxiv_id)
            continue
        if dry_run:
            report.deleted += 1
            report.bytes_freed += download.byte_count
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            report.already_gone += 1
        else:
            report.deleted += 1
            report.bytes_freed += download.byte_count
        settled.append(download.arxiv_id)
        if len(settled) >= RECORD_EVERY:
            corpus.record_pruned(settled)
            settled = []
    if settled:
        corpus.record_pruned(settled)
    return report


def restore_pdfs(
    corpus: CorpusStore,
    *,
    limit: int | None = None,
    fetch_pdf: Callable[[str], bytes] | None = None,
    delay: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> RestoreReport:
    """Download deleted PDFs again, to the path each had, pausing between requests."""
    report = RestoreReport()
    pause = config.DOWNLOAD_DELAY if delay is None else delay
    pruned = corpus.pruned_downloads(limit=limit)
    papers = corpus.papers_by_id([download.arxiv_id for download in pruned])
    for position, download in enumerate(pruned):
        paper = papers.get(download.arxiv_id)
        if paper is None:
            report.failed.append((download.arxiv_id, "the paper is no longer in the corpus"))
            continue
        if position:
            sleep(pause)
        fetched = download_papers(
            [paper], Path(download.local_path).parent, 1, fetch_pdf=fetch_pdf, delay=0,
        )
        if fetched.failed:
            report.failed.append((download.arxiv_id, fetched.failed[0].reason))
            continue
        record = fetched.downloaded[0]
        corpus.record_downloads([record])
        corpus.forget_pruned([download.arxiv_id])
        report.restored += 1
        report.changed += record.sha256 != download.sha256
    return report


def describe_prune(report: PruneReport, *, dry_run: bool) -> str:
    verb = "Would delete" if dry_run else "Deleted"
    noun = "PDF" if report.deleted == 1 else "PDFs"
    lines = [
        f"{verb} {report.deleted:,} {noun} ({report.bytes_freed / 1e9:.1f} GB); their "
        "HTML renderings, passages and vectors stay."
    ]
    if report.already_gone:
        lines.append(
            f"  {report.already_gone:,} were already gone and are now recorded as deleted."
        )
    if report.refused:
        lines.append(
            f"  ! {len(report.refused):,} recorded path(s) are not a .pdf and were left alone,"
            f" e.g. {report.refused[0]}."
        )
    return "\n".join(lines)


def describe_restore(report: RestoreReport) -> str:
    lines = [f"Restored {report.restored:,} PDFs."]
    if report.changed:
        lines.append(
            f"  {report.changed:,} differ from the deleted file: arXiv rebuilt them. "
            "The new checksum is recorded."
        )
    for arxiv_id, reason in report.failed[:10]:
        lines.append(f"  ! {arxiv_id}: {reason}")
    if report.failed:
        lines.append(
            f"  {len(report.failed):,} failed; they stay recorded as deleted. Re-run to retry."
        )
    return "\n".join(lines)
