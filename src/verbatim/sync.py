"""Paging through an arXiv category into the corpus, resumably."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

from verbatim.arxiv import FeedPage, Paper, fetch_metadata_page
from verbatim.corpus import CorpusStore, SyncSummary

# arXiv asks callers to leave a gap between requests; this is the default pause.
REQUEST_DELAY = 3.0
PAGE_SIZE = 100


@dataclass(frozen=True)
class SyncReport:
    """What one sync run did, and why it stopped."""

    category: str
    fetched: int
    inserted: int
    updated: int
    requests: int
    total_available: int
    stopped_because: str

    @property
    def caught_up(self) -> bool:
        return self.stopped_because in {"reached the watermark", "exhausted the category"}


def sync_category(
    corpus: CorpusStore,
    category: str,
    *,
    limit: int = 1_000,
    page_size: int = PAGE_SIZE,
    delay: float = REQUEST_DELAY,
    since: str | None = None,
    full: bool = False,
    fetch_page: Callable[..., FeedPage] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_page: Callable[[int, int], None] | None = None,
) -> SyncReport:
    """Page through a category, newest first, stopping once already covered.

    Results are sorted by last-updated so a revision resurfaces as well as a
    new paper.  Paging stops at the stored watermark, which is what makes a
    repeat sync cost one request instead of walking the category again; `full`
    ignores the watermark and re-walks everything.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if page_size < 1:
        raise ValueError("page_size must be at least 1")

    page_fetcher = fetch_page or fetch_metadata_page
    watermark = None if full else (since or corpus.sync_watermark(category))
    fetched = inserted = updated = requests = 0
    total_available = 0
    newest_seen: str | None = None
    stopped = "reached the limit"
    start = 0

    while fetched < limit:
        if requests:
            sleep(delay)
        try:
            page = page_fetcher(
                category,
                start=start,
                page_size=min(page_size, limit - fetched),
            )
        except (OSError, RuntimeError, ValueError) as error:
            # Stop cleanly rather than raising: papers from earlier pages are
            # already committed, and recording the watermark below means the
            # next run resumes instead of walking the category again.
            stopped = f"failed after {requests} request(s): {error}"
            break
        requests += 1
        total_available = page.total_results or total_available
        if not page.papers:
            stopped = "exhausted the category"
            break

        fresh: list[Paper] = []
        for paper in page.papers:
            if watermark is not None and paper.updated <= watermark:
                stopped = "reached the watermark"
                break
            fresh.append(paper)

        if fresh:
            summary = corpus.upsert_papers(fresh)
            inserted += summary.inserted
            updated += summary.updated
            fetched += len(fresh)
            newest_seen = max(newest_seen or "", max(paper.updated for paper in fresh))
            if on_page is not None:
                on_page(fetched, min(limit, total_available or limit))

        if stopped == "reached the watermark":
            break
        start += len(page.papers)
        if total_available and start >= total_available:
            stopped = "exhausted the category"
            break

    corpus.record_sync(category, newest_seen, fetched)
    return SyncReport(
        category=category,
        fetched=fetched,
        inserted=inserted,
        updated=updated,
        requests=requests,
        total_available=total_available,
        stopped_because=stopped,
    )


def format_sync_report(report: SyncReport) -> str:
    return (
        f"{report.category}: {report.fetched} papers "
        f"({report.inserted} new, {report.updated} refreshed) "
        f"in {report.requests} request(s); stopped after it {report.stopped_because}"
        + (f"; {report.total_available:,} exist in this category" if report.total_available else "")
    )


__all__ = ["SyncReport", "SyncSummary", "format_sync_report", "sync_category"]
