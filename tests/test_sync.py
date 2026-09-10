from pathlib import Path

import pytest

from verbatim.arxiv import FeedPage
from verbatim.corpus import CorpusStore
from verbatim.sync import sync_category

from helpers import paper


class FakeArxiv:
    """A category of N papers, served in pages, newest `updated` first."""

    def __init__(self, count: int) -> None:
        # Descending timestamps, so index 0 is the most recently updated.
        self.papers = [
            paper(f"2608.{n:05d}v1", updated=f"2026-08-{(count - n) % 28 + 1:02d}T00:00:00Z")
            for n in range(count)
        ]
        self.papers.sort(key=lambda p: p.updated, reverse=True)
        self.requests: list[tuple[int, int]] = []

    def __call__(self, category: str, *, start: int = 0, page_size: int = 100, **kwargs) -> FeedPage:
        self.requests.append((start, page_size))
        return FeedPage(
            papers=self.papers[start : start + page_size],
            total_results=len(self.papers),
            start=start,
        )

    def publish(self, arxiv_id: str, updated: str) -> None:
        """Add a newer paper, as arXiv would."""
        self.papers.insert(0, paper(arxiv_id, updated=updated))


def no_sleep(_seconds: float) -> None:
    return None


def test_sync_pages_past_a_single_request(tmp_path: Path) -> None:
    feed = FakeArxiv(250)
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        report = sync_category(
            corpus, "cs.CL", limit=250, page_size=100, fetch_page=feed, sleep=no_sleep
        )
        stored = corpus.pipeline_stats("f", "m")["papers"]

    assert report.requests == 3  # 100 + 100 + 50
    assert report.fetched == 250
    assert stored == 250
    assert feed.requests == [(0, 100), (100, 100), (200, 50)]


def test_a_repeat_sync_stops_at_the_watermark(tmp_path: Path) -> None:
    """The point of the watermark: re-syncing must not re-walk the category."""
    feed = FakeArxiv(250)
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        sync_category(corpus, "cs.CL", limit=250, page_size=100, fetch_page=feed, sleep=no_sleep)
        feed.requests.clear()
        second = sync_category(
            corpus, "cs.CL", limit=250, page_size=100, fetch_page=feed, sleep=no_sleep
        )

    assert second.fetched == 0
    assert second.requests == 1  # one request to discover there is nothing new
    assert second.stopped_because == "reached the watermark"
    assert second.caught_up


def test_only_papers_newer_than_the_watermark_are_ingested(tmp_path: Path) -> None:
    feed = FakeArxiv(50)
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        sync_category(corpus, "cs.CL", limit=50, page_size=50, fetch_page=feed, sleep=no_sleep)
        feed.publish("2608.99999v1", updated="2026-09-01T00:00:00Z")
        second = sync_category(
            corpus, "cs.CL", limit=50, page_size=50, fetch_page=feed, sleep=no_sleep
        )

    assert second.fetched == 1
    assert second.inserted == 1
    assert second.stopped_because == "reached the watermark"


def test_full_ignores_the_watermark(tmp_path: Path) -> None:
    feed = FakeArxiv(120)
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        sync_category(corpus, "cs.CL", limit=120, page_size=60, fetch_page=feed, sleep=no_sleep)
        again = sync_category(
            corpus, "cs.CL", limit=120, page_size=60, fetch_page=feed, sleep=no_sleep, full=True
        )

    assert again.fetched == 120
    assert again.inserted == 0  # already known; refreshed rather than inserted
    assert again.updated == 120


def test_sync_stops_when_the_category_runs_out(tmp_path: Path) -> None:
    feed = FakeArxiv(30)
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        report = sync_category(
            corpus, "cs.CL", limit=1_000, page_size=100, fetch_page=feed, sleep=no_sleep
        )

    assert report.fetched == 30
    assert report.stopped_because == "exhausted the category"
    assert report.total_available == 30


def test_a_limited_run_marks_what_it_took_and_leaves_the_rest_behind(
    tmp_path: Path,
) -> None:
    """A deliberate limit is a slice of the newest papers, not a pause.

    This test used to be called "the next run resumes" and never ran a second
    sync. The name was also wrong: papers arrive newest-first and the mark only
    moves forward, so everything older than a limited run is behind the mark
    for good and a later, larger run fetches nothing. `--full` is what reaches
    it. That is a real property of the design and is documented in the README;
    it is asserted here so the name cannot drift from the behaviour again.
    """
    feed = FakeArxiv(200)
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        first = sync_category(
            corpus, "cs.CL", limit=50, page_size=50, fetch_page=feed, sleep=no_sleep
        )
        watermark = corpus.sync_watermark("cs.CL")
        # A later run asking for far more gets nothing: the newest paper it
        # sees is already at the mark.
        second = sync_category(
            corpus, "cs.CL", limit=200, page_size=50, fetch_page=feed, sleep=no_sleep
        )
        # Only --full reaches the 150 papers the first run did not take.
        third = sync_category(
            corpus, "cs.CL", limit=200, page_size=50, fetch_page=feed,
            sleep=no_sleep, full=True,
        )
        stored = corpus.pipeline_stats("f", "m")["papers"]

    assert first.fetched == 50 and first.stopped_because == "reached the limit"
    assert watermark is not None
    assert second.fetched == 0 and second.stopped_because == "reached the watermark"
    assert third.fetched == 200
    assert stored == 200


def test_the_watermark_never_moves_backwards(tmp_path: Path) -> None:
    """A later, smaller run must not lower the mark and cause refetching."""
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        corpus.record_sync("cs.CL", "2026-08-20T00:00:00Z", 10)
        corpus.record_sync("cs.CL", "2026-08-01T00:00:00Z", 5)

        assert corpus.sync_watermark("cs.CL") == "2026-08-20T00:00:00Z"
        assert corpus.sync_summary() == [("cs.CL", "2026-08-20T00:00:00Z", 15)]


def test_a_run_that_finds_nothing_leaves_the_watermark_intact(tmp_path: Path) -> None:
    feed = FakeArxiv(20)
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        sync_category(corpus, "cs.CL", limit=20, page_size=20, fetch_page=feed, sleep=no_sleep)
        before = corpus.sync_watermark("cs.CL")
        sync_category(corpus, "cs.CL", limit=20, page_size=20, fetch_page=feed, sleep=no_sleep)

        assert corpus.sync_watermark("cs.CL") == before


def test_sync_paces_its_requests(tmp_path: Path) -> None:
    feed = FakeArxiv(300)
    pauses: list[float] = []
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        sync_category(
            corpus,
            "cs.CL",
            limit=300,
            page_size=100,
            delay=3.0,
            fetch_page=feed,
            sleep=pauses.append,
        )

    assert pauses == [3.0, 3.0]  # between requests, not before the first


@pytest.mark.parametrize(("limit", "page_size"), [(0, 10), (10, 0)])
def test_sync_rejects_nonsense_paging(tmp_path: Path, limit: int, page_size: int) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        with pytest.raises(ValueError):
            sync_category(corpus, "cs.CL", limit=limit, page_size=page_size)


def test_a_failing_page_keeps_the_papers_already_ingested(tmp_path: Path) -> None:
    """A long ingest must not lose committed work when arXiv fails mid-run."""
    feed = FakeArxiv(300)
    calls = {"n": 0}

    def flaky(category, *, start=0, page_size=100, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("503 Service Unavailable")
        return feed(category, start=start, page_size=page_size)

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        report = sync_category(
            corpus, "cs.CL", limit=300, page_size=100, fetch_page=flaky, sleep=no_sleep
        )
        stored = corpus.pipeline_stats("f", "m")["papers"]
        watermark = corpus.sync_watermark("cs.CL")

    assert report.fetched == 200  # two pages committed before the failure
    assert stored == 200
    assert "failed after" in report.stopped_because
    assert not report.caught_up
    # The watermark stays where it was. This assertion used to read
    # `is not None`, on the reasoning that a recorded mark lets the next run
    # resume - but the mark is the *newest* paper seen, and papers arrive
    # newest-first, so advancing it does not resume anything: it puts every
    # page the failure prevented this run from reaching permanently behind the
    # mark. See the test below, which walks that through.
    assert watermark is None


def test_a_failed_run_does_not_strand_the_pages_it_never_reached(tmp_path: Path) -> None:
    """The papers a failure prevented must still arrive on the next run.

    Papers are fetched newest-first and the watermark only moves forward, so a
    failed run that advanced it would make the following run stop at the first
    paper it saw and skip the rest of the category for good.
    """
    feed = FakeArxiv(300)
    calls = {"n": 0}

    def flaky(category, *, start=0, page_size=100, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("503 Service Unavailable")
        return feed(category, start=start, page_size=page_size)

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        first = sync_category(
            corpus, "cs.CL", limit=300, page_size=100, fetch_page=flaky, sleep=no_sleep
        )
        second = sync_category(
            corpus, "cs.CL", limit=300, page_size=100, fetch_page=feed, sleep=no_sleep
        )
        stored = corpus.pipeline_stats("f", "m")["papers"]

    assert first.fetched == 200 and "failed after" in first.stopped_because
    # The retry re-walks the two committed pages, which refresh rather than
    # duplicate, and goes on to the hundred the failure cost.
    assert second.fetched == 300
    assert stored == 300


def test_a_failed_deep_walk_starts_from_the_newest_paper_again(
    tmp_path: Path,
) -> None:
    """`--full` is not resumable, and this is the shape of that.

    A deep backfill is the run most likely to need resuming - arXiv returns
    503 under load - and it is the one that cannot. Nothing records how deep a
    walk reached, so the next one pages from the newest paper again and
    re-requests what is already held before it reaches anything new.

    Asserted rather than fixed: decision 12 specifies the backfill cursor that
    would fix it and says why it is not built yet. If that changes, this test
    is where the old behaviour is written down.
    """
    feed = FakeArxiv(300)
    calls = {"n": 0}

    def flaky(category, *, start=0, page_size=100, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("503 Service Unavailable")
        return feed(category, start=start, page_size=page_size)

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        first = sync_category(
            corpus, "cs.CL", limit=300, page_size=100, fetch_page=flaky,
            sleep=no_sleep, full=True,
        )
        feed.requests.clear()
        second = sync_category(
            corpus, "cs.CL", limit=300, page_size=100, fetch_page=feed,
            sleep=no_sleep, full=True,
        )

    assert first.fetched == 200  # the two newest pages, then the failure
    # The retry does not pick up at page three. It asks for the newest papers
    # first, which it already has, and pays for them again.
    assert feed.requests[0] == (0, 100)
    assert second.fetched == 300
