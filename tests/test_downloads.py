from pathlib import Path
from urllib.error import HTTPError

from verbatim.arxiv import Paper
from verbatim.downloads import download_papers
from verbatim.storage import save_download_manifest


def make_paper(*, arxiv_id: str = "2401.12345v1", license_url: str | None = None) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title="Test paper",
        summary="Test abstract",
        authors=["Ada Lovelace"],
        categories=["physics.gen-ph"],
        published="2024-01-01T00:00:00Z",
        updated="2024-01-01T00:00:00Z",
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        license_url=license_url,
    )


def test_download_papers_writes_pdf_and_manifest(tmp_path: Path) -> None:
    requested_urls: list[str] = []

    def fake_get(url: str) -> bytes:
        requested_urls.append(url)
        return b"%PDF-1.7 test content\n%%EOF\n"

    report = download_papers([make_paper()], tmp_path / "pdfs", 1, fetch_pdf=fake_get, delay=0)
    manifest_path = tmp_path / "pdfs" / "manifest.json"
    save_download_manifest(report.downloaded, manifest_path)

    assert requested_urls == ["https://arxiv.org/pdf/2401.12345v1"]
    assert (tmp_path / "pdfs" / "2401.12345v1.pdf").read_bytes().startswith(b"%PDF")
    assert '"arxiv_id": "2401.12345v1"' in manifest_path.read_text()
    assert report.downloaded[0].byte_count == len(b"%PDF-1.7 test content\n%%EOF\n")
    assert report.failed == []


def test_download_papers_can_require_an_explicit_license(tmp_path: Path) -> None:
    licensed = make_paper(arxiv_id="2401.00001v1", license_url="https://creativecommons.org/licenses/by/4.0/")
    unlicensed = make_paper(arxiv_id="2401.00002v1")

    report = download_papers(
        [unlicensed, licensed],
        tmp_path,
        2,
        require_license=True,
        fetch_pdf=lambda _: b"%PDF-1.7 test content\n%%EOF\n",
        delay=0,
    )

    assert [download.arxiv_id for download in report.downloaded] == ["2401.00001v1"]


def test_download_papers_reports_non_pdf_content_as_a_failure(tmp_path: Path) -> None:
    report = download_papers([make_paper()], tmp_path, 1, fetch_pdf=lambda _: b"HTML error page", delay=0)

    assert report.downloaded == []
    assert [failure.arxiv_id for failure in report.failed] == ["2401.12345v1"]
    assert "not a complete PDF" in report.failed[0].reason


def test_download_papers_replaces_cached_file_that_fails_manifest_validation(tmp_path: Path) -> None:
    content = b"%PDF-1.7 valid content\n%%EOF\n"
    first = download_papers([make_paper()], tmp_path, 1, fetch_pdf=lambda _: content, delay=0)
    (tmp_path / "2401.12345v1.pdf").write_bytes(b"%PDF-1.7 interrupted download")
    requested_urls: list[str] = []

    def fake_get(url: str) -> bytes:
        requested_urls.append(url)
        return content

    second = download_papers(
        [make_paper()],
        tmp_path,
        1,
        fetch_pdf=fake_get,
        existing_downloads=first.downloaded,
        delay=0,
    )

    assert requested_urls == ["https://arxiv.org/pdf/2401.12345v1"]
    assert second.downloaded[0].sha256 == first.downloaded[0].sha256


def test_download_papers_reports_an_unapproved_pdf_url_as_a_failure(tmp_path: Path) -> None:
    paper = make_paper()
    paper = Paper(**{**paper.to_dict(), "pdf_url": "file:///tmp/not-a-paper.pdf"})

    report = download_papers([paper], tmp_path, 1, fetch_pdf=lambda _: b"%PDF\n%%EOF", delay=0)

    assert report.downloaded == []
    assert "must use HTTPS" in report.failed[0].reason


def test_one_papers_failure_does_not_discard_the_others(tmp_path: Path) -> None:
    """The orphaning bug: a 404 mid-batch used to abort the whole run."""
    good_a = make_paper(arxiv_id="2401.00001v1")
    withdrawn = make_paper(arxiv_id="2401.00002v1")
    good_b = make_paper(arxiv_id="2401.00003v1")

    def fake_get(url: str) -> bytes:
        if "2401.00002v1" in url:
            raise HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]
        return b"%PDF-1.7 body\n%%EOF\n"

    report = download_papers([good_a, withdrawn, good_b], tmp_path, 3, fetch_pdf=fake_get, delay=0)

    assert [d.arxiv_id for d in report.downloaded] == ["2401.00001v1", "2401.00003v1"]
    assert [f.arxiv_id for f in report.failed] == ["2401.00002v1"]
    assert "404" in report.failed[0].reason
    # The paper that followed the failure still landed on disk *and* is reported.
    assert (tmp_path / "2401.00003v1.pdf").exists()


def test_a_failure_never_leaves_an_unreported_file_on_disk(tmp_path: Path) -> None:
    """Every PDF written must appear in the report, or it is an orphan."""
    papers = [make_paper(arxiv_id=f"2401.0000{n}v1") for n in range(1, 5)]

    def fake_get(url: str) -> bytes:
        if "2401.00003v1" in url:
            raise HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]
        return b"%PDF-1.7 body\n%%EOF\n"

    report = download_papers(papers, tmp_path, 4, fetch_pdf=fake_get, delay=0)

    on_disk = {path.stem for path in tmp_path.glob("*.pdf")}
    reported = {download.arxiv_id for download in report.downloaded}
    assert on_disk == reported
    assert len(report.failed) == 1


def test_downloads_are_paced_but_a_reused_file_costs_no_pause(tmp_path: Path) -> None:
    """arXiv asks callers not to hammer it; a cached hit made no request."""
    content = b"%PDF-1.7 body\n%%EOF\n"
    papers = [make_paper(arxiv_id=f"2401.0000{n}v1") for n in range(1, 4)]
    pauses: list[float] = []

    first = download_papers(papers, tmp_path, 3, fetch_pdf=lambda _: content, delay=0)
    again = download_papers(
        papers,
        tmp_path,
        3,
        fetch_pdf=lambda _: content,
        existing_downloads=first.downloaded,
        delay=3.0,
        sleep=pauses.append,
    )

    assert len(again.downloaded) == 3
    assert pauses == []  # all three were reused, so nothing was requested


def test_a_fresh_batch_pauses_between_requests(tmp_path: Path) -> None:
    papers = [make_paper(arxiv_id=f"2401.0001{n}v1") for n in range(1, 4)]
    pauses: list[float] = []

    download_papers(
        papers,
        tmp_path,
        3,
        fetch_pdf=lambda _: b"%PDF-1.7 body\n%%EOF\n",
        delay=3.0,
        sleep=pauses.append,
    )

    assert pauses == [3.0, 3.0]  # between requests, not before the first


def test_each_download_is_reported_as_it_lands(tmp_path: Path) -> None:
    """A long run must not lose hours of work to one interruption."""
    papers = [make_paper(arxiv_id=f"2401.0002{n}v1") for n in range(1, 4)]
    recorded: list[str] = []

    def failing_get(url: str) -> bytes:
        if "2401.00023v1" in url:
            raise HTTPError(url, 500, "Server Error", {}, None)  # type: ignore[arg-type]
        return b"%PDF-1.7 body\n%%EOF\n"

    report = download_papers(
        papers,
        tmp_path,
        3,
        fetch_pdf=failing_get,
        delay=0,
        on_download=lambda record: recorded.append(record.arxiv_id),
    )

    # The two that succeeded were reported before the third one failed.
    assert recorded == ["2401.00021v1", "2401.00022v1"]
    assert [d.arxiv_id for d in report.downloaded] == recorded
    assert len(report.failed) == 1
