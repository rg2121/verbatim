from pathlib import Path

import pytest

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

    downloads = download_papers([make_paper()], tmp_path / "pdfs", 1, fetch_pdf=fake_get)
    manifest_path = tmp_path / "pdfs" / "manifest.json"
    save_download_manifest(downloads, manifest_path)

    assert requested_urls == ["https://arxiv.org/pdf/2401.12345v1"]
    assert (tmp_path / "pdfs" / "2401.12345v1.pdf").read_bytes().startswith(b"%PDF")
    assert '"arxiv_id": "2401.12345v1"' in manifest_path.read_text()
    assert downloads[0].byte_count == len(b"%PDF-1.7 test content\n%%EOF\n")


def test_download_papers_can_require_an_explicit_license(tmp_path: Path) -> None:
    licensed = make_paper(arxiv_id="2401.00001v1", license_url="https://creativecommons.org/licenses/by/4.0/")
    unlicensed = make_paper(arxiv_id="2401.00002v1")

    downloads = download_papers(
        [unlicensed, licensed],
        tmp_path,
        2,
        require_license=True,
        fetch_pdf=lambda _: b"%PDF-1.7 test content\n%%EOF\n",
    )

    assert [download.arxiv_id for download in downloads] == ["2401.00001v1"]


def test_download_papers_rejects_non_pdf_content(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a complete PDF"):
        download_papers([make_paper()], tmp_path, 1, fetch_pdf=lambda _: b"HTML error page")


def test_download_papers_replaces_cached_file_that_fails_manifest_validation(tmp_path: Path) -> None:
    content = b"%PDF-1.7 valid content\n%%EOF\n"
    first = download_papers([make_paper()], tmp_path, 1, fetch_pdf=lambda _: content)
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
        existing_downloads=first,
    )

    assert requested_urls == ["https://arxiv.org/pdf/2401.12345v1"]
    assert second[0].sha256 == first[0].sha256


def test_download_papers_rejects_unapproved_pdf_urls(tmp_path: Path) -> None:
    paper = make_paper()
    paper = Paper(**{**paper.to_dict(), "pdf_url": "file:///tmp/not-a-paper.pdf"})

    with pytest.raises(ValueError, match="must use HTTPS"):
        download_papers([paper], tmp_path, 1, fetch_pdf=lambda _: b"%PDF\n%%EOF")
