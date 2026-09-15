"""Controlled local downloads of PDFs referenced by paper metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import os
import time
from pathlib import Path
import tempfile
from collections.abc import Callable
from urllib.parse import urlparse

from verbatim import config
from verbatim.arxiv import ARXIV_HOSTS, Paper
from verbatim.fetching import fetch


@dataclass(frozen=True)
class DownloadedPaper:
    """An auditable record of one local PDF file."""

    arxiv_id: str
    source_url: str
    pdf_url: str
    license_url: str | None
    local_path: str
    sha256: str
    byte_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FailedDownload:
    """A paper that could not be fetched, and why."""

    arxiv_id: str
    pdf_url: str
    reason: str
    # True when arXiv says the file does not exist, rather than failing to
    # serve it. Retrying a 404 tomorrow gets the same 404.
    permanent: bool = False


@dataclass(frozen=True)
class DownloadReport:
    """What one download run achieved, including what it could not."""

    downloaded: list[DownloadedPaper]
    failed: list[FailedDownload]


def _is_permanently_absent(error: Exception) -> bool:
    """Whether arXiv is saying this file does not exist, rather than failing.

    A 404 or 410 for a versioned PDF is settled: that version has no PDF and
    asking again tomorrow will not change it. A timeout or a 503 is arXiv
    being busy and is worth retrying. Only the first is worth writing down as
    an answer.
    """
    status = getattr(error, "code", None)
    return status in (404, 410)


def download_papers(
    papers: list[Paper],
    destination: Path,
    limit: int,
    *,
    require_license: bool = False,
    fetch_pdf: Callable[[str], bytes] | None = None,
    existing_downloads: list[DownloadedPaper] | None = None,
    delay: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_download: Callable[[DownloadedPaper], None] | None = None,
) -> DownloadReport:
    """Download a bounded set of PDFs and report what succeeded and what did not.

    One paper's failure never discards another's: a withdrawn paper returning
    404 used to abort the run, leaving already-written PDFs on disk with no
    record of them anywhere.  Each paper is now isolated, and the caller records
    the successes regardless of how many failed.

    `on_download` is called after each success, so a caller can persist progress
    as it happens.  Without it, a run interrupted after hours leaves every PDF
    on disk with nothing recording that it arrived, and the next run fetches
    them all again.

    Cached files are reused only when they match a previous download manifest,
    and a reused file costs no pause because nothing was requested.
    `require_license` restricts the operation to records whose metadata
    explicitly reports a licence.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")

    candidates = [
        paper
        for paper in papers
        if paper.pdf_url is not None and (not require_license or paper.license_url is not None)
    ][:limit]
    destination.mkdir(parents=True, exist_ok=True)
    get = fetch_pdf or _get_pdf
    existing_by_id = {download.arxiv_id: download for download in existing_downloads or []}
    downloaded: list[DownloadedPaper] = []
    failed: list[FailedDownload] = []

    pause = config.DOWNLOAD_DELAY if delay is None else delay
    requested = 0

    for paper in candidates:
        assert paper.pdf_url is not None  # guaranteed by the candidate filter
        try:
            prior = existing_by_id.get(paper.arxiv_id)
            # Pace only actual requests: arXiv asks callers not to hammer it,
            # and thousands of downloads without a gap is exactly that.
            if requested and not _matches_prior_manifest(
                destination / f"{_filename_stem(paper.arxiv_id)}.pdf", paper, prior
            ):
                sleep(pause)
            record = _download_one(paper, destination, get, prior)
            downloaded.append(record)
            requested += 1
            if on_download is not None:
                on_download(record)
        except (OSError, ValueError) as error:
            # OSError covers HTTPError and URLError; ValueError covers a
            # rejected URL or a truncated response.  Neither says anything
            # about the other papers in this batch.
            failed.append(
                FailedDownload(
                    paper.arxiv_id, paper.pdf_url, str(error),
                    permanent=_is_permanently_absent(error),
                )
            )
    return DownloadReport(downloaded=downloaded, failed=failed)


def _download_one(
    paper: Paper,
    destination: Path,
    get: Callable[[str], bytes],
    prior: DownloadedPaper | None,
) -> DownloadedPaper:
    pdf_url = _validate_pdf_url(paper.pdf_url or "")
    local_file = destination / f"{_filename_stem(paper.arxiv_id)}.pdf"
    if _matches_prior_manifest(local_file, paper, prior):
        content = local_file.read_bytes()
        if not _looks_like_pdf(content):
            raise ValueError(f"Cached file for {paper.arxiv_id} is not a complete PDF")
    else:
        content = get(pdf_url)
        if not _looks_like_pdf(content):
            raise ValueError(f"Download for {paper.arxiv_id} is not a complete PDF")
        _atomic_write_bytes(local_file, content)
    return DownloadedPaper(
        arxiv_id=paper.arxiv_id,
        source_url=paper.source_url,
        pdf_url=pdf_url,
        license_url=paper.license_url,
        local_path=str(local_file),
        sha256=sha256(content).hexdigest(),
        byte_count=len(content),
    )


def _get_pdf(url: str) -> bytes:
    """Fetch one PDF, with the same retry policy as the metadata client.

    A withdrawn paper's 404 is not retried, so it still surfaces promptly as a
    per-paper failure rather than costing four attempts.
    """
    return fetch(
        _validate_pdf_url(url),
        timeout=config.PDF_TIMEOUT,
        max_bytes=config.PDF_MAX_BYTES,
    )


def file_sha256(path: Path) -> str:
    """Calculate a file checksum without holding the full file in memory."""
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches_prior_manifest(
    local_file: Path,
    paper: Paper,
    prior: DownloadedPaper | None,
) -> bool:
    if prior is None or not local_file.is_file():
        return False
    if prior.source_url != paper.source_url or prior.pdf_url != paper.pdf_url:
        return False
    return prior.byte_count == local_file.stat().st_size and prior.sha256 == file_sha256(local_file)


def _validate_pdf_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ARXIV_HOSTS:
        raise ValueError(f"PDF URL must use HTTPS and an approved arXiv host: {url}")
    if not parsed.path.startswith("/pdf/"):
        raise ValueError(f"PDF URL must point to an arXiv PDF: {url}")
    return url


def _looks_like_pdf(content: bytes) -> bool:
    return content.startswith(b"%PDF") and content.rstrip().endswith(b"%%EOF")


def _atomic_write_bytes(destination: Path, content: bytes) -> None:
    """Write a file fully before making it visible at its final path."""
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _filename_stem(arxiv_id: str) -> str:
    """Map modern and legacy arXiv identifiers to one safe local filename."""
    return arxiv_id.replace("/", "-")
