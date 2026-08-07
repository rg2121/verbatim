"""Controlled local downloads of PDFs referenced by paper metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import os
from pathlib import Path
import tempfile
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from verbatim.arxiv import ARXIV_HOSTS, Paper


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


def download_papers(
    papers: list[Paper],
    destination: Path,
    limit: int,
    *,
    require_license: bool = False,
    fetch_pdf: Callable[[str], bytes] | None = None,
    existing_downloads: list[DownloadedPaper] | None = None,
) -> list[DownloadedPaper]:
    """Download a bounded set of PDFs and return records for a local manifest.

    Cached files are reused only when they match a previous download manifest.
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

    for paper in candidates:
        if paper.pdf_url is None:
            raise ValueError(f"Paper {paper.arxiv_id} has no PDF URL")
        pdf_url = _validate_pdf_url(paper.pdf_url)
        filename = f"{_filename_stem(paper.arxiv_id)}.pdf"
        local_file = destination / filename
        prior = existing_by_id.get(paper.arxiv_id)
        if _matches_prior_manifest(local_file, paper, prior):
            content = local_file.read_bytes()
        else:
            content = get(pdf_url)
            if not _looks_like_pdf(content):
                raise ValueError(f"Download for {paper.arxiv_id} is not a complete PDF")
            _atomic_write_bytes(local_file, content)
        if not _looks_like_pdf(content):
            raise ValueError(f"Download for {paper.arxiv_id} is not a PDF")
        downloaded.append(
            DownloadedPaper(
                arxiv_id=paper.arxiv_id,
                source_url=paper.source_url,
                pdf_url=paper.pdf_url,
                license_url=paper.license_url,
                local_path=str(local_file),
                sha256=sha256(content).hexdigest(),
                byte_count=len(content),
            )
        )
    return downloaded


def _get_pdf(url: str) -> bytes:
    url = _validate_pdf_url(url)
    request = Request(
        url,
        headers={"User-Agent": "verbatim/0.1 (personal research project)"},
    )
    with urlopen(request, timeout=60) as response:  # noqa: S310 - supplied by arXiv metadata
        return response.read()


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
