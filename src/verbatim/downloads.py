"""Controlled local downloads of PDFs referenced by paper metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from verbatim.arxiv import Paper


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
) -> list[DownloadedPaper]:
    """Download a bounded set of PDFs and return records for a local manifest.

    Existing files are reused.  `require_license` restricts the operation to
    records whose metadata explicitly reports a licence.
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
    downloaded: list[DownloadedPaper] = []

    for paper in candidates:
        assert paper.pdf_url is not None  # narrowed by the candidate filter
        filename = f"{_filename_stem(paper.arxiv_id)}.pdf"
        local_file = destination / filename
        content = local_file.read_bytes() if local_file.exists() else get(paper.pdf_url)
        if not content.startswith(b"%PDF"):
            raise ValueError(f"Download for {paper.arxiv_id} is not a PDF")
        if not local_file.exists():
            local_file.write_bytes(content)
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
    request = Request(
        url,
        headers={"User-Agent": "verbatim/0.1 (personal research project)"},
    )
    with urlopen(request, timeout=60) as response:  # noqa: S310 - supplied by arXiv metadata
        return response.read()


def _filename_stem(arxiv_id: str) -> str:
    """Map modern and legacy arXiv identifiers to one safe local filename."""
    return arxiv_id.replace("/", "-")
