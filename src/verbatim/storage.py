"""Local persistence for ingested paper metadata."""

from __future__ import annotations

from collections.abc import Iterable
import json
from datetime import UTC, datetime
import os
from pathlib import Path
import tempfile

from verbatim.arxiv import Paper
from verbatim.chunking import Chunk
from verbatim.downloads import DownloadedPaper


def save_papers(papers: list[Paper], destination: Path) -> None:
    """Save an inspectable JSON snapshot, creating parent directories as needed."""
    payload = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "paper_count": len(papers),
        "papers": [paper.to_dict() for paper in papers],
    }
    _atomic_write_text(destination, json.dumps(payload, indent=2) + "\n")


def load_papers(source: Path) -> list[Paper]:
    """Load a metadata snapshot created by :func:`save_papers`."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    return [Paper(**paper) for paper in payload["papers"]]


def save_download_manifest(downloads: list[DownloadedPaper], destination: Path) -> None:
    """Write the provenance and checksum records for downloaded PDFs."""
    payload = {
        "paper_count": len(downloads),
        "papers": [download.to_dict() for download in downloads],
    }
    _atomic_write_text(destination, json.dumps(payload, indent=2) + "\n")


def load_download_manifest(source: Path) -> list[DownloadedPaper]:
    """Load an auditable PDF manifest created by :func:`save_download_manifest`."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    return [DownloadedPaper(**paper) for paper in payload["papers"]]


def save_chunks(chunks: Iterable[Chunk], destination: Path) -> int:
    """Write chunks as JSON Lines, streaming so the corpus never sits in memory.

    Returns the number written, since a generator cannot be counted twice.
    """
    return _atomic_write_lines(
        destination, (json.dumps(chunk.to_dict()) + "\n" for chunk in chunks)
    )


def _atomic_write_lines(destination: Path, lines: Iterable[str]) -> int:
    """Stream lines into a temporary file, then replace the destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as temporary:
            temporary_name = temporary.name
            for line in lines:
                temporary.write(line)
                written += 1
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return written


def _atomic_write_text(destination: Path, content: str) -> None:
    """Write JSON artifacts fully before replacing their previous version."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
