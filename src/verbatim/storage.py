"""Local persistence for ingested paper metadata."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from verbatim.arxiv import Paper
from verbatim.downloads import DownloadedPaper


def save_papers(papers: list[Paper], destination: Path) -> None:
    """Save an inspectable JSON snapshot, creating parent directories as needed."""
    payload = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "paper_count": len(papers),
        "papers": [paper.to_dict() for paper in papers],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
