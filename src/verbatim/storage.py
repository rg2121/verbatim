"""Local persistence for ingested paper metadata."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from verbatim.arxiv import Paper


def save_papers(papers: list[Paper], destination: Path) -> None:
    """Save an inspectable JSON snapshot, creating parent directories as needed."""
    payload = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "paper_count": len(papers),
        "papers": [paper.to_dict() for paper in papers],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

