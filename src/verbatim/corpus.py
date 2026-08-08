"""SQLite-backed, cumulative state for the local research corpus."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sqlite3

from verbatim.arxiv import Paper
from verbatim.chunking import Chunk
from verbatim.downloads import DownloadedPaper

SCHEMA_VERSION = "2"
VERSION_PATTERN = re.compile(r"^(?P<base>.+)v(?P<version>\d+)$")


@dataclass(frozen=True)
class SyncSummary:
    inserted: int
    updated: int


class CorpusStore:
    """Persistent catalogue of paper versions and verified local PDFs."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def __enter__(self) -> CorpusStore:
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def upsert_papers(self, papers: list[Paper]) -> SyncSummary:
        """Insert or refresh paper versions without removing older records."""
        now = _timestamp()
        inserted = 0
        updated = 0
        with self.connection:
            for paper in papers:
                base_id, version = split_arxiv_version(paper.arxiv_id)
                existing = self.connection.execute(
                    "SELECT 1 FROM papers WHERE arxiv_id = ?", (paper.arxiv_id,)
                ).fetchone()
                if existing is None:
                    inserted += 1
                else:
                    updated += 1

                latest_version = self.connection.execute(
                    "SELECT MAX(version) FROM papers WHERE base_arxiv_id = ?", (base_id,)
                ).fetchone()[0]
                is_latest = latest_version is None or version >= latest_version
                self.connection.execute(
                    """
                    INSERT INTO papers (
                        arxiv_id, base_arxiv_id, version, title, summary, authors_json,
                        categories_json, published, updated, source_url, pdf_url, license_url,
                        is_latest, first_seen_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        title = excluded.title,
                        summary = excluded.summary,
                        authors_json = excluded.authors_json,
                        categories_json = excluded.categories_json,
                        published = excluded.published,
                        updated = excluded.updated,
                        source_url = excluded.source_url,
                        pdf_url = excluded.pdf_url,
                        license_url = excluded.license_url,
                        is_latest = excluded.is_latest,
                        last_seen_at = excluded.last_seen_at
                    """,
                    (
                        paper.arxiv_id,
                        base_id,
                        version,
                        paper.title,
                        paper.summary,
                        json.dumps(paper.authors),
                        json.dumps(paper.categories),
                        paper.published,
                        paper.updated,
                        paper.source_url,
                        paper.pdf_url,
                        paper.license_url,
                        int(is_latest),
                        now,
                        now,
                    ),
                )
                if is_latest:
                    self.connection.execute(
                        "UPDATE papers SET is_latest = 0 WHERE base_arxiv_id = ? AND version < ?",
                        (base_id, version),
                    )
        return SyncSummary(inserted=inserted, updated=updated)

    def pending_downloads(self, limit: int, *, require_license: bool = False) -> list[Paper]:
        """Return newest current versions that do not yet have a local PDF."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        clauses = ["p.is_latest = 1", "p.pdf_url IS NOT NULL", "d.arxiv_id IS NULL"]
        if require_license:
            clauses.append("p.license_url IS NOT NULL")
        rows = self.connection.execute(
            f"""
            SELECT p.* FROM papers AS p
            LEFT JOIN downloads AS d ON d.arxiv_id = p.arxiv_id
            WHERE {' AND '.join(clauses)}
            ORDER BY p.updated DESC, p.arxiv_id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_paper_from_row(row) for row in rows]

    def record_downloads(self, downloads: list[DownloadedPaper]) -> None:
        """Record verified local PDFs without losing downloads from previous runs."""
        with self.connection:
            for download in downloads:
                self.connection.execute(
                    """
                    INSERT INTO downloads (
                        arxiv_id, source_url, pdf_url, license_url, local_path, sha256,
                        byte_count, downloaded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        source_url = excluded.source_url,
                        pdf_url = excluded.pdf_url,
                        license_url = excluded.license_url,
                        local_path = excluded.local_path,
                        sha256 = excluded.sha256,
                        byte_count = excluded.byte_count,
                        downloaded_at = excluded.downloaded_at
                    """,
                    (
                        download.arxiv_id,
                        download.source_url,
                        download.pdf_url,
                        download.license_url,
                        download.local_path,
                        download.sha256,
                        download.byte_count,
                        _timestamp(),
                    ),
                )

    def all_downloads(self) -> list[DownloadedPaper]:
        """Return every tracked local PDF, including superseded paper versions."""
        rows = self.connection.execute(
            """
            SELECT d.* FROM downloads AS d
            JOIN papers AS p ON p.arxiv_id = d.arxiv_id
            ORDER BY p.updated DESC, d.arxiv_id ASC
            """
        ).fetchall()
        return [
            DownloadedPaper(
                arxiv_id=row["arxiv_id"],
                source_url=row["source_url"],
                pdf_url=row["pdf_url"],
                license_url=row["license_url"],
                local_path=row["local_path"],
                sha256=row["sha256"],
                byte_count=row["byte_count"],
            )
            for row in rows
        ]

    def record_chunks(self, chunks: list[Chunk]) -> None:
        """Replace the stored chunks for every PDF represented in `chunks`.

        Re-chunking a PDF supersedes its previous chunks entirely, so the table
        always holds exactly one chunking of each file.
        """
        now = _timestamp()
        with self.connection:
            for arxiv_id, pdf_sha256 in {(chunk.arxiv_id, chunk.pdf_sha256) for chunk in chunks}:
                self.connection.execute(
                    "DELETE FROM chunks WHERE arxiv_id = ? AND pdf_sha256 = ?",
                    (arxiv_id, pdf_sha256),
                )
            self.connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, arxiv_id, pdf_sha256, source_url, license_url, page_number,
                    chunk_index, char_start, char_end, text, chunker_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.arxiv_id,
                        chunk.pdf_sha256,
                        chunk.source_url,
                        chunk.license_url,
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.text,
                        chunk.chunker_version,
                        now,
                    )
                    for chunk in chunks
                ],
            )

    def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk in a stable, citable order."""
        rows = self.connection.execute(
            """
            SELECT c.* FROM chunks AS c
            JOIN papers AS p ON p.arxiv_id = c.arxiv_id
            ORDER BY p.updated DESC, c.arxiv_id ASC, c.page_number ASC, c.chunk_index ASC
            """
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def chunk_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def _initialize(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS papers (
                    arxiv_id TEXT PRIMARY KEY,
                    base_arxiv_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    authors_json TEXT NOT NULL,
                    categories_json TEXT NOT NULL,
                    published TEXT NOT NULL,
                    updated TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    pdf_url TEXT,
                    license_url TEXT,
                    is_latest INTEGER NOT NULL CHECK (is_latest IN (0, 1)),
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS papers_latest_updated
                    ON papers (is_latest, updated DESC);

                CREATE TABLE IF NOT EXISTS downloads (
                    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
                    source_url TEXT NOT NULL,
                    pdf_url TEXT NOT NULL,
                    license_url TEXT,
                    local_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    byte_count INTEGER NOT NULL,
                    downloaded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id),
                    pdf_sha256 TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    license_url TEXT,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    context TEXT,
                    chunker_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (arxiv_id, pdf_sha256, page_number, chunk_index)
                );

                CREATE INDEX IF NOT EXISTS chunks_by_paper
                    ON chunks (arxiv_id, page_number, chunk_index);
                """
            )
            self.connection.execute(
                """
                INSERT INTO schema_metadata (key, value) VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (SCHEMA_VERSION,),
            )


def split_arxiv_version(arxiv_id: str) -> tuple[str, int]:
    """Split both modern and legacy arXiv IDs into an identifier and version."""
    match = VERSION_PATTERN.fullmatch(arxiv_id)
    if match is None:
        raise ValueError(f"arXiv identifier has no version suffix: {arxiv_id}")
    return match.group("base"), int(match.group("version"))


def _paper_from_row(row: sqlite3.Row) -> Paper:
    return Paper(
        arxiv_id=row["arxiv_id"],
        title=row["title"],
        summary=row["summary"],
        authors=json.loads(row["authors_json"]),
        categories=json.loads(row["categories_json"]),
        published=row["published"],
        updated=row["updated"],
        source_url=row["source_url"],
        pdf_url=row["pdf_url"],
        license_url=row["license_url"],
    )


def _chunk_from_row(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        arxiv_id=row["arxiv_id"],
        source_url=row["source_url"],
        license_url=row["license_url"],
        pdf_sha256=row["pdf_sha256"],
        page_number=row["page_number"],
        chunk_index=row["chunk_index"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        text=row["text"],
        chunker_version=row["chunker_version"],
    )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
