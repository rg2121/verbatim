"""SQLite-backed, cumulative state for the local research corpus."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sqlite3
import struct

from verbatim.arxiv import Paper, license_class
from verbatim.chunking import REFERENCES, Chunk
from verbatim.downloads import DownloadedPaper
from verbatim.html_source import HtmlSource
from verbatim.references import Reference

SCHEMA_VERSION = "12"
# Terms are quoted before reaching FTS5, so a query may still use a trailing
# '*' for prefix matching and double quotes for phrases.
FTS_TERM_PATTERN = re.compile(r'"[^"]*"\*?|\S+')

# Passages carry their paper's licence, joined in under this name.
CHUNK_COLUMNS = "c.*, lic.license_url AS paper_license_url"
PAPER_LICENCE_JOIN = "LEFT JOIN papers AS lic ON lic.arxiv_id = c.arxiv_id"
VERSION_PATTERN = re.compile(r"^(?P<base>.+)v(?P<version>\d+)$")
# Bibliographies that were read, at the checksum they were read from. Needed
# for the ones that yield no entries, which otherwise look unread for ever.
BIBLIOGRAPHIES_READ = "bibliographies_read"
# Long enough for one stage's write to finish while another waits: chunking
# holds the write lock for minutes.
BUSY_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class PaperFilter:
    """Restrict retrieval by date, subject, licence or paper; both legs rank within it."""

    since: str | None = None
    category: str | None = None
    # Licence classes allowed. `distill` filters here, so an ineligible paper is
    # never retrieved at all.
    license_classes: tuple[str, ...] = ()
    # Only these papers, such as the ones a previous turn rested on.
    include_ids: tuple[str, ...] = ()
    # Papers to treat as absent, to plant a known gap without deleting anything.
    exclude_ids: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(
            self.since or self.category or self.license_classes
            or self.include_ids or self.exclude_ids
        )

    def conditions(self, table: str = "p") -> tuple[list[str], list[object]]:
        """The filter as SQL conditions on `table`, shared by every retrieval leg."""
        conditions: list[str] = []
        parameters: list[object] = []
        if self.since:
            conditions.append(f"{table}.published >= ?")
            parameters.append(self.since)
        if self.category:
            # categories_json is a JSON array; match the quoted element so
            # "cs.CL" does not also match "cs.CLX".
            conditions.append(f"{table}.categories_json LIKE ?")
            parameters.append(f'%"{self.category}"%')
        if self.license_classes:
            placeholders = ",".join("?" * len(self.license_classes))
            conditions.append(f"{table}.license_class IN ({placeholders})")
            parameters.extend(self.license_classes)
        if self.include_ids:
            placeholders = ",".join("?" * len(self.include_ids))
            conditions.append(f"{table}.arxiv_id IN ({placeholders})")
            parameters.extend(self.include_ids)
        if self.exclude_ids:
            placeholders = ",".join("?" * len(self.exclude_ids))
            conditions.append(f"{table}.arxiv_id NOT IN ({placeholders})")
            parameters.extend(self.exclude_ids)
        return conditions, parameters

    def clause(self, alias: str = "c") -> tuple[str, list[object]]:
        """An EXISTS clause restricting `alias` to matching papers.

        EXISTS rather than an id list, which could exceed SQLite's variable limit.
        """
        if not self:
            return "", []
        conditions, parameters = self.conditions("p")
        conditions.insert(0, f"p.arxiv_id = {alias}.arxiv_id")
        return (
            f"EXISTS (SELECT 1 FROM papers AS p WHERE {' AND '.join(conditions)})",
            parameters,
        )


@dataclass(frozen=True)
class SyncSummary:
    inserted: int
    updated: int


@dataclass(frozen=True)
class EmbeddingModel:
    """The exact model that produced a set of vectors."""

    model_id: str
    dimensions: int
    normalized: bool
    revision: str


@dataclass(frozen=True)
class ChunkMatch:
    """One retrieved passage, carrying everything a citation needs."""

    chunk: Chunk
    score: float
    snippet: str


class CorpusStore:
    """Persistent catalogue of paper versions and verified local PDFs."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Kept so a worker thread can open its own connection to the same
        # database: SQLite connections belong to the thread that made them.
        self.path = Path(path)
        self.connection = sqlite3.connect(path, timeout=BUSY_TIMEOUT_SECONDS)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        # WAL lets `stats` read while a long stage writes; the busy timeout makes a
        # second writer wait rather than fail.
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_SECONDS * 1000}")
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
                        license_class, is_latest, first_seen_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        title = excluded.title,
                        summary = excluded.summary,
                        authors_json = excluded.authors_json,
                        categories_json = excluded.categories_json,
                        published = excluded.published,
                        updated = excluded.updated,
                        source_url = excluded.source_url,
                        pdf_url = excluded.pdf_url,
                        -- The query API never returns a licence, so keep the stored one.
                        license_url = COALESCE(excluded.license_url, papers.license_url),
                        license_class = CASE
                            WHEN excluded.license_url IS NULL THEN papers.license_class
                            ELSE excluded.license_class
                        END,
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
                        license_class(paper.license_url),
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

    def pending_downloads(
        self,
        limit: int,
        *,
        require_license: bool = False,
        license_classes: Sequence[str] | None = None,
    ) -> list[Paper]:
        """Newest current versions with no local PDF, optionally limited by licence class."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        clauses = [
            "p.is_latest = 1",
            "p.pdf_url IS NOT NULL",
            "d.arxiv_id IS NULL",
            # arXiv has already said it has no PDF for these.
            "p.arxiv_id NOT IN (SELECT arxiv_id FROM download_failures)",
        ]
        parameters: list[object] = []
        if require_license:
            clauses.append("p.license_url IS NOT NULL")
        if license_classes:
            placeholders = ",".join("?" * len(license_classes))
            clauses.append(f"p.license_class IN ({placeholders})")
            parameters.extend(license_classes)
        rows = self.connection.execute(
            f"""
            SELECT p.* FROM papers AS p
            LEFT JOIN downloads AS d ON d.arxiv_id = p.arxiv_id
            WHERE {' AND '.join(clauses)}
            ORDER BY p.updated DESC, p.arxiv_id ASC
            LIMIT ?
            """,
            (*parameters, limit),
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
        return [_download_from_row(row) for row in rows]

    def record_html_source(
        self, source: HtmlSource, *, usable: bool = True, note: str | None = None
    ) -> None:
        """Record one locally held HTML rendering; an unusable one is kept with its reason."""
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO html_sources (
                    arxiv_id, url, local_path, sha256, byte_count, usable, note, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(arxiv_id) DO UPDATE SET
                    url = excluded.url, local_path = excluded.local_path,
                    sha256 = excluded.sha256, byte_count = excluded.byte_count,
                    usable = excluded.usable, note = excluded.note,
                    fetched_at = excluded.fetched_at
                """,
                (source.arxiv_id, source.url, source.local_path, source.sha256,
                 source.byte_count, int(usable), note, _timestamp()),
            )

    def record_html_absent(self, arxiv_id: str, url: str) -> None:
        """Record that arXiv publishes no HTML rendering, so it is not asked for again."""
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO html_sources (arxiv_id, url, usable, note, fetched_at)
                VALUES (?, ?, 0, 'arXiv publishes no HTML rendering', ?)
                ON CONFLICT(arxiv_id) DO UPDATE SET
                    url = excluded.url, fetched_at = excluded.fetched_at,
                    local_path = NULL, sha256 = NULL, byte_count = NULL,
                    usable = 0, note = excluded.note
                """,
                (arxiv_id, url, _timestamp()),
            )

    def papers_without_html(self, limit: int) -> list[str]:
        """Downloaded papers whose HTML has not been fetched or ruled out."""
        rows = self.connection.execute(
            """
            SELECT d.arxiv_id FROM downloads AS d
            LEFT JOIN html_sources AS h ON h.arxiv_id = d.arxiv_id
            WHERE h.arxiv_id IS NULL
            ORDER BY d.arxiv_id LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [row["arxiv_id"] for row in rows]

    def html_sources_pending_chunking(
        self, fingerprint: str, *, limit: int | None = None
    ) -> list[HtmlSource]:
        """HTML renderings whose chunks are missing or out of date."""
        rows = self.connection.execute(
            """
            SELECT * FROM html_sources
            WHERE usable = 1
              AND (chunked_at IS NULL
               OR chunked_with IS NOT ?
               OR chunked_sha256 IS NOT sha256)
            ORDER BY arxiv_id LIMIT ?
            """,
            (fingerprint, -1 if limit is None else limit),
        ).fetchall()
        return [
            HtmlSource(
                arxiv_id=r["arxiv_id"], url=r["url"], local_path=r["local_path"],
                sha256=r["sha256"], byte_count=r["byte_count"],
            )
            for r in rows
        ]

    def mark_html_chunked(self, sources: list[HtmlSource], fingerprint: str) -> None:
        now = _timestamp()
        with self.connection:
            self.connection.executemany(
                """UPDATE html_sources SET chunked_at = ?, chunked_with = ?,
                   chunked_sha256 = ? WHERE arxiv_id = ?""",
                [(now, fingerprint, s.sha256, s.arxiv_id) for s in sources],
            )

    def download_for(self, arxiv_id: str) -> DownloadedPaper | None:
        """The stored PDF for one paper, if it has been downloaded."""
        row = self.connection.execute(
            "SELECT * FROM downloads WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        return None if row is None else _download_from_row(row)

    def pending_chunking(
        self,
        fingerprint: str,
        *,
        limit: int | None = None,
        require_html_checked: bool = False,
    ) -> list[DownloadedPaper]:
        """PDFs whose chunks are missing, from another chunker, or from a replaced file.

        `require_html_checked` holds back papers whose HTML has not been looked
        for, so a rendering that arrives later does not discard their chunks.
        """
        rows = self.connection.execute(
            f"""
            SELECT d.* FROM downloads AS d
            WHERE {"EXISTS (SELECT 1 FROM html_sources AS c WHERE c.arxiv_id = d.arxiv_id) AND"
                   if require_html_checked else ""}
            NOT EXISTS (
                -- A paper with a usable HTML rendering is chunked from that, not its PDF.
                SELECT 1 FROM html_sources AS h
                WHERE h.arxiv_id = d.arxiv_id AND h.usable = 1
            )
              AND (d.chunked_at IS NULL
                   OR d.chunked_with IS NOT ?
                   OR d.chunked_sha256 IS NOT d.sha256)
            ORDER BY d.arxiv_id ASC
            LIMIT ?
            """,
            (fingerprint, -1 if limit is None else limit),
        ).fetchall()
        return [_download_from_row(row) for row in rows]

    def mark_chunked(self, downloads: list[DownloadedPaper], fingerprint: str) -> None:
        """Record a chunking pass, even one that yielded nothing, so it is not retried."""
        now = _timestamp()
        with self.connection:
            self.connection.executemany(
                """
                UPDATE downloads
                SET chunked_at = ?, chunked_with = ?, chunked_sha256 = ?
                WHERE arxiv_id = ?
                """,
                [(now, fingerprint, download.sha256, download.arxiv_id) for download in downloads],
            )

    def record_chunks(
        self, chunks: list[Chunk], for_papers: Sequence[str] | None = None
    ) -> None:
        """Replace the stored chunks of every paper this pass covered.

        A paper holds one chunking from one source. `for_papers` names the papers
        attempted, so one that now yields nothing loses its old passages.
        """
        now = _timestamp()
        covered = set(for_papers or ()) | {chunk.arxiv_id for chunk in chunks}
        with self.connection:
            # Bump the generation: a new source changes chunk ids but not their
            # count, which a count-based index check cannot see.
            self.connection.execute(
                """
                INSERT INTO schema_metadata (key, value) VALUES ('chunk_generation', '1')
                ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
                """
            )
            for arxiv_id in covered:
                self.connection.execute(
                    "DELETE FROM chunks WHERE arxiv_id = ?", (arxiv_id,)
                )
            self.connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, arxiv_id, source_sha256, source_url, license_url, page_number,
                    chunk_index, char_start, char_end, text, chunker_version, kind, section_path, source_kind, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.arxiv_id,
                        chunk.source_sha256,
                        chunk.source_url,
                        chunk.license_url,
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.text,
                        chunk.chunker_version,
                        chunk.kind,
                        chunk.section_path,
                        chunk.source_kind,
                        now,
                    )
                    for chunk in chunks
                ],
            )

    def chunk_generation(self) -> int:
        """How many times chunks have been written. See record_chunks."""
        row = self.connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'chunk_generation'"
        ).fetchone()
        return 0 if row is None else int(row["value"])

    def base_ids_missing_license(self, limit: int) -> list[str]:
        """Base identifiers whose licence was never looked up; OAI-PMH is keyed on them."""
        rows = self.connection.execute(
            """
            SELECT base_arxiv_id FROM papers
            WHERE license_url IS NULL AND license_checked_at IS NULL
            GROUP BY base_arxiv_id
            ORDER BY MAX(updated) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [row["base_arxiv_id"] for row in rows]

    def record_license(self, base_arxiv_id: str, license_url: str | None) -> None:
        """Store a licence on every version of one paper.

        Marked checked even when none came back, so it is not looked up again.
        """
        with self.connection:
            self.connection.execute(
                """
                UPDATE papers
                SET license_url = COALESCE(?, license_url),
                    license_class = ?,
                    license_checked_at = ?
                WHERE base_arxiv_id = ?
                """,
                (license_url, license_class(license_url), _timestamp(), base_arxiv_id),
            )

    def pipeline_stats(self, chunker_fingerprint: str, model_id: str) -> dict[str, int]:
        """Counts for each stage and the work each has left."""

        def scalar(sql: str, args: tuple = ()) -> int:
            return int(self.connection.execute(sql, args).fetchone()[0])
        return {
            "papers": scalar("SELECT COUNT(*) FROM papers"),
            "papers_current": scalar("SELECT COUNT(*) FROM papers WHERE is_latest = 1"),
            "licences_known": scalar("SELECT COUNT(*) FROM papers WHERE license_url IS NOT NULL"),
            "licences_pending": scalar(
                "SELECT COUNT(DISTINCT base_arxiv_id) FROM papers"
                " WHERE license_url IS NULL AND license_checked_at IS NULL"
            ),
            "downloaded": scalar("SELECT COUNT(*) FROM downloads"),
            "downloads_pending": scalar(
                "SELECT COUNT(*) FROM papers AS p LEFT JOIN downloads AS d"
                " ON d.arxiv_id = p.arxiv_id"
                " WHERE p.is_latest = 1 AND p.pdf_url IS NOT NULL AND d.arxiv_id IS NULL"
                " AND p.arxiv_id NOT IN (SELECT arxiv_id FROM download_failures)"
            ),
            "downloads_unavailable": scalar("SELECT COUNT(*) FROM download_failures"),
            "html_available": scalar("SELECT COUNT(*) FROM html_sources WHERE usable = 1"),
            "html_absent": scalar("SELECT COUNT(*) FROM html_sources WHERE usable = 0"),
            "html_pending": scalar(
                "SELECT COUNT(*) FROM downloads AS d LEFT JOIN html_sources AS h"
                " ON h.arxiv_id = d.arxiv_id WHERE h.arxiv_id IS NULL"
            ),
            "references": scalar("SELECT COUNT(*) FROM paper_references"),
            "references_pending": scalar(
                "SELECT COUNT(*) FROM html_sources AS h"
                f" WHERE {self._unread_bibliography('h')}"
            ),
            "chunks": scalar("SELECT COUNT(*) FROM chunks"),
            "chunks_html": scalar("SELECT COUNT(*) FROM chunks WHERE source_kind = 'html'"),
            "chunks_body": scalar("SELECT COUNT(*) FROM chunks WHERE kind = 'body'"),
            "chunks_tables": scalar("SELECT COUNT(*) FROM chunks WHERE kind = 'table'"),
            "chunks_references": scalar("SELECT COUNT(*) FROM chunks WHERE kind = 'references'"),
            "chunking_pending": scalar(
                "SELECT ("
                "  SELECT COUNT(*) FROM downloads AS d WHERE NOT EXISTS ("
                "    SELECT 1 FROM html_sources AS h"
                "    WHERE h.arxiv_id = d.arxiv_id AND h.usable = 1)"
                "  AND (d.chunked_at IS NULL OR d.chunked_with IS NOT ?"
                "       OR d.chunked_sha256 IS NOT d.sha256)"
                ") + ("
                "  SELECT COUNT(*) FROM html_sources WHERE usable = 1"
                "  AND (chunked_at IS NULL OR chunked_with IS NOT ?"
                "       OR chunked_sha256 IS NOT sha256)"
                ")",
                (chunker_fingerprint, chunker_fingerprint),
            ),
            "vectors": scalar("SELECT COUNT(*) FROM embeddings WHERE model_id = ?", (model_id,)),
            "embedding_pending": scalar(
                "SELECT COUNT(*) FROM chunks AS c WHERE NOT EXISTS ("
                " SELECT 1 FROM embeddings AS e WHERE e.chunk_id = c.chunk_id AND e.model_id = ?)",
                (model_id,),
            ),
            "bytes_on_disk": scalar("SELECT COALESCE(SUM(byte_count), 0) FROM downloads"),
        }

    def license_summary(self) -> dict[str, int]:
        rows = self.connection.execute(
            """
            SELECT COALESCE(license_url, '(none reported)') AS license, COUNT(*) AS n
            FROM papers GROUP BY license ORDER BY n DESC
            """
        ).fetchall()
        return {row["license"]: row["n"] for row in rows}

    def papers_updated_since(
        self,
        since: str | None = None,
        *,
        category: str | None = None,
        limit: int = 20,
        latest_only: bool = True,
    ) -> list[Paper]:
        """Papers by recency, which similarity search cannot answer."""
        clauses: list[str] = []
        parameters: list[object] = []
        if latest_only:
            clauses.append("is_latest = 1")
        if since is not None:
            clauses.append("updated >= ?")
            parameters.append(since)
        if category is not None:
            clauses.append("categories_json LIKE ?")
            parameters.append(f'%"{category}"%')
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        rows = self.connection.execute(
            f"SELECT * FROM papers {where} ORDER BY updated DESC, arxiv_id ASC LIMIT ?",
            parameters,
        ).fetchall()
        return [_paper_from_row(row) for row in rows]

    def latest_version_of(self, base_arxiv_id: str) -> Paper | None:
        """Resolve a version-less identifier, as people write it, to the current version."""
        row = self.connection.execute(
            """
            SELECT * FROM papers WHERE base_arxiv_id = ?
            ORDER BY is_latest DESC, version DESC LIMIT 1
            """,
            (base_arxiv_id.strip(),),
        ).fetchone()
        return None if row is None else _paper_from_row(row)

    def find_papers_by_title(self, text: str, *, limit: int = 5) -> list[Paper]:
        """Resolve a paper from a partial or approximate title."""
        rows = self.connection.execute(
            """
            SELECT * FROM papers WHERE is_latest = 1 AND title LIKE ?
            ORDER BY LENGTH(title) ASC, updated DESC LIMIT ?
            """,
            (f"%{text.strip()}%", limit),
        ).fetchall()
        return [_paper_from_row(row) for row in rows]

    def chunks_on_pages(self, arxiv_id: str, pages: Sequence[int]) -> list[Chunk]:
        """Every passage on the given pages of one paper, in reading order."""
        if not pages:
            return []
        placeholders = ",".join("?" * len(pages))
        rows = self.connection.execute(
            f"""
            SELECT {CHUNK_COLUMNS} FROM chunks AS c {PAPER_LICENCE_JOIN}
            WHERE c.arxiv_id = ? AND c.page_number IN ({placeholders})
            ORDER BY c.page_number, c.chunk_index
            """,
            (arxiv_id, *pages),
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def get_paper(self, arxiv_id: str) -> Paper | None:
        row = self.connection.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        return None if row is None else _paper_from_row(row)

    def papers_by_id(self, arxiv_ids: Sequence[str]) -> dict[str, Paper]:
        if not arxiv_ids:
            return {}
        placeholders = ",".join("?" * len(arxiv_ids))
        rows = self.connection.execute(
            f"SELECT * FROM papers WHERE arxiv_id IN ({placeholders})", tuple(arxiv_ids)
        ).fetchall()
        return {row["arxiv_id"]: _paper_from_row(row) for row in rows}

    def chunks_for_paper(self, arxiv_id: str) -> list[Chunk]:
        rows = self.connection.execute(
            f"SELECT {CHUNK_COLUMNS} FROM chunks AS c {PAPER_LICENCE_JOIN}"
            " WHERE c.arxiv_id = ? ORDER BY c.page_number, c.chunk_index",
            (arxiv_id,),
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def chunks_by_id(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self.connection.execute(
            f"SELECT {CHUNK_COLUMNS} FROM chunks AS c {PAPER_LICENCE_JOIN}"
            f" WHERE c.chunk_id IN ({placeholders})",
            tuple(chunk_ids),
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def iter_chunks(self):
        """Yield every chunk in a stable order, without holding them all in memory."""
        cursor = self.connection.execute(
            """
            SELECT c.*, p.license_url AS paper_license_url FROM chunks AS c
            JOIN papers AS p ON p.arxiv_id = c.arxiv_id
            ORDER BY p.updated DESC, c.arxiv_id ASC, c.page_number ASC, c.chunk_index ASC
            """
        )
        for row in cursor:
            yield _chunk_from_row(row)

    def sync_watermark(self, category: str) -> str | None:
        """The newest `updated` timestamp ingested for a category; paging stops there."""
        row = self.connection.execute(
            "SELECT newest_updated FROM sync_state WHERE category = ?", (category,)
        ).fetchone()
        return None if row is None else row["newest_updated"]

    def record_sync(self, category: str, newest_updated: str | None, papers_seen: int) -> None:
        """Advance the watermark; it never moves back."""
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO sync_state (category, newest_updated, last_synced_at, papers_seen)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(category) DO UPDATE SET
                    newest_updated = MAX(
                        COALESCE(excluded.newest_updated, ''),
                        COALESCE(sync_state.newest_updated, '')
                    ),
                    last_synced_at = excluded.last_synced_at,
                    papers_seen = sync_state.papers_seen + excluded.papers_seen
                """,
                (category, newest_updated, _timestamp(), papers_seen),
            )

    def sync_summary(self) -> list[tuple[str, str | None, int]]:
        rows = self.connection.execute(
            "SELECT category, newest_updated, papers_seen FROM sync_state ORDER BY category"
        ).fetchall()
        return [(r["category"], r["newest_updated"], r["papers_seen"]) for r in rows]

    def register_embedding_model(self, model: EmbeddingModel) -> None:
        """Record which model produced a set of vectors.

        A different dimension or revision under the same id is rejected.
        """
        existing = self.connection.execute(
            "SELECT dimensions, revision FROM embedding_models WHERE model_id = ?",
            (model.model_id,),
        ).fetchone()
        if existing is not None and (
            existing["dimensions"] != model.dimensions or existing["revision"] != model.revision
        ):
            raise ValueError(
                f"Embedding model {model.model_id} is already stored as "
                f"{existing['dimensions']}d revision {existing['revision']}; "
                f"use a new model_id rather than mixing incompatible vectors"
            )
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO embedding_models (model_id, dimensions, normalized, revision, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO NOTHING
                """,
                (model.model_id, model.dimensions, int(model.normalized), model.revision, _timestamp()),
            )

    def pending_embeddings(
        self,
        model_id: str,
        *,
        limit: int | None = None,
        chunker_version: str | None = None,
    ) -> list[Chunk]:
        """Chunks with no vector from this model yet.

        `chunker_version` skips chunks from an older chunker, whose vectors the
        next re-chunk would discard.
        """
        rows = self.connection.execute(
            """
            SELECT c.* FROM chunks AS c
            WHERE NOT EXISTS (
                SELECT 1 FROM embeddings AS e
                WHERE e.chunk_id = c.chunk_id AND e.model_id = ?
            )
              AND (? IS NULL OR c.chunker_version = ?)
            ORDER BY c.arxiv_id ASC, c.page_number ASC, c.chunk_index ASC
            LIMIT ?
            """,
            (model_id, chunker_version, chunker_version, -1 if limit is None else limit),
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def record_embeddings(self, model_id: str, vectors: list[tuple[str, Sequence[float]]]) -> None:
        """Store one vector per chunk for the given model."""
        now = _timestamp()
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO embeddings (chunk_id, model_id, vector, embedded_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chunk_id, model_id) DO UPDATE SET
                    vector = excluded.vector,
                    embedded_at = excluded.embedded_at
                """,
                [(chunk_id, model_id, pack_vector(vector), now) for chunk_id, vector in vectors],
            )

    def record_permanent_download_failures(
        self, failures: list[tuple[str, str, str]]
    ) -> None:
        """Record papers arXiv has no PDF for, so they stop being retried."""
        now = _timestamp()
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO download_failures (arxiv_id, pdf_url, reason, failed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(arxiv_id) DO UPDATE SET
                    pdf_url = excluded.pdf_url, reason = excluded.reason,
                    failed_at = excluded.failed_at
                """,
                [(arxiv_id, url, reason, now) for arxiv_id, url, reason in failures],
            )

    def record_references(
        self,
        references: list[Reference],
        source_sha256: str,
        *,
        for_papers: Sequence[str] | None = None,
    ) -> None:
        """Replace the stored bibliography of every paper this pass covered.

        `for_papers` names the papers read. Each read is recorded at its checksum,
        so a bibliography with no entries is not re-read every run.
        """
        now = _timestamp()
        covered = set(for_papers or ()) | {r.citing_arxiv_id for r in references}
        with self.connection:
            for citing in covered:
                self.connection.execute(
                    "DELETE FROM paper_references WHERE citing_arxiv_id = ?", (citing,)
                )
            if for_papers:
                # Created on first use, so opening the committed sample never changes it.
                self.connection.execute(
                    f"CREATE TABLE IF NOT EXISTS {BIBLIOGRAPHIES_READ} ("
                    " arxiv_id TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL,"
                    " read_at TEXT NOT NULL)"
                )
                self.connection.executemany(
                    f"""
                    INSERT INTO {BIBLIOGRAPHIES_READ} (arxiv_id, source_sha256, read_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        source_sha256 = excluded.source_sha256,
                        read_at = excluded.read_at
                    """,
                    [(arxiv_id, source_sha256, now) for arxiv_id in for_papers],
                )
            self.connection.executemany(
                """
                INSERT INTO paper_references (
                    citing_arxiv_id, ordinal, raw_text, cited_arxiv_id,
                    cited_doi, source_sha256, extracted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (r.citing_arxiv_id, r.ordinal, r.raw_text, r.cited_arxiv_id,
                     r.cited_doi, source_sha256, now)
                    for r in references
                ],
            )

    def papers_without_references(self, limit: int) -> list[str]:
        """Papers with a usable HTML rendering whose bibliography is unread."""
        rows = self.connection.execute(
            f"""
            SELECT h.arxiv_id FROM html_sources AS h
            WHERE {self._unread_bibliography("h")}
            ORDER BY h.arxiv_id LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [row["arxiv_id"] for row in rows]

    def _unread_bibliography(self, alias: str) -> str:
        """SQL condition: a usable rendering whose bibliography is unread at its checksum."""
        condition = (
            f"{alias}.usable = 1 AND NOT EXISTS ("
            f" SELECT 1 FROM paper_references AS r"
            f" WHERE r.citing_arxiv_id = {alias}.arxiv_id"
            f" AND r.source_sha256 = {alias}.sha256)"
        )
        if self._table_exists(BIBLIOGRAPHIES_READ):
            condition += (
                f" AND NOT EXISTS (SELECT 1 FROM {BIBLIOGRAPHIES_READ} AS b"
                f" WHERE b.arxiv_id = {alias}.arxiv_id"
                f" AND b.source_sha256 = {alias}.sha256)"
            )
        return condition

    def most_cited_absent_papers(self, *, limit: int = 50) -> list[tuple[str, int]]:
        """arXiv papers this corpus cites most and does not hold: its reading list."""
        rows = self.connection.execute(
            """
            SELECT r.cited_arxiv_id, COUNT(DISTINCT r.citing_arxiv_id) AS citations
            FROM paper_references AS r
            WHERE r.cited_arxiv_id IS NOT NULL
              AND r.cited_arxiv_id NOT IN (SELECT base_arxiv_id FROM papers)
            GROUP BY r.cited_arxiv_id
            ORDER BY citations DESC, r.cited_arxiv_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [(row["cited_arxiv_id"], int(row["citations"])) for row in rows]

    def reference_count(self) -> int:
        return int(
            self.connection.execute("SELECT COUNT(*) FROM paper_references").fetchone()[0]
        )

    def papers_citing(self, base_arxiv_id: str, *, limit: int = 25) -> list[Paper]:
        """Held papers citing this base identifier, newest first, without self-citations."""
        rows = self.connection.execute(
            """
            SELECT p.* FROM papers AS p
            WHERE p.arxiv_id IN (
                SELECT r.citing_arxiv_id FROM paper_references AS r
                WHERE r.cited_arxiv_id = ?
            )
              AND p.base_arxiv_id != ?
            ORDER BY p.updated DESC, p.version DESC, p.arxiv_id
            """,
            (base_arxiv_id, base_arxiv_id),
        ).fetchall()
        # One entry per citing paper, keeping its newest version.
        citers: dict[str, Paper] = {}
        for row in rows:
            citers.setdefault(row["base_arxiv_id"], _paper_from_row(row))
            if len(citers) == limit:
                break
        return list(citers.values())

    def papers_cited_by(self, arxiv_id: str, *, limit: int = 50) -> list[Paper]:
        """Held papers this paper's bibliography resolves to, in citation order."""
        base_arxiv_id, _ = split_arxiv_version(arxiv_id)
        rows = self.connection.execute(
            """
            SELECT held.*, MIN(r.ordinal) AS first_ordinal
            FROM paper_references AS r
            JOIN papers AS held
              ON held.base_arxiv_id = r.cited_arxiv_id AND held.is_latest = 1
            WHERE r.citing_arxiv_id = ? AND r.cited_arxiv_id != ?
            GROUP BY held.base_arxiv_id
            ORDER BY first_ordinal
            LIMIT ?
            """,
            (arxiv_id, base_arxiv_id, limit),
        ).fetchall()
        return [_paper_from_row(row) for row in rows]

    def citation_counts(self, base_arxiv_ids: Sequence[str]) -> dict[str, int]:
        """How many distinct held papers cite each of these base identifiers."""
        if not base_arxiv_ids:
            return {}
        unique = list(dict.fromkeys(base_arxiv_ids))
        placeholders = ",".join("?" * len(unique))
        rows = self.connection.execute(
            f"""
            SELECT r.cited_arxiv_id AS cited,
                   COUNT(DISTINCT citer.base_arxiv_id) AS citations
            FROM paper_references AS r
            JOIN papers AS citer ON citer.arxiv_id = r.citing_arxiv_id
            WHERE r.cited_arxiv_id IN ({placeholders})
              AND r.cited_arxiv_id != citer.base_arxiv_id
            GROUP BY r.cited_arxiv_id
            """,
            unique,
        ).fetchall()
        return {row["cited"]: int(row["citations"]) for row in rows}

    def most_cited_held_papers(self, *, limit: int = 25) -> list[tuple[Paper, int]]:
        """The held papers this corpus cites most, with their citer counts."""
        rows = self.connection.execute(
            """
            SELECT held.*, COUNT(DISTINCT citer.base_arxiv_id) AS citations
            FROM paper_references AS r
            JOIN papers AS citer ON citer.arxiv_id = r.citing_arxiv_id
            JOIN papers AS held
              ON held.base_arxiv_id = r.cited_arxiv_id AND held.is_latest = 1
            WHERE r.cited_arxiv_id != citer.base_arxiv_id
            GROUP BY held.base_arxiv_id
            ORDER BY citations DESC, held.base_arxiv_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [(_paper_from_row(row), int(row["citations"])) for row in rows]

    def citation_graph_stats(self) -> dict[str, int]:
        """Edges between held papers, papers touching one, and the paper total."""
        edges_sql = """
            SELECT DISTINCT citer.base_arxiv_id AS citing, r.cited_arxiv_id AS cited
            FROM paper_references AS r
            JOIN papers AS citer ON citer.arxiv_id = r.citing_arxiv_id
            WHERE r.cited_arxiv_id IN (SELECT base_arxiv_id FROM papers)
              AND r.cited_arxiv_id != citer.base_arxiv_id
        """
        edges = int(
            self.connection.execute(
                f"SELECT COUNT(*) FROM ({edges_sql})"
            ).fetchone()[0]
        )
        connected = int(
            self.connection.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT citing AS node FROM ({edges_sql})
                    UNION
                    SELECT cited FROM ({edges_sql})
                )
                """
            ).fetchone()[0]
        )
        papers = int(
            self.connection.execute(
                "SELECT COUNT(DISTINCT base_arxiv_id) FROM papers"
            ).fetchone()[0]
        )
        return {"edges": edges, "connected_papers": connected, "papers": papers}

    def pending_paper_embeddings(
        self, model_id: str, *, limit: int | None = None
    ) -> list[Paper]:
        """Current papers with no abstract vector for this model."""
        rows = self.connection.execute(
            """
            SELECT p.* FROM papers AS p
            WHERE p.is_latest = 1
              AND NOT EXISTS (
                  SELECT 1 FROM paper_embeddings AS e
                  WHERE e.arxiv_id = p.arxiv_id AND e.model_id = ?
              )
            ORDER BY p.arxiv_id LIMIT ?
            """,
            (model_id, -1 if limit is None else limit),
        ).fetchall()
        return [_paper_from_row(row) for row in rows]

    def record_paper_embeddings(
        self, model_id: str, vectors: list[tuple[str, Sequence[float]]]
    ) -> None:
        """Store one vector per paper for the given model."""
        now = _timestamp()
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO paper_embeddings (arxiv_id, model_id, vector, embedded_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(arxiv_id, model_id) DO UPDATE SET
                    vector = excluded.vector,
                    embedded_at = excluded.embedded_at
                """,
                [(arxiv_id, model_id, pack_vector(vector), now) for arxiv_id, vector in vectors],
            )

    def paper_embedding_count(self, model_id: str) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM paper_embeddings WHERE model_id = ?", (model_id,)
            ).fetchone()[0]
        )

    def embedding_count(self, model_id: str) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_id = ?", (model_id,)
            ).fetchone()[0]
        )

    def search_chunks(
        self,
        query: str,
        *,
        limit: int = 10,
        arxiv_id: str | None = None,
        match: str = "all",
        include_references: bool = False,
        papers: PaperFilter | None = None,
    ) -> list[ChunkMatch]:
        """Rank chunks by BM25 against a keyword query, filtering before ranking."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        expression = build_fts_query(query, match=match)
        if not expression:
            return []
        clauses = ["chunks_fts MATCH ?"]
        parameters: list[object] = [expression]
        if arxiv_id is not None:
            clauses.append("c.arxiv_id = ?")
            parameters.append(arxiv_id)
        if not include_references:
            # Exclude bibliographies rather than select body text, so tables stay in.
            clauses.append("c.kind != ?")
            parameters.append(REFERENCES)
        if papers:
            clause, filter_parameters = papers.clause("c")
            clauses.append(clause)
            parameters.extend(filter_parameters)
        parameters.append(limit)
        rows = self.connection.execute(
            f"""
            SELECT {CHUNK_COLUMNS},
                   bm25(chunks_fts) AS relevance,
                   snippet(chunks_fts, 0, '[', ']', '…', 16) AS snippet
            FROM chunks_fts
            JOIN chunks AS c ON c.rowid = chunks_fts.rowid
            {PAPER_LICENCE_JOIN}
            WHERE {' AND '.join(clauses)}
            ORDER BY relevance ASC
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        # bm25() is negative with better matches further from zero; negate so
        # callers can sort descending like every other relevance score.
        return [
            ChunkMatch(
                chunk=_chunk_from_row(row),
                score=-float(row["relevance"]),
                snippet=row["snippet"],
            )
            for row in rows
        ]

    def _initialize(self) -> None:
        # Checked first: after the schema runs, nothing can tell whether an FTS
        # index was just created and needs a backfill.
        search_index_existed = self._table_exists("chunks_fts")
        title_index_existed = self._table_exists("papers_fts")
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
                    license_class TEXT,
                    is_latest INTEGER NOT NULL CHECK (is_latest IN (0, 1)),
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    license_checked_at TEXT
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
                    downloaded_at TEXT NOT NULL,
                    chunked_at TEXT,
                    chunked_with TEXT,
                    chunked_sha256 TEXT
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id),
                    source_sha256 TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    license_url TEXT,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    -- Unused; dropping it would mean rebuilding the table and its index.
                    context TEXT,
                    kind TEXT NOT NULL DEFAULT 'body',
                    source_kind TEXT NOT NULL DEFAULT 'pdf',
                    section_path TEXT NOT NULL DEFAULT '',
                    chunker_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (arxiv_id, source_sha256, page_number, chunk_index)
                );

                CREATE INDEX IF NOT EXISTS chunks_by_paper
                    ON chunks (arxiv_id, page_number, chunk_index);

                CREATE TABLE IF NOT EXISTS html_sources (
                    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
                    url TEXT NOT NULL,
                    -- All three NULL: arXiv has no rendering, so the PDF is the source.
                    local_path TEXT,
                    sha256 TEXT,
                    byte_count INTEGER,
                    -- 0 when there is no rendering or it is truncated; note says which.
                    usable INTEGER NOT NULL DEFAULT 1,
                    note TEXT,
                    fetched_at TEXT NOT NULL,
                    chunked_at TEXT,
                    chunked_with TEXT,
                    chunked_sha256 TEXT
                );

                CREATE TABLE IF NOT EXISTS sync_state (
                    category TEXT PRIMARY KEY,
                    newest_updated TEXT,
                    last_synced_at TEXT NOT NULL,
                    papers_seen INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS embedding_models (
                    model_id TEXT PRIMARY KEY,
                    dimensions INTEGER NOT NULL,
                    normalized INTEGER NOT NULL CHECK (normalized IN (0, 1)),
                    revision TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                -- Papers arXiv has no PDF for, so they are not asked for again.
                CREATE TABLE IF NOT EXISTS download_failures (
                    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
                    pdf_url TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    failed_at TEXT NOT NULL
                );
                -- One row per bibliography entry. cited_arxiv_id is set only when the
                -- entry states an identifier, never guessed from a title.
                CREATE TABLE IF NOT EXISTS paper_references (
                    citing_arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id),
                    ordinal INTEGER NOT NULL,
                    raw_text TEXT NOT NULL,
                    cited_arxiv_id TEXT,
                    cited_doi TEXT,
                    source_sha256 TEXT NOT NULL,
                    extracted_at TEXT NOT NULL,
                    PRIMARY KEY (citing_arxiv_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS references_by_cited
                    ON paper_references (cited_arxiv_id);
                -- One vector per paper, over its title and abstract: what the paper is
                -- about, where a chunk vector says what a passage mentions.
                CREATE TABLE IF NOT EXISTS paper_embeddings (
                    arxiv_id TEXT NOT NULL
                        REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                    model_id TEXT NOT NULL REFERENCES embedding_models(model_id),
                    vector BLOB NOT NULL,
                    embedded_at TEXT NOT NULL,
                    PRIMARY KEY (arxiv_id, model_id)
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT NOT NULL
                        REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                    model_id TEXT NOT NULL REFERENCES embedding_models(model_id),
                    vector BLOB NOT NULL,
                    embedded_at TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, model_id)
                );

                CREATE INDEX IF NOT EXISTS embeddings_by_model ON embeddings (model_id);

                -- Titles, searched apart from passages: for a definitional question the
                -- paper named after the thing beats the many that merely use it.
                CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
                    title,
                    content='papers',
                    content_rowid='rowid',
                    tokenize="unicode61 remove_diacritics 2"
                );

                CREATE TRIGGER IF NOT EXISTS papers_fts_after_insert
                AFTER INSERT ON papers BEGIN
                    INSERT INTO papers_fts (rowid, title) VALUES (new.rowid, new.title);
                END;

                CREATE TRIGGER IF NOT EXISTS papers_fts_after_delete
                AFTER DELETE ON papers BEGIN
                    INSERT INTO papers_fts (papers_fts, rowid, title)
                    VALUES ('delete', old.rowid, old.title);
                END;

                CREATE TRIGGER IF NOT EXISTS papers_fts_after_update
                AFTER UPDATE ON papers BEGIN
                    INSERT INTO papers_fts (papers_fts, rowid, title)
                    VALUES ('delete', old.rowid, old.title);
                    INSERT INTO papers_fts (rowid, title) VALUES (new.rowid, new.title);
                END;

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    text,
                    content='chunks',
                    content_rowid='rowid',
                    tokenize="unicode61 remove_diacritics 2"
                );

                CREATE TRIGGER IF NOT EXISTS chunks_fts_after_insert
                AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts (rowid, text) VALUES (new.rowid, new.text);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_fts_after_delete
                AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts (chunks_fts, rowid, text)
                    VALUES ('delete', old.rowid, old.text);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_fts_after_update
                AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts (chunks_fts, rowid, text)
                    VALUES ('delete', old.rowid, old.text);
                    INSERT INTO chunks_fts (rowid, text) VALUES (new.rowid, new.text);
                END;
                """
            )
            if not search_index_existed:
                self._backfill_search_index()
            if not title_index_existed:
                self.connection.execute(
                    "INSERT INTO papers_fts (papers_fts) VALUES ('rebuild')"
                )
            self.connection.execute(
                """
                INSERT INTO schema_metadata (key, value) VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (SCHEMA_VERSION,),
            )

    def _table_exists(self, name: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def papers_matching(
        self, papers: "PaperFilter", arxiv_ids: Sequence[str]
    ) -> set[str]:
        """Which of these papers pass the filter, in one bounded query."""
        if not papers or not arxiv_ids:
            return set(arxiv_ids)
        unique = list(dict.fromkeys(arxiv_ids))
        conditions, parameters = papers.conditions("papers")
        conditions.insert(0, "arxiv_id IN (" + ",".join("?" * len(unique)) + ")")
        parameters = list(unique) + parameters
        rows = self.connection.execute(
            f"SELECT arxiv_id FROM papers WHERE {' AND '.join(conditions)}", parameters
        ).fetchall()
        return {row["arxiv_id"] for row in rows}

    def search_titles(self, query: str, *, limit: int = 10) -> list[tuple[str, float]]:
        """Papers whose title matches the query, best first."""
        expression = build_title_query(query)
        if not expression:
            return []
        try:
            rows = self.connection.execute(
                """
                SELECT p.arxiv_id, bm25(papers_fts) AS relevance
                FROM papers_fts
                JOIN papers AS p ON p.rowid = papers_fts.rowid
                WHERE papers_fts MATCH ? AND p.is_latest = 1
                ORDER BY relevance
                LIMIT ?
                """,
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # A query FTS5 cannot parse is a miss, not a crash.
            return []
        return [(row["arxiv_id"], float(row["relevance"])) for row in rows]

    def _backfill_search_index(self) -> None:
        """Rebuild the passage search index from the chunks table."""
        self.connection.execute("INSERT INTO chunks_fts (chunks_fts) VALUES ('rebuild')")


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


def pack_vector(vector: Sequence[float]) -> bytes:
    """Serialize a vector as little-endian float32, the layout sqlite-vec reads."""
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


# Question words dominate BM25 over a field as short as a title, so they are
# dropped there; in a 1,200-character passage they are diluted anyway.
TITLE_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from how in into is it its of on
    or that the their there this to was were what when where which who why with
    you your""".split()
)


def build_title_query(query: str) -> str:
    """An FTS5 expression over titles with question words removed; empty if none remain."""
    content = [
        word
        for word in re.findall(r"[\w'-]+", query.lower())
        if word not in TITLE_STOPWORDS and len(word) > 1
    ]
    return build_fts_query(" ".join(content), match="any")


def build_fts_query(query: str, *, match: str = "all") -> str:
    """Turn a user query into a safe FTS5 MATCH expression.

    Terms are quoted so hyphens, colons and arXiv ids are not read as operators;
    a trailing '*' still means prefix search. `match="all"` requires every term,
    `match="any"` ranks by matched terms, which natural-language questions need.
    """
    if match not in {"all", "any"}:
        raise ValueError("match must be 'all' or 'any'")
    terms: list[str] = []
    for raw in FTS_TERM_PATTERN.findall(query):
        prefix = raw.endswith("*")
        body = raw[:-1] if prefix else raw
        body = body.strip('"').replace('"', "")
        if not body:
            continue
        terms.append(f'"{body}"*' if prefix else f'"{body}"')
    return (" OR " if match == "any" else " ").join(terms)


def _download_from_row(row: sqlite3.Row) -> DownloadedPaper:
    return DownloadedPaper(
        arxiv_id=row["arxiv_id"],
        source_url=row["source_url"],
        pdf_url=row["pdf_url"],
        license_url=row["license_url"],
        local_path=row["local_path"],
        sha256=row["sha256"],
        byte_count=row["byte_count"],
    )


def _chunk_from_row(row: sqlite3.Row) -> Chunk:
    # The paper's licence is the authority: HTML chunks never store one, and
    # licences are usually looked up after chunking.
    license_url = row["license_url"]
    if "paper_license_url" in row.keys() and row["paper_license_url"] is not None:
        license_url = row["paper_license_url"]
    return Chunk(
        chunk_id=row["chunk_id"],
        arxiv_id=row["arxiv_id"],
        source_url=row["source_url"],
        license_url=license_url,
        source_sha256=row["source_sha256"],
        page_number=row["page_number"],
        chunk_index=row["chunk_index"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        text=row["text"],
        chunker_version=row["chunker_version"],
        kind=row["kind"],
        section_path=row["section_path"],
        source_kind=row["source_kind"],
    )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
