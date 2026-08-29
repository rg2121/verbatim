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
from verbatim.chunking import Chunk
from verbatim.downloads import DownloadedPaper
from verbatim.html_source import HtmlSource

SCHEMA_VERSION = "12"
# Terms are quoted before reaching FTS5, so a query may still use a trailing
# '*' for prefix matching and double quotes for phrases.
FTS_TERM_PATTERN = re.compile(r'"[^"]*"\*?|\S+')

# Columns added to existing databases after their table was first created.
ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "chunks": {
        "source_sha256": "TEXT NOT NULL DEFAULT ''",
        "kind": "TEXT NOT NULL DEFAULT 'body'",
        "source_kind": "TEXT NOT NULL DEFAULT 'pdf'",
        "section_path": "TEXT NOT NULL DEFAULT ''",
    },
    "papers": {"license_checked_at": "TEXT", "license_class": "TEXT"},
    "html_sources": {
        "usable": "INTEGER NOT NULL DEFAULT 1",
        "note": "TEXT",
    },
    "downloads": {
        "chunked_at": "TEXT",
        "chunked_with": "TEXT",
        "chunked_sha256": "TEXT",
    },
}
# Columns renamed after data was already stored: (old, new).
RENAMED_COLUMNS: dict[str, tuple[str, str]] = {
    "chunks": ("pdf_sha256", "source_sha256"),
}
VERSION_PATTERN = re.compile(r"^(?P<base>.+)v(?P<version>\d+)$")


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
    """One keyword hit, carrying everything a citation needs."""

    chunk: Chunk
    score: float
    snippet: str


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
                        license_url = excluded.license_url,
                        license_class = excluded.license_class,
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
        """Return newest current versions that do not yet have a local PDF.

        `license_classes` gates on what a licence *permits*, which is not the
        same as having one: arXiv's default grants arXiv distribution rights
        and grants the caller nothing.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        clauses = ["p.is_latest = 1", "p.pdf_url IS NOT NULL", "d.arxiv_id IS NULL"]
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
        """Record one locally held HTML rendering.

        An unusable rendering is kept, with its checksum and the reason: the
        bytes are evidence for the decision and allow it to be revisited when
        arXiv re-renders the paper.
        """
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
        """Record that arXiv publishes no HTML rendering of this paper.

        Without this the paper looks unchecked forever, and every later run
        asks arXiv for a page that has already been shown not to exist. The
        absence is a finding, so it is stored like any other.
        """
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
        self, fingerprint: str, *, limit: int | None = None
    ) -> list[DownloadedPaper]:
        """Return PDFs whose stored chunks do not match the current chunker.

        Re-chunking is needed when a PDF has never been chunked, when the
        chunker version or its parameters changed, or when the file on disk was
        replaced since it was last chunked.
        """
        rows = self.connection.execute(
            """
            SELECT d.* FROM downloads AS d
            WHERE NOT EXISTS (
                -- HTML is the better source where it exists, so a paper with
                -- one is never chunked from its PDF. A row with no checksum
                -- records that arXiv has no rendering, and such a paper must
                -- still be chunked from its PDF or it is never chunked at all.
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
        """Record a completed chunking pass, including passes that yielded nothing.

        Without this a PDF with no extractable text would be retried forever.
        """
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

    def record_chunks(self, chunks: list[Chunk]) -> None:
        """Replace the stored chunks for every paper represented in `chunks`.

        A paper is held as exactly one chunking, from one source. Deleting only
        the chunks sharing a checksum would leave a paper's PDF chunks behind
        when it was re-read from its HTML, so the same passages would sit in
        the index twice, under two sources, and be retrieved twice.
        """
        now = _timestamp()
        with self.connection:
            for arxiv_id in {chunk.arxiv_id for chunk in chunks}:
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

    def base_ids_missing_license(self, limit: int) -> list[str]:
        """Base identifiers whose licence has never been looked up.

        Keyed on the base id because OAI-PMH is: one lookup covers every
        version of a paper.
        """
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
        """Store a looked-up licence against every version of one paper.

        The checked timestamp is written even when no licence came back, so a
        paper that predates the field is not looked up again on every run.
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
        """Counts for each stage, and how much work each has left.

        One query set rather than several calls, so a long ingest can be
        watched without loading the corpus into memory.
        """
        scalar = lambda sql, args=(): int(self.connection.execute(sql, args).fetchone()[0])
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
            ),
            "html_available": scalar("SELECT COUNT(*) FROM html_sources WHERE usable = 1"),
            "html_absent": scalar("SELECT COUNT(*) FROM html_sources WHERE usable = 0"),
            "html_pending": scalar(
                "SELECT COUNT(*) FROM downloads AS d LEFT JOIN html_sources AS h"
                " ON h.arxiv_id = d.arxiv_id WHERE h.arxiv_id IS NULL"
            ),
            "chunks": scalar("SELECT COUNT(*) FROM chunks"),
            "chunks_html": scalar("SELECT COUNT(*) FROM chunks WHERE source_kind = 'html'"),
            "chunks_body": scalar("SELECT COUNT(*) FROM chunks WHERE kind = 'body'"),
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
        """Papers by recency — the question vector search cannot answer.

        Similarity search returns things that resemble a query; it has no
        notion of "this week" and cannot promise completeness. Recency and
        exhaustiveness are SQL questions, so they are answered in SQL.
        """
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
        """Resolve a version-less identifier to the current version.

        People and models both write "2608.05440" while the catalogue keys on
        "2608.05440v1", so an exact-match-only lookup fails on the form
        everyone actually uses.
        """
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
            SELECT * FROM chunks WHERE arxiv_id = ? AND page_number IN ({placeholders})
            ORDER BY page_number, chunk_index
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
            "SELECT * FROM chunks WHERE arxiv_id = ? ORDER BY page_number, chunk_index",
            (arxiv_id,),
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def chunks_by_id(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self.connection.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", tuple(chunk_ids)
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def body_chunk_ids(self, arxiv_id: str | None = None) -> list[str]:
        """Chunk ids excluding bibliographies, for filtering vector search."""
        if arxiv_id is None:
            rows = self.connection.execute("SELECT chunk_id FROM chunks WHERE kind = 'body'")
        else:
            rows = self.connection.execute(
                "SELECT chunk_id FROM chunks WHERE kind = 'body' AND arxiv_id = ?", (arxiv_id,)
            )
        return [row["chunk_id"] for row in rows]

    def iter_chunks(self):
        """Yield every chunk without holding the corpus in memory.

        all_chunks() materialises a list, which is fine for inspection and
        wrong for exporting a corpus of hundreds of thousands of passages.
        """
        cursor = self.connection.execute(
            """
            SELECT c.* FROM chunks AS c
            JOIN papers AS p ON p.arxiv_id = c.arxiv_id
            ORDER BY p.updated DESC, c.arxiv_id ASC, c.page_number ASC, c.chunk_index ASC
            """
        )
        for row in cursor:
            yield _chunk_from_row(row)

    def chunk_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def sync_watermark(self, category: str) -> str | None:
        """The newest `updated` timestamp already ingested for a category.

        Paging stops once results are older than this, so a repeat sync costs
        one request rather than walking the whole category again.
        """
        row = self.connection.execute(
            "SELECT newest_updated FROM sync_state WHERE category = ?", (category,)
        ).fetchone()
        return None if row is None else row["newest_updated"]

    def record_sync(self, category: str, newest_updated: str | None, papers_seen: int) -> None:
        """Advance the watermark, never retreat it.

        A sync bounded by --limit can return older papers than a previous run
        already covered; taking the max keeps the mark monotonic.
        """
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
        """Record which model produced a set of vectors, so they can be swapped.

        Re-registering with a different dimension or revision is rejected: those
        vectors are not comparable with the ones already stored.
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

    def pending_embeddings(self, model_id: str, *, limit: int | None = None) -> list[Chunk]:
        """Return chunks that have no vector from this model yet."""
        rows = self.connection.execute(
            """
            SELECT c.* FROM chunks AS c
            WHERE NOT EXISTS (
                SELECT 1 FROM embeddings AS e
                WHERE e.chunk_id = c.chunk_id AND e.model_id = ?
            )
            ORDER BY c.arxiv_id ASC, c.page_number ASC, c.chunk_index ASC
            LIMIT ?
            """,
            (model_id, -1 if limit is None else limit),
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
    ) -> list[ChunkMatch]:
        """Rank chunks by BM25 relevance to a keyword query.

        `arxiv_id` filters before ranking, so scores are not diluted by matches
        in papers the caller already excluded.
        """
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
            clauses.append("c.kind = 'body'")
        parameters.append(limit)
        rows = self.connection.execute(
            f"""
            SELECT c.*,
                   bm25(chunks_fts) AS relevance,
                   snippet(chunks_fts, 0, '[', ']', '…', 16) AS snippet
            FROM chunks_fts
            JOIN chunks AS c ON c.rowid = chunks_fts.rowid
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
        # Checked before the schema runs: CREATE VIRTUAL TABLE IF NOT EXISTS
        # cannot report whether it created the index, and COUNT(*) on an
        # external-content FTS5 table reads the content table rather than the
        # index, so neither can tell us the index needs a backfill.
        search_index_existed = self._table_exists("chunks_fts")
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
                    -- NULL in all three means arXiv has no rendering for this
                    -- paper: it was checked and ruled out, so it is not asked
                    -- for again. The PDF stays its source.
                    local_path TEXT,
                    sha256 TEXT,
                    byte_count INTEGER,
                    -- Whether this rendering is fit to be the paper's source.
                    -- 0 covers both "arXiv has none" and "arXiv has one but it
                    -- is truncated"; either way the PDF is used instead, and
                    -- note says which it was.
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

                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT NOT NULL
                        REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                    model_id TEXT NOT NULL REFERENCES embedding_models(model_id),
                    vector BLOB NOT NULL,
                    embedded_at TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, model_id)
                );

                CREATE INDEX IF NOT EXISTS embeddings_by_model ON embeddings (model_id);

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
            self._add_missing_columns()
            self._retire_renamed_columns()
            self._relax_html_source_nullability()
            # `usable` defaults to 1, which is wrong for rows written before it
            # existed to record that arXiv has no rendering at all.
            self.connection.execute(
                "UPDATE html_sources SET usable = 0,"
                " note = COALESCE(note, 'arXiv publishes no HTML rendering')"
                " WHERE sha256 IS NULL AND usable = 1"
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

    def _backfill_search_index(self) -> None:
        """Index chunks stored before the search index existed.

        The triggers only cover writes made after they were created, so a
        database upgraded in place would otherwise never match its own chunks.
        """
        self.connection.execute("INSERT INTO chunks_fts (chunks_fts) VALUES ('rebuild')")

    def _relax_html_source_nullability(self) -> None:
        """Let html_sources record a paper arXiv publishes no rendering for.

        The first version of the table required a path and a checksum, so the
        only representable state was "we hold this rendering". A paper with no
        rendering could not be written down, and so looked unchecked forever.
        """
        columns = self.connection.execute("PRAGMA table_info(html_sources)").fetchall()
        if not any(c["name"] == "sha256" and c["notnull"] for c in columns):
            return
        self.connection.executescript(
            """
            CREATE TABLE html_sources_rebuilt (
                arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
                url TEXT NOT NULL,
                local_path TEXT,
                sha256 TEXT,
                byte_count INTEGER,
                fetched_at TEXT NOT NULL,
                chunked_at TEXT,
                chunked_with TEXT,
                chunked_sha256 TEXT
            );
            INSERT INTO html_sources_rebuilt
                SELECT arxiv_id, url, local_path, sha256, byte_count,
                       fetched_at, chunked_at, chunked_with, chunked_sha256
                FROM html_sources;
            DROP TABLE html_sources;
            ALTER TABLE html_sources_rebuilt RENAME TO html_sources;
            """
        )

    def _retire_renamed_columns(self) -> None:
        """Finish the pdf_sha256 -> source_sha256 rename on an older database.

        Adding the new column is not enough: the old one is NOT NULL, so every
        insert fails on a constraint for a field the code no longer sets. Nor
        can it simply be dropped, because it sits inside a UNIQUE constraint,
        which SQLite refuses to alter.

        So the table is rebuilt. That also corrects the constraint itself: it
        keyed on the source checksum alone, but a paper may now hold both a PDF
        chunking and an HTML one, so source_kind belongs in the key. Values are
        carried across first, and the row count is checked before the old table
        is dropped.
        """
        present = {row["name"] for row in self.connection.execute("PRAGMA table_info(chunks)")}
        if "pdf_sha256" not in present or "source_sha256" not in present:
            return
        before = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        embeddings_before = self.connection.execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()[0]
        # embeddings.chunk_id cascades on delete, so dropping the old chunks
        # table would take every vector with it - hours of work, silently. The
        # rows are still valid because chunk_id is preserved by the copy, so
        # the constraint is suspended for the swap rather than obeyed.
        self.connection.execute("PRAGMA foreign_keys = OFF")
        self.connection.executescript(
            """
            DROP TRIGGER IF EXISTS chunks_fts_after_insert;
            DROP TRIGGER IF EXISTS chunks_fts_after_delete;
            DROP TRIGGER IF EXISTS chunks_fts_after_update;

            CREATE TABLE chunks_rebuilt (
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
                context TEXT,
                chunker_version TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'body',
                section_path TEXT NOT NULL DEFAULT '',
                source_kind TEXT NOT NULL DEFAULT 'pdf',
                created_at TEXT NOT NULL,
                UNIQUE (arxiv_id, source_kind, source_sha256, page_number, chunk_index)
            );

            INSERT INTO chunks_rebuilt
            SELECT chunk_id, arxiv_id,
                   CASE WHEN source_sha256 = '' THEN pdf_sha256 ELSE source_sha256 END,
                   source_url, license_url, page_number, chunk_index, char_start,
                   char_end, text, context, chunker_version, kind, section_path,
                   source_kind, created_at
            FROM chunks;
            """
        )
        after = self.connection.execute("SELECT COUNT(*) FROM chunks_rebuilt").fetchone()[0]
        if after != before:
            self.connection.execute("DROP TABLE chunks_rebuilt")
            raise RuntimeError(
                f"Refusing to migrate chunks: rebuilt {after} rows from {before}"
            )
        self.connection.executescript(
            """
            DROP TABLE chunks;
            ALTER TABLE chunks_rebuilt RENAME TO chunks;
            CREATE INDEX IF NOT EXISTS chunks_by_paper
                ON chunks (arxiv_id, page_number, chunk_index);
            """
        )
        self.connection.execute("PRAGMA foreign_keys = ON")
        embeddings_after = self.connection.execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()[0]
        if embeddings_after != embeddings_before:
            raise RuntimeError(
                f"Migration lost embeddings: {embeddings_before} -> {embeddings_after}"
            )
        # Rowids changed, so the external-content search index must be reloaded.
        self._backfill_search_index()

    def _add_missing_columns(self) -> None:
        """Bring databases created by an earlier schema up to date.

        Every change so far has been additive, so adding the absent columns is
        the whole migration.
        """
        for table, columns in ADDED_COLUMNS.items():
            present = {
                row["name"] for row in self.connection.execute(f"PRAGMA table_info({table})")
            }
            for name, declaration in columns.items():
                if name not in present:
                    self.connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
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


def pack_vector(vector: Sequence[float]) -> bytes:
    """Serialize a vector as little-endian float32, the layout sqlite-vec reads."""
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def build_fts_query(query: str, *, match: str = "all") -> str:
    """Turn a user query into a safe FTS5 MATCH expression.

    Terms are quoted so that punctuation common in scientific text — hyphens,
    colons, arXiv identifiers — cannot be read as FTS5 operators and raise a
    syntax error.  A trailing '*' still means prefix search, and double-quoted
    input is preserved as a phrase.

    `match="all"` requires every term, which suits an explicit keyword search.
    `match="any"` ranks by how many and how rare the matched terms are, which is
    what a natural-language question needs — requiring every word of "how does a
    changing field induce current" matches nothing at all.
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
    return Chunk(
        chunk_id=row["chunk_id"],
        arxiv_id=row["arxiv_id"],
        source_url=row["source_url"],
        license_url=row["license_url"],
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
