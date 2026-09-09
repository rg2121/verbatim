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
from verbatim.references import Reference

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
# Long enough for one stage's write to finish while another stage waits.
# Chunking commits in large batches and holds the write lock for a long time
# while its FTS triggers run. A minute was not enough: it killed a six-hour
# HTML fetch outright.
BUSY_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class PaperFilter:
    """Restrict retrieval to papers by date or subject.

    Similarity search has no notion of recency and a keyword index has no
    notion of subject, so "what came out in cs.CL this month" is a question
    neither can answer on its own. This narrows the corpus first, and both
    legs then rank within it.
    """

    since: str | None = None
    category: str | None = None

    def __bool__(self) -> bool:
        return bool(self.since or self.category)

    def clause(self, alias: str = "c") -> tuple[str, list[object]]:
        """A SQL fragment restricting `alias` to matching papers.

        Written as EXISTS rather than an id list because SQLite caps a
        statement at 32,766 variables, which any useful filter would exceed.
        """
        if not self:
            return "", []
        conditions = [f"p.arxiv_id = {alias}.arxiv_id"]
        parameters: list[object] = []
        if self.since:
            conditions.append("p.published >= ?")
            parameters.append(self.since)
        if self.category:
            # categories_json is a JSON array; match the quoted element so
            # "cs.CL" does not also match "cs.CLX".
            conditions.append("p.categories_json LIKE ?")
            parameters.append(f'%"{self.category}"%')
        return (
            f"EXISTS (SELECT 1 FROM papers AS p WHERE {' AND '.join(conditions)})",
            parameters,
        )

    def matches(self, published: str, categories: Sequence[str]) -> bool:
        """Whether one paper passes, for filtering results the index returned."""
        if self.since and published < self.since:
            return False
        return not self.category or self.category in categories


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
        # Kept so a worker thread can open its own connection to the same
        # database: SQLite connections belong to the thread that made them.
        self.path = Path(path)
        self.connection = sqlite3.connect(path, timeout=BUSY_TIMEOUT_SECONDS)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        # Stages run for hours, and the point of `stats` is to watch one while
        # it runs. Under the default rollback journal a reader and a writer
        # cannot overlap at all, and a second writer fails at once rather than
        # waiting - which is how a licence backfill and a chunking run
        # collided. WAL lets readers proceed during a write, and the busy
        # timeout makes a second writer wait its turn instead of failing.
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
        self,
        fingerprint: str,
        *,
        limit: int | None = None,
        require_html_checked: bool = False,
    ) -> list[DownloadedPaper]:
        """Return PDFs whose stored chunks do not match the current chunker.

        Re-chunking is needed when a PDF has never been chunked, when the
        chunker version or its parameters changed, or when the file on disk was
        replaced since it was last chunked.

        `require_html_checked` holds back papers whose HTML has not been looked
        for yet. Without it, chunking during an unfinished `html` run reads a
        paper from its PDF, and the rendering that arrives minutes later throws
        that work away along with every vector made from it. It is off by
        default because a corpus where `html` is never run must still chunk.
        """
        rows = self.connection.execute(
            f"""
            SELECT d.* FROM downloads AS d
            WHERE {"EXISTS (SELECT 1 FROM html_sources AS c WHERE c.arxiv_id = d.arxiv_id) AND"
                   if require_html_checked else ""}
            NOT EXISTS (
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

    def record_chunks(
        self, chunks: list[Chunk], for_papers: Sequence[str] | None = None
    ) -> None:
        """Replace the stored chunks for every paper this pass covered.

        A paper is held as exactly one chunking, from one source. Deleting only
        the chunks sharing a checksum would leave a paper's PDF chunks behind
        when it was re-read from its HTML, so the same passages would sit in
        the index twice, under two sources, and be retrieved twice.

        `for_papers` is the papers that were *attempted*, which is not the same
        as the papers that produced something. Derived from the chunks alone, a
        re-chunk that yielded nothing deleted nothing: a paper whose PDF now
        fails to extract - or exceeds the layout timeout, which is recorded as
        chunked so it is not retried for ever - kept its old passages and their
        vectors, still searchable and still citable, under a checksum the file
        no longer has. Callers that know what they attempted should say so.
        """
        now = _timestamp()
        covered = set(for_papers or ()) | {chunk.arxiv_id for chunk in chunks}
        with self.connection:
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
                "SELECT COUNT(*) FROM html_sources AS h WHERE h.usable = 1"
                " AND NOT EXISTS (SELECT 1 FROM paper_references AS r"
                "   WHERE r.citing_arxiv_id = h.arxiv_id AND r.source_sha256 = h.sha256)"
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

    def pending_embeddings(
        self,
        model_id: str,
        *,
        limit: int | None = None,
        chunker_version: str | None = None,
    ) -> list[Chunk]:
        """Return chunks that have no vector from this model yet.

        `chunker_version` restricts the work to passages the current chunker
        actually produces. A chunk left over from an older chunker is going to
        be deleted the next time its paper is chunked, and the vector made from
        it deleted with it: embedding one is work that is guaranteed to be
        thrown away. Measured mid-migration here, 490,620 of 729,347 chunks
        were in that state, which is around eleven hours of embedding.

        Off by default, because those chunks are still the only ones a paper
        has until it is re-chunked, and a corpus that is not mid-migration
        should embed everything it holds.
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

    def unavailable_downloads(self) -> int:
        return int(
            self.connection.execute("SELECT COUNT(*) FROM download_failures").fetchone()[0]
        )

    def record_references(self, references: list[Reference], source_sha256: str) -> None:
        """Replace the stored bibliography of every paper represented here."""
        now = _timestamp()
        with self.connection:
            for citing in {reference.citing_arxiv_id for reference in references}:
                self.connection.execute(
                    "DELETE FROM paper_references WHERE citing_arxiv_id = ?", (citing,)
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
            """
            SELECT h.arxiv_id FROM html_sources AS h
            WHERE h.usable = 1
              AND NOT EXISTS (
                  SELECT 1 FROM paper_references AS r
                  WHERE r.citing_arxiv_id = h.arxiv_id
                    AND r.source_sha256 = h.sha256
              )
            ORDER BY h.arxiv_id LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [row["arxiv_id"] for row in rows]

    def most_cited_absent_papers(self, *, limit: int = 50) -> list[tuple[str, int]]:
        """arXiv papers this corpus cites often and does not hold.

        The corpus is a slice of the newest papers in each category, so the work
        it rests on is mostly outside it. This is the reading list that would
        fix that, ordered by how many held papers ask for each one.
        """
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
        """Held papers whose bibliography cites this paper, newest first.

        Keyed on the base identifier because that is what a bibliography names:
        a reference to "arXiv:1412.6980" is a claim about the paper, not about
        any one version of it. Self-citations are excluded - a preprint listing
        its own identifier is provenance, not influence.
        """
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
        # The corpus can hold several versions of a citing paper, each with its
        # own stored bibliography; one paper still cites, however many times it
        # was read. Newest-first order makes this keep the current version.
        citers: dict[str, Paper] = {}
        for row in rows:
            citers.setdefault(row["base_arxiv_id"], _paper_from_row(row))
            if len(citers) == limit:
                break
        return list(citers.values())

    def papers_cited_by(self, arxiv_id: str, *, limit: int = 50) -> list[Paper]:
        """Held papers this paper's bibliography resolves to, in citation order.

        The outward half of the graph: what a paper builds on, as far as the
        corpus holds it. References to absent papers are `grow`'s business and
        do not appear here.
        """
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
        """How many held papers cite each of these, keyed by base identifier.

        Counts distinct citing papers, not bibliography rows, so a paper held
        at two versions - each with its stored bibliography - counts once.
        Identifiers nothing cites are simply absent from the result.
        """
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
        """The papers this corpus itself cites most, with their citer counts.

        The held-paper counterpart of most_cited_absent_papers: that one is a
        reading list, this one is an authority ranking. "What is the most
        influential paper on X here" is a graph question, and no similarity
        score answers it.
        """
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
        """The size and reach of the internal citation graph.

        An edge is one held paper citing another, counted once regardless of
        how many versions of the citer were read. Connected means touching at
        least one edge in either direction.
        """
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
        """Current papers with no abstract vector for this model.

        Only current versions: an abstract index exists to say what the corpus
        is about, and superseded versions would answer that twice.
        """
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
        """Rank chunks by BM25 relevance to a keyword query.

        `arxiv_id` and `papers` filter before ranking, so scores are not
        diluted by matches in papers the caller already excluded.
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
        if papers:
            clause, filter_parameters = papers.clause("c")
            clauses.append(clause)
            parameters.extend(filter_parameters)
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

                -- One vector per paper, over its title and abstract. Separate
                -- from `embeddings` because it answers a different question:
                -- a chunk vector says what a passage mentions, a paper vector
                -- says what the paper is about. A paper can mention anything
                -- in passing; its abstract states its subject.
                -- One row per bibliography entry. cited_arxiv_id is set only
                -- when the entry states an identifier; a reference resolved by
                -- title similarity would be a guess recorded as a fact.
                -- Papers arXiv has no PDF for at the version we hold. Kept
                -- so they are not asked for again on every run and not
                -- reported as outstanding work for ever.
                CREATE TABLE IF NOT EXISTS download_failures (
                    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
                    pdf_url TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    failed_at TEXT NOT NULL
                );
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

                -- Titles, searched separately from passages. A definitional
                -- question loses to the papers that merely use the thing:
                -- 3,152 papers here mention Adam and the one that defines it
                -- holds 53 passages. A title says what a paper is called,
                -- which is the signal a passage cannot carry.
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

    def papers_matching(
        self, papers: "PaperFilter", arxiv_ids: Sequence[str]
    ) -> set[str]:
        """Which of these papers pass the filter.

        Asked in one query over a bounded list - the ids the vector index just
        returned - rather than over the corpus, so it stays well inside
        SQLite's limit on statement variables.
        """
        if not papers or not arxiv_ids:
            return set(arxiv_ids)
        unique = list(dict.fromkeys(arxiv_ids))
        conditions = ["arxiv_id IN (" + ",".join("?" * len(unique)) + ")"]
        parameters: list[object] = list(unique)
        if papers.since:
            conditions.append("published >= ?")
            parameters.append(papers.since)
        if papers.category:
            conditions.append("categories_json LIKE ?")
            parameters.append(f'%"{papers.category}"%')
        rows = self.connection.execute(
            f"SELECT arxiv_id FROM papers WHERE {' AND '.join(conditions)}", parameters
        ).fetchall()
        return {row["arxiv_id"] for row in rows}

    def search_titles(self, query: str, *, limit: int = 10) -> list[tuple[str, float]]:
        """Papers whose title matches the query, best first.

        Separate from passage search because it answers a different question.
        A passage says what a paper mentions; a title says what it is called,
        and for "what is the Adam optimizer" the paper called "Adam: A Method
        for Stochastic Optimization" is the answer while three thousand papers
        that merely used Adam are not.
        """
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
            # Same guard as passage search: a query that FTS5 cannot parse is
            # a miss, not a crash.
            return []
        return [(row["arxiv_id"], float(row["relevance"])) for row in rows]

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


# A title is a handful of words, so a match on "what" or "is" carries almost
# no information and yet dominates BM25 over so short a field. Passage search
# does not need this: a match on a stopword inside 1,200 characters is diluted
# by everything around it.
TITLE_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from how in into is it its of on
    or that the their there this to was were what when where which who why with
    you your""".split()
)


def build_title_query(query: str) -> str:
    """An FTS5 expression over titles, with the question words removed.

    Returns an empty string when nothing but stopwords is left, which is a
    question with no subject to match a title against.
    """
    content = [
        word
        for word in re.findall(r"[\w'-]+", query.lower())
        if word not in TITLE_STOPWORDS and len(word) > 1
    ]
    return build_fts_query(" ".join(content), match="any")


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
