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

from verbatim.arxiv import Paper
from verbatim.chunking import Chunk
from verbatim.downloads import DownloadedPaper

SCHEMA_VERSION = "5"
# Terms are quoted before reaching FTS5, so a query may still use a trailing
# '*' for prefix matching and double quotes for phrases.
FTS_TERM_PATTERN = re.compile(r'"[^"]*"\*?|\S+')

# Columns added to existing databases after their table was first created.
ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "downloads": {
        "chunked_at": "TEXT",
        "chunked_with": "TEXT",
        "chunked_sha256": "TEXT",
    },
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
        return [_download_from_row(row) for row in rows]

    def pending_chunking(self, fingerprint: str) -> list[DownloadedPaper]:
        """Return PDFs whose stored chunks do not match the current chunker.

        Re-chunking is needed when a PDF has never been chunked, when the
        chunker version or its parameters changed, or when the file on disk was
        replaced since it was last chunked.
        """
        rows = self.connection.execute(
            """
            SELECT d.* FROM downloads AS d
            WHERE d.chunked_at IS NULL
               OR d.chunked_with IS NOT ?
               OR d.chunked_sha256 IS NOT d.sha256
            ORDER BY d.arxiv_id ASC
            """,
            (fingerprint,),
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

    def chunk_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

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
                    downloaded_at TEXT NOT NULL,
                    chunked_at TEXT,
                    chunked_with TEXT,
                    chunked_sha256 TEXT
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
