"""Nearest-neighbour search over chunk embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import sqlite3
from typing import Protocol

from verbatim.corpus import pack_vector

VECTOR_TABLE = "vec_chunks"
PAPER_VECTOR_TABLE = "vec_papers"
# Filters are metadata columns inside the index; passing chunk ids into the
# query instead would exceed SQLite's variable limit.
# Which model and corpus generation each index holds; a vec0 table records neither.
INDEX_STATE_TABLE = "vector_index_state"


def _ensure_state_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""CREATE TABLE IF NOT EXISTS {INDEX_STATE_TABLE} (
            index_name TEXT PRIMARY KEY, model_id TEXT NOT NULL,
            generation INTEGER NOT NULL DEFAULT 0)"""
    )


def remember_model(
    connection: sqlite3.Connection, index_name: str, model_id: str, generation: int = 0
) -> None:
    """Record whose vectors an index holds, and which corpus they came from."""
    _ensure_state_table(connection)
    with connection:
        connection.execute(
            f"""INSERT INTO {INDEX_STATE_TABLE} (index_name, model_id, generation)
                VALUES (?, ?, ?)
                ON CONFLICT(index_name) DO UPDATE SET
                    model_id = excluded.model_id, generation = excluded.generation""",
            (index_name, model_id, generation),
        )


def remembered_model(connection: sqlite3.Connection, index_name: str) -> str | None:
    """Whose vectors an index holds, or None when it has never said."""
    return _remembered(connection, index_name)[0]


def remembered_generation(connection: sqlite3.Connection, index_name: str) -> int | None:
    """Which corpus generation those vectors were taken from."""
    return _remembered(connection, index_name)[1]


def _remembered(
    connection: sqlite3.Connection, index_name: str
) -> tuple[str | None, int | None]:
    try:
        row = connection.execute(
            f"SELECT model_id, generation FROM {INDEX_STATE_TABLE} WHERE index_name = ?",
            (index_name,),
        ).fetchone()
    except sqlite3.OperationalError:  # no index has been recorded yet
        return None, None
    return (None, None) if row is None else (str(row[0]), int(row[1]))


@dataclass(frozen=True)
class IndexSync:
    """How much the index changed when it was last reconciled."""

    added: int
    removed: int


class VectorIndex(Protocol):
    """A nearest-neighbour index over chunk vectors."""

    def count(self) -> int: ...

    def indexed_model(self) -> str | None: ...

    def indexed_generation(self) -> int | None: ...

    def sync(self, model_id: str, generation: int | None = None) -> IndexSync: ...

    def rebuild(self, model_id: str, generation: int | None = None) -> int: ...

    def search(
        self,
        query: Sequence[float],
        *,
        limit: int,
        arxiv_id: str | None = None,
        exclude_kind: str | None = None,
    ) -> list[tuple[str, float]]: ...


def ensure_current(
    index: VectorIndex,
    corpus_vectors: int,
    model_id: str,
    generation: int,
) -> bool:
    """Bring the index in line with the stored embeddings; returns whether it changed.

    A different model rebuilds it: same-width vectors from another model would
    silently return nonsense. A different generation (chunks rewritten, perhaps
    with new ids but the same count) or a count mismatch syncs it. An index that
    is already current costs one COUNT.
    """
    if index.indexed_model() != model_id:
        index.rebuild(model_id, generation)
        return True
    if index.indexed_generation() != generation:
        index.sync(model_id, generation)
        return True
    if index.count() == corpus_vectors:
        return False
    index.sync(model_id, generation)
    return True


class SqliteVecIndex:
    """sqlite-vec index in the corpus file; k=50 over 455,000 vectors takes about 94 ms."""

    def __init__(self, connection: sqlite3.Connection, dimensions: int) -> None:
        self.connection = connection
        self.dimensions = dimensions
        self._load_extension()
        self._create_table()

    def count(self) -> int:
        return int(self.connection.execute(f"SELECT COUNT(*) FROM {VECTOR_TABLE}").fetchone()[0])

    def indexed_model(self) -> str | None:
        return remembered_model(self.connection, VECTOR_TABLE)

    def indexed_generation(self) -> int | None:
        return remembered_generation(self.connection, VECTOR_TABLE)

    def sync(self, model_id: str, generation: int | None = None) -> IndexSync:
        """Add the vectors the index is missing and drop the ones it should not hold."""
        with self.connection:
            added = self.connection.execute(
                f"""
                INSERT INTO {VECTOR_TABLE} (chunk_id, arxiv_id, kind, embedding)
                SELECT e.chunk_id, c.arxiv_id, c.kind, e.vector
                FROM embeddings AS e
                JOIN chunks AS c ON c.chunk_id = e.chunk_id
                WHERE e.model_id = ?
                  AND e.chunk_id NOT IN (SELECT chunk_id FROM {VECTOR_TABLE})
                """,
                (model_id,),
            ).rowcount
            removed = self.connection.execute(
                f"""
                DELETE FROM {VECTOR_TABLE}
                WHERE chunk_id NOT IN (
                    SELECT chunk_id FROM embeddings WHERE model_id = ?
                )
                """,
                (model_id,),
            ).rowcount
        remember_model(self.connection, VECTOR_TABLE, model_id, generation or 0)
        return IndexSync(added=max(added, 0), removed=max(removed, 0))

    def rebuild(self, model_id: str, generation: int | None = None) -> int:
        """Discard the index and rebuild it from `embeddings`."""
        with self.connection:
            self.connection.execute(f"DELETE FROM {VECTOR_TABLE}")
        return self.sync(model_id, generation).added

    def search(
        self,
        query: Sequence[float],
        *,
        limit: int,
        arxiv_id: str | None = None,
        exclude_kind: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, distance), nearest first, filtering inside the KNN query.

        `exclude_kind` names the one kind to keep out, so a new kind is searched.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if len(query) != self.dimensions:
            raise ValueError(
                f"Query vector has {len(query)} dimensions, index expects {self.dimensions}"
            )
        clauses = ["embedding MATCH ?", "k = ?"]
        parameters: list[object] = [pack_vector(query), limit]
        if arxiv_id is not None:
            clauses.append("arxiv_id = ?")
            parameters.append(arxiv_id)
        if exclude_kind is not None:
            clauses.append("kind != ?")
            parameters.append(exclude_kind)
        rows = self.connection.execute(
            f"SELECT chunk_id, distance FROM {VECTOR_TABLE}"
            f" WHERE {' AND '.join(clauses)} ORDER BY distance",
            parameters,
        ).fetchall()
        return [(row["chunk_id"], float(row["distance"])) for row in rows]

    def _create_table(self) -> None:
        # A vec0 table fixes its width in its schema, and CREATE IF NOT EXISTS
        # would keep the old one, so an index of another width is replaced.
        existing = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (VECTOR_TABLE,)
        ).fetchone()
        if existing is not None and f"float[{self.dimensions}]" not in existing[0]:
            self.connection.execute(f"DROP TABLE {VECTOR_TABLE}")
        self.connection.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {VECTOR_TABLE} USING vec0(
                chunk_id TEXT PRIMARY KEY,
                arxiv_id TEXT,
                kind TEXT,
                embedding float[{self.dimensions}]
            )
            """
        )

    def _load_extension(self) -> None:
        load_sqlite_vec(self.connection)


class SqlitePaperIndex:
    """Nearest-neighbour index over paper abstracts.

    It answers which papers are about a subject, not which passages mention it.
    """

    def __init__(self, connection: sqlite3.Connection, dimensions: int) -> None:
        self.connection = connection
        self.dimensions = dimensions
        load_sqlite_vec(self.connection)
        existing = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (PAPER_VECTOR_TABLE,),
        ).fetchone()
        if existing is not None and f"float[{dimensions}]" not in existing[0]:
            # A different width is replaced, as for the passage index.
            self.connection.execute(f"DROP TABLE {PAPER_VECTOR_TABLE}")
        self.connection.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {PAPER_VECTOR_TABLE} USING vec0(
                arxiv_id TEXT PRIMARY KEY,
                embedding float[{self.dimensions}]
            )
            """
        )

    def count(self) -> int:
        return int(
            self.connection.execute(f"SELECT COUNT(*) FROM {PAPER_VECTOR_TABLE}").fetchone()[0]
        )

    def indexed_model(self) -> str | None:
        return remembered_model(self.connection, PAPER_VECTOR_TABLE)

    def indexed_generation(self) -> int | None:
        return remembered_generation(self.connection, PAPER_VECTOR_TABLE)

    def sync(self, model_id: str, generation: int | None = None) -> IndexSync:
        """Reconcile the index with the stored paper vectors."""
        with self.connection:
            added = self.connection.execute(
                f"""
                INSERT INTO {PAPER_VECTOR_TABLE} (arxiv_id, embedding)
                SELECT e.arxiv_id, e.vector FROM paper_embeddings AS e
                WHERE e.model_id = ?
                  AND e.arxiv_id NOT IN (SELECT arxiv_id FROM {PAPER_VECTOR_TABLE})
                """,
                (model_id,),
            ).rowcount
            removed = self.connection.execute(
                f"""
                DELETE FROM {PAPER_VECTOR_TABLE}
                WHERE arxiv_id NOT IN (
                    SELECT arxiv_id FROM paper_embeddings WHERE model_id = ?
                )
                """,
                (model_id,),
            ).rowcount
        remember_model(self.connection, PAPER_VECTOR_TABLE, model_id, generation or 0)
        return IndexSync(added=max(added, 0), removed=max(removed, 0))

    def rebuild(self, model_id: str, generation: int | None = None) -> int:
        with self.connection:
            self.connection.execute(f"DELETE FROM {PAPER_VECTOR_TABLE}")
        return self.sync(model_id, generation).added

    def search(self, query: Sequence[float], *, limit: int) -> list[tuple[str, float]]:
        """Return (arxiv_id, distance) nearest first; smaller distance is closer."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if len(query) != self.dimensions:
            raise ValueError(
                f"Query vector has {len(query)} dimensions, index expects {self.dimensions}"
            )
        rows = self.connection.execute(
            f"SELECT arxiv_id, distance FROM {PAPER_VECTOR_TABLE}"
            " WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (pack_vector(query), limit),
        ).fetchall()
        return [(row["arxiv_id"], float(row["distance"])) for row in rows]


def load_sqlite_vec(connection: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into one connection."""
    try:
        import sqlite_vec
    except ImportError as error:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "Vector search needs the 'sqlite-vec' package; install it with "
            "'uv add sqlite-vec'"
        ) from error
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
    finally:
        connection.enable_load_extension(False)
