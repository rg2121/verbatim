"""Nearest-neighbour search over chunk embeddings."""

from __future__ import annotations

from collections.abc import Sequence
import sqlite3
from typing import Protocol

from verbatim.corpus import pack_vector

VECTOR_TABLE = "vec_chunks"
# Filters live in the index as metadata columns rather than being applied by
# passing chunk ids into the query: SQLite caps a statement at 32766 variables,
# which a body-only filter would exceed at roughly 500 papers.
VECTOR_COLUMNS = ("chunk_id", "arxiv_id", "kind", "embedding")


class VectorIndex(Protocol):
    """A nearest-neighbour index over chunk vectors.

    Kept deliberately small: sqlite-vec is brute force, so a corpus far past a
    few million vectors will want a different implementation behind this.
    """

    def count(self) -> int: ...

    def rebuild(self, model_id: str) -> int: ...

    def search(
        self,
        query: Sequence[float],
        *,
        limit: int,
        arxiv_id: str | None = None,
        kind: str | None = None,
    ) -> list[tuple[str, float]]: ...


def ensure_current(index: VectorIndex, corpus_vectors: int, model_id: str) -> bool:
    """Rebuild the index if it has drifted from the stored embeddings.

    The index is derived data, so reconciling by count is enough: chunks are
    only ever added or replaced wholesale, never edited in place.
    """
    if index.count() == corpus_vectors:
        return False
    index.rebuild(model_id)
    return True


class SqliteVecIndex:
    """sqlite-vec index living in the same file as the rest of the corpus.

    Measured on this machine: 455,000 vectors of 384 dimensions answer a k=50
    query in about 94 ms, filtered or not, in a 728 MB database. That covers a
    corpus of roughly five thousand papers with room to spare, which is why
    there is no separate vector server to run.
    """

    def __init__(self, connection: sqlite3.Connection, dimensions: int) -> None:
        self.connection = connection
        self.dimensions = dimensions
        self._load_extension()
        self._create_table()

    def count(self) -> int:
        return int(self.connection.execute(f"SELECT COUNT(*) FROM {VECTOR_TABLE}").fetchone()[0])

    def rebuild(self, model_id: str) -> int:
        """Reload every stored vector for one model, with its filter metadata.

        Cheaper than maintaining triggers against a virtual table, and the
        index is derived data that can always be rebuilt from `embeddings`.
        """
        rows = self.connection.execute(
            """
            SELECT e.chunk_id, e.vector, c.arxiv_id, c.kind
            FROM embeddings AS e
            JOIN chunks AS c ON c.chunk_id = e.chunk_id
            WHERE e.model_id = ?
            """,
            (model_id,),
        ).fetchall()
        with self.connection:
            self.connection.execute(f"DELETE FROM {VECTOR_TABLE}")
            self.connection.executemany(
                f"INSERT INTO {VECTOR_TABLE} (chunk_id, arxiv_id, kind, embedding)"
                " VALUES (?, ?, ?, ?)",
                [(row["chunk_id"], row["arxiv_id"], row["kind"], row["vector"]) for row in rows],
            )
        return len(rows)

    def search(
        self,
        query: Sequence[float],
        *,
        limit: int,
        arxiv_id: str | None = None,
        kind: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, distance) nearest first; smaller distance is closer.

        Filters are applied inside the KNN query, so a restricted search costs
        the same as an unrestricted one and stays exact.
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
        if kind is not None:
            clauses.append("kind = ?")
            parameters.append(kind)
        rows = self.connection.execute(
            f"SELECT chunk_id, distance FROM {VECTOR_TABLE}"
            f" WHERE {' AND '.join(clauses)} ORDER BY distance",
            parameters,
        ).fetchall()
        return [(row["chunk_id"], float(row["distance"])) for row in rows]

    def _create_table(self) -> None:
        # A table built before the metadata columns existed cannot be filtered,
        # so replace it; the index is derived and rebuilds from `embeddings`.
        existing = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (VECTOR_TABLE,)
        ).fetchone()
        if existing is not None and not all(column in existing[0] for column in VECTOR_COLUMNS):
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
        try:
            import sqlite_vec
        except ImportError as error:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "Vector search needs the 'sqlite-vec' package; install it with "
                "'uv add sqlite-vec'"
            ) from error
        try:
            self.connection.enable_load_extension(True)
            sqlite_vec.load(self.connection)
        finally:
            self.connection.enable_load_extension(False)
