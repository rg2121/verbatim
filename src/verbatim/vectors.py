"""Nearest-neighbour search over chunk embeddings."""

from __future__ import annotations

from collections.abc import Sequence
import sqlite3
from typing import Protocol

from verbatim.corpus import pack_vector

VECTOR_TABLE = "vec_chunks"


class VectorIndex(Protocol):
    """A nearest-neighbour index over chunk vectors.

    Kept deliberately small: sqlite-vec is brute force and will not carry a
    corpus of millions, so the replacement should be a drop-in.
    """

    def count(self) -> int: ...

    def rebuild(self, model_id: str) -> int: ...

    def search(
        self, query: Sequence[float], *, limit: int, chunk_ids: Sequence[str] | None = None
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
    """sqlite-vec index living in the same file as the rest of the corpus."""

    def __init__(self, connection: sqlite3.Connection, dimensions: int) -> None:
        self.connection = connection
        self.dimensions = dimensions
        self._load_extension()
        self.connection.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {VECTOR_TABLE} USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding float[{dimensions}]
            )
            """
        )

    def count(self) -> int:
        return int(self.connection.execute(f"SELECT COUNT(*) FROM {VECTOR_TABLE}").fetchone()[0])

    def rebuild(self, model_id: str) -> int:
        """Reload every stored vector for one model.

        Cheaper than maintaining triggers against a virtual table, and the
        index is derived data that can always be rebuilt from `embeddings`.
        """
        rows = self.connection.execute(
            "SELECT chunk_id, vector FROM embeddings WHERE model_id = ?", (model_id,)
        ).fetchall()
        with self.connection:
            self.connection.execute(f"DELETE FROM {VECTOR_TABLE}")
            self.connection.executemany(
                f"INSERT INTO {VECTOR_TABLE} (chunk_id, embedding) VALUES (?, ?)",
                [(row["chunk_id"], row["vector"]) for row in rows],
            )
        return len(rows)

    def search(
        self, query: Sequence[float], *, limit: int, chunk_ids: Sequence[str] | None = None
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, distance) nearest first; smaller distance is closer.

        `chunk_ids` restricts the search to a subset.  sqlite-vec's KNN operator
        cannot express that filter, so a restricted search scans the subset
        exactly instead of approximating over the whole corpus.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if len(query) != self.dimensions:
            raise ValueError(
                f"Query vector has {len(query)} dimensions, index expects {self.dimensions}"
            )
        packed = pack_vector(query)
        if chunk_ids is None:
            rows = self.connection.execute(
                f"""
                SELECT chunk_id, distance FROM {VECTOR_TABLE}
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (packed, limit),
            ).fetchall()
            return [(row["chunk_id"], float(row["distance"])) for row in rows]

        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self.connection.execute(
            f"""
            SELECT chunk_id, vec_distance_l2(embedding, ?) AS distance
            FROM {VECTOR_TABLE}
            WHERE chunk_id IN ({placeholders})
            ORDER BY distance
            LIMIT ?
            """,
            (packed, *chunk_ids, limit),
        ).fetchall()
        return [(row["chunk_id"], float(row["distance"])) for row in rows]

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
