"""Nearest-neighbour search over chunk embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import sqlite3
from typing import Protocol

from verbatim.corpus import pack_vector

VECTOR_TABLE = "vec_chunks"
PAPER_VECTOR_TABLE = "vec_papers"
# Filters live in the index as metadata columns rather than being applied by
# passing chunk ids into the query: SQLite caps a statement at 32766 variables,
# which a body-only filter would exceed at roughly 500 papers.
VECTOR_COLUMNS = ("chunk_id", "arxiv_id", "kind", "embedding")


@dataclass(frozen=True)
class IndexSync:
    """How much the index changed when it was last reconciled."""

    added: int
    removed: int


class VectorIndex(Protocol):
    """A nearest-neighbour index over chunk vectors.

    Kept deliberately small: sqlite-vec is brute force, so a corpus far past a
    few million vectors will want a different implementation behind this.
    """

    def count(self) -> int: ...

    def sync(self, model_id: str) -> IndexSync: ...

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
    """Bring the index in line with the stored embeddings, cheaply.

    A count comparison is the fast path, so a search over an up-to-date index
    costs one COUNT rather than two set-difference queries.  On a mismatch the
    index is synced incrementally, not rebuilt, so the cost tracks what changed.

    The count check cannot see a change that adds and removes equal numbers of
    vectors; `verbatim index --rebuild` forces a full reload for that case.
    """
    if index.count() == corpus_vectors:
        return False
    index.sync(model_id)
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

    def sync(self, model_id: str) -> IndexSync:
        """Add vectors the index is missing and drop ones it should not hold.

        Work scales with what changed rather than with corpus size, which is
        what keeps adding a paper cheap as the corpus grows: measured over
        455,000 vectors, adding one paper's 91 chunks took 0.38s against 2.9s
        to rebuild everything.
        """
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
        return IndexSync(added=max(added, 0), removed=max(removed, 0))

    def rebuild(self, model_id: str) -> int:
        """Discard the index and rebuild it from scratch.

        Only needed when the index may disagree with `embeddings` in a way the
        incremental sync cannot see, such as after changing how metadata is
        derived.
        """
        with self.connection:
            self.connection.execute(f"DELETE FROM {VECTOR_TABLE}")
        return self.sync(model_id).added

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
        load_sqlite_vec(self.connection)


class SqlitePaperIndex:
    """Nearest-neighbour index over paper abstracts, not passages.

    Answers "which papers are about this?" rather than "which passages mention
    this?". The distinction is the point: a passage can mention a subject in
    passing - a pretraining-data paper quoting sample text about nuclear
    forces - while an abstract states what the work is actually about.
    """

    def __init__(self, connection: sqlite3.Connection, dimensions: int) -> None:
        self.connection = connection
        self.dimensions = dimensions
        load_sqlite_vec(self.connection)
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

    def sync(self, model_id: str) -> IndexSync:
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
        return IndexSync(added=max(added, 0), removed=max(removed, 0))

    def rebuild(self, model_id: str) -> int:
        with self.connection:
            self.connection.execute(f"DELETE FROM {PAPER_VECTOR_TABLE}")
        return self.sync(model_id).added

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
