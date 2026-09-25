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
# Which model's vectors each index currently holds. A vec0 table stores no
# such thing, and without it an index is only comparable to `embeddings` by
# counting rows - which cannot tell one model's vectors from another's of the
# same width.
INDEX_STATE_TABLE = "vector_index_state"


def _ensure_state_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""CREATE TABLE IF NOT EXISTS {INDEX_STATE_TABLE} (
            index_name TEXT PRIMARY KEY, model_id TEXT NOT NULL,
            generation INTEGER NOT NULL DEFAULT 0)"""
    )
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({INDEX_STATE_TABLE})")}
    if "generation" not in columns:  # a state table written before this existed
        connection.execute(
            f"ALTER TABLE {INDEX_STATE_TABLE} ADD COLUMN generation INTEGER NOT NULL DEFAULT 0"
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
    except sqlite3.OperationalError:  # the table predates this record
        return None, None
    return (None, None) if row is None else (str(row[0]), int(row[1]))


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
        kind: str | None = None,
        exclude_kind: str | None = None,
    ) -> list[tuple[str, float]]: ...


def ensure_current(
    index: VectorIndex,
    corpus_vectors: int,
    model_id: str,
    generation: int,
) -> bool:
    """Bring the index in line with the stored embeddings, cheaply.

    Identity is checked before size. `embeddings` is keyed by
    (chunk_id, model_id) so two models' vectors coexist there, while the index
    holds one set with no record of whose - and a count comparison cannot tell
    them apart when the widths match, which they do for the variants this
    project actually uses: `bge-small` and `bge-small+context` differ only in
    what was embedded. Searching one model's index with another's query vector
    returns confident nonsense and raises nothing, so a mismatch rebuilds.

    An index that has never recorded a model is rebuilt once for the same
    reason: it cannot be trusted to hold what the caller is about to ask for.

    Past that, a count comparison is the fast path, so a search over an
    up-to-date index costs one COUNT rather than two set-difference queries. On
    a mismatch the index is synced incrementally, not rebuilt, so the cost
    tracks what changed.

    The count is the third check rather than the only one, and the two ahead of
    it cover what it cannot see. Swapping equal numbers of vectors used to be
    invisible here; it is now caught by the generation, because every chunk
    write moves the corpus on. What remains outside all three is a vector
    changed in place under the same model and generation, which nothing in this
    pipeline does - `index --rebuild` is there for that and for any other
    disagreement worth forcing.
    """
    if index.indexed_model() != model_id:
        index.rebuild(model_id, generation)
        return True
    if index.indexed_generation() != generation:
        # Chunks were rewritten since this index was built. Ids are derived
        # from the source checksum, so a paper whose source was replaced has
        # entirely new ids and the same count - which the comparison below
        # reads as up to date while every one of those passages has silently
        # dropped out of vector search. Measured: a search returned 0 of 1.
        index.sync(model_id, generation)
        return True
    if index.count() == corpus_vectors:
        return False
    index.sync(model_id, generation)
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

    def indexed_model(self) -> str | None:
        return remembered_model(self.connection, VECTOR_TABLE)

    def indexed_generation(self) -> int | None:
        return remembered_generation(self.connection, VECTOR_TABLE)

    def sync(self, model_id: str, generation: int | None = None) -> IndexSync:
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
        remember_model(self.connection, VECTOR_TABLE, model_id, generation or 0)
        return IndexSync(added=max(added, 0), removed=max(removed, 0))

    def rebuild(self, model_id: str, generation: int | None = None) -> int:
        """Discard the index and rebuild it from scratch.

        Only needed when the index may disagree with `embeddings` in a way the
        incremental sync cannot see, such as after changing how metadata is
        derived.
        """
        with self.connection:
            self.connection.execute(f"DELETE FROM {VECTOR_TABLE}")
        return self.sync(model_id, generation).added

    def search(
        self,
        query: Sequence[float],
        *,
        limit: int,
        arxiv_id: str | None = None,
        kind: str | None = None,
        exclude_kind: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, distance) nearest first; smaller distance is closer.

        Filters are applied inside the KNN query, so a restricted search costs
        the same as an unrestricted one and stays exact. `exclude_kind` is
        what retrieval uses: it names the one kind to keep out, so a kind
        added later is searched rather than silently dropped.
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
        # A table built before the metadata columns existed cannot be filtered,
        # so replace it; the index is derived and rebuilds from `embeddings`.
        # The same applies to its width: a vec0 table declares its dimension in
        # its schema and CREATE IF NOT EXISTS silently keeps the old one, so
        # opening a 384-wide index at 768 used to succeed and then fail deep
        # inside an insert with a dimension mismatch nobody could act on.
        existing = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (VECTOR_TABLE,)
        ).fetchone()
        stale = existing is not None and (
            not all(column in existing[0] for column in VECTOR_COLUMNS)
            or f"float[{self.dimensions}]" not in existing[0]
        )
        if stale:
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
        existing = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (PAPER_VECTOR_TABLE,),
        ).fetchone()
        if existing is not None and f"float[{dimensions}]" not in existing[0]:
            # Same reasoning as the passage index: the width is part of the
            # schema, and a mismatch is a rebuild rather than an error later.
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
