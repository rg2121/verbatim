"""Export a small self-contained corpus, so a fresh clone works without an ingest.

The sample is the database only: papers, passages, vectors and bibliography
records, with no PDF or HTML files.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

from verbatim.corpus import CorpusStore
from verbatim.vectors import SqlitePaperIndex, SqliteVecIndex

# Tables copied for the selected papers, in foreign-key order. Download and HTML
# records are left out: the sample ships no files, and `stats` would count them.
PAPER_SCOPED_TABLES = (
    ("papers", "arxiv_id"),
    # Bibliography records make `graph` work; bibliography chunks are left out,
    # since retrieval excludes them.
    ("paper_references", "citing_arxiv_id"),
    ("chunks", "arxiv_id"),
)
# A paper long enough to demonstrate retrieval, short enough that forty of
# them stay small.
MIN_PASSAGES = 40
MAX_PASSAGES = 120
# Slots reserved for a hub paper and the papers that cite it, so the sample's
# citation graph has edges to show rather than forty mutual strangers.
NEIGHBORHOOD_SHARE = 4


@dataclass(frozen=True)
class SampleReport:
    """What a sample ended up containing."""

    papers: int
    chunks: int
    vectors: int
    bytes_written: int
    references: int = 0

    def render(self) -> str:
        megabytes = self.bytes_written / 1_000_000
        return (
            f"{self.papers} papers, {self.chunks:,} passages, "
            f"{self.vectors:,} vectors, {self.references:,} bibliography "
            f"entries, {megabytes:.1f} MB"
        )


# Papers a sample may hold: current, `open`-licensed, embedded, ordinary length.
# Every selection path uses this, so none bypasses the licence gate.
ELIGIBLE_SQL = """
    SELECT p.arxiv_id, p.base_arxiv_id, p.categories_json
    FROM papers AS p
    JOIN chunks AS c ON c.arxiv_id = p.arxiv_id AND c.kind = 'body'
    JOIN embeddings AS e ON e.chunk_id = c.chunk_id AND e.model_id = :model_id
    WHERE p.is_latest = 1 AND p.license_class = 'open'
    GROUP BY p.arxiv_id
    HAVING COUNT(e.chunk_id) BETWEEN :min_passages AND :max_passages
"""


def connected_neighborhood(
    source: sqlite3.Connection, *, model_id: str, limit: int
) -> list[str]:
    """An eligible paper and the eligible papers citing it, so the sample's graph has edges."""
    if limit < 2:
        return []
    parameters = {
        "model_id": model_id,
        "min_passages": MIN_PASSAGES,
        "max_passages": MAX_PASSAGES,
    }
    hub = source.execute(
        f"""
        WITH eligible AS ({ELIGIBLE_SQL})
        SELECT held.arxiv_id AS arxiv_id, held.base_arxiv_id AS base_arxiv_id
        FROM paper_references AS r
        JOIN eligible AS held ON held.base_arxiv_id = r.cited_arxiv_id
        JOIN eligible AS citer ON citer.arxiv_id = r.citing_arxiv_id
        WHERE r.cited_arxiv_id != citer.base_arxiv_id
        GROUP BY held.arxiv_id
        ORDER BY COUNT(DISTINCT citer.base_arxiv_id) DESC, held.arxiv_id
        LIMIT 1
        """,
        parameters,
    ).fetchone()
    if hub is None:
        return []
    citers = source.execute(
        f"""
        WITH eligible AS ({ELIGIBLE_SQL})
        SELECT DISTINCT citer.arxiv_id
        FROM paper_references AS r
        JOIN eligible AS citer ON citer.arxiv_id = r.citing_arxiv_id
        WHERE r.cited_arxiv_id = :base_arxiv_id
        ORDER BY citer.arxiv_id
        LIMIT :citers
        """,
        {**parameters, "base_arxiv_id": hub["base_arxiv_id"], "citers": limit - 1},
    ).fetchall()
    return [hub["arxiv_id"], *(row["arxiv_id"] for row in citers)]


def choose_papers(source: sqlite3.Connection, *, papers: int, model_id: str) -> list[str]:
    """Pick the papers for a sample.

    A share goes to a cited paper and its citers, so the citation graph has
    edges; the rest are taken in turn across categories. Only embedded,
    ordinary-length papers under an `open` licence qualify, because a sample
    redistributes their text.
    """
    chosen = connected_neighborhood(
        source, model_id=model_id, limit=min(papers // NEIGHBORHOOD_SHARE, papers)
    )
    rows = source.execute(
        ELIGIBLE_SQL + " ORDER BY p.arxiv_id",
        {"model_id": model_id, "min_passages": MIN_PASSAGES, "max_passages": MAX_PASSAGES},
    ).fetchall()

    taken = set(chosen)
    by_category: dict[str, list[str]] = {}
    for row in rows:
        if row["arxiv_id"] in taken:
            continue
        try:
            primary = (json.loads(row["categories_json"]) or ["unknown"])[0]
        except (TypeError, ValueError):
            primary = "unknown"
        by_category.setdefault(primary, []).append(row["arxiv_id"])

    while len(chosen) < papers and any(by_category.values()):
        for category in sorted(by_category):
            if not by_category[category]:
                continue
            chosen.append(by_category[category].pop(0))
            if len(chosen) == papers:
                break
    return chosen


def _column_copier(connection: sqlite3.Connection):
    """Copy rows between the attached databases by column name, since column order can differ."""
    def copy(table: str, where: str = "", parameters=()) -> None:
        columns = [
            row[1] for row in connection.execute(f"PRAGMA main.table_info({table})")
        ]
        available = {
            row[1] for row in connection.execute(f"PRAGMA source.table_info({table})")
        }
        shared = [name for name in columns if name in available]
        names = ", ".join(f'"{name}"' for name in shared)
        connection.execute(
            f"INSERT OR IGNORE INTO main.{table} ({names})"
            f" SELECT {names} FROM source.{table} {where}",
            tuple(parameters),
        )

    return copy


def export_sample(
    source_path: Path, destination: Path, *, papers: int, model_id: str
) -> SampleReport:
    """Write a self-contained corpus holding a handful of papers."""
    if papers < 1:
        raise ValueError("papers must be at least 1")
    if destination.exists() and destination.samefile(source_path):
        raise ValueError(f"{destination} is the corpus being sampled; write the sample elsewhere")

    reader = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    reader.row_factory = sqlite3.Row
    try:
        chosen = choose_papers(reader, papers=papers, model_id=model_id)
        if not chosen:
            raise ValueError(
                "No embedded papers to sample from; run 'verbatim embed' first"
            )
    finally:
        reader.close()

    # Built beside the destination and moved into place only when complete, so
    # a failed export never touches an existing file.
    partial = destination.with_name(destination.name + ".partial")
    _remove_database(partial)
    try:
        counts = _write_sample(partial, source_path, chosen, model_id)
        partial.replace(destination)
    except BaseException:
        _remove_database(partial)
        raise
    # The old file's log goes only once the file is gone: left behind, it would
    # be replayed onto the new one.
    for suffix in ("-wal", "-shm"):
        Path(f"{destination}{suffix}").unlink(missing_ok=True)
    return SampleReport(
        papers=counts[0], chunks=counts[1], vectors=counts[2],
        references=counts[3],
        bytes_written=destination.stat().st_size,
    )


def _write_sample(
    path: Path, source_path: Path, chosen: list[str], model_id: str
) -> tuple[int, int, int, int]:
    """Copy the chosen papers into a new corpus at `path`; returns its counts."""
    # Opening the destination creates the schema; only rows are copied.
    with CorpusStore(path) as sample:
        sample.connection.execute("ATTACH DATABASE ? AS source", (str(source_path),))
        placeholders = ",".join("?" * len(chosen))
        with sample.connection:
            copy = _column_copier(sample.connection)
            copy("embedding_models")
            for table, column in PAPER_SCOPED_TABLES:
                where = f"WHERE {column} IN ({placeholders})"
                if table == "chunks":
                    where += " AND kind != 'references'"
                copy(table, where, chosen)
            # Matched to the chunks actually copied: a vector whose passage
            # was left out would fail the foreign key it depends on.
            copy(
                "embeddings",
                f"WHERE chunk_id IN (SELECT chunk_id FROM source.chunks"
                f" WHERE arxiv_id IN ({placeholders}) AND kind != 'references')"
                f" AND model_id = ?",
                (*chosen, model_id),
            )
            copy(
                "paper_embeddings",
                f"WHERE arxiv_id IN ({placeholders}) AND model_id = ?",
                (*chosen, model_id),
            )
        sample.connection.execute("DETACH DATABASE source")
        # The vector index is derived, so it is rebuilt rather than copied.
        dimensions = sample.connection.execute(
            "SELECT dimensions FROM embedding_models WHERE model_id = ?", (model_id,)
        ).fetchone()[0]
        generation = sample.chunk_generation()
        SqliteVecIndex(sample.connection, dimensions).rebuild(model_id, generation)
        SqlitePaperIndex(sample.connection, dimensions).rebuild(model_id, generation)
        counts = tuple(
            sample.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("papers", "chunks", "embeddings", "paper_references")
        )
        sample.connection.execute("VACUUM")
    return counts


def _remove_database(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
