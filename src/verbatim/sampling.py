"""Exporting a small, self-contained corpus that works out of the box.

Building the real corpus means 28 GB of PDFs and most of a day of embedding.
Nobody evaluates a tool that way. A sample holds a few dozen papers with their
passages and vectors already in place, so `search` and `ask` work the moment
the repository is cloned, offline and with nothing to fetch.

It carries the database only, not the PDFs or renderings those passages were
read from. Everything the retrieval and citation paths need is in the database
- the text, the offsets, the checksums, the licences - and re-verifying a
passage against its original bytes is a thing the full corpus does, not a
demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

from verbatim.corpus import CorpusStore

# Tables copied wholesale for the selected papers. Order matters: a row may
# only be inserted once the row it references exists.
# Deliberately no downloads or html_sources rows. Those record files held on
# disk, and the sample ships none: carrying them would make `stats` report 40
# PDFs that are not there, which is exactly the kind of claim this project is
# built to not make. Every passage still carries its own source URL, licence
# and checksum, so citations remain complete.
PAPER_SCOPED_TABLES = (
    ("papers", "arxiv_id"),
    ("chunks", "arxiv_id"),
)
# A paper long enough to demonstrate retrieval, short enough that forty of
# them stay small. Bibliographies are left out: retrieval excludes them, so
# they would be weight the sample carries and never uses.
MIN_PASSAGES = 40
MAX_PASSAGES = 120


@dataclass(frozen=True)
class SampleReport:
    """What a sample ended up containing."""

    papers: int
    chunks: int
    vectors: int
    bytes_written: int

    def render(self) -> str:
        megabytes = self.bytes_written / 1_000_000
        return (
            f"{self.papers} papers, {self.chunks:,} passages, "
            f"{self.vectors:,} vectors, {megabytes:.1f} MB"
        )


def choose_papers(source: sqlite3.Connection, *, papers: int, model_id: str) -> list[str]:
    """Pick papers that actually demonstrate the system.

    Only papers whose passages are embedded, since a sample where search
    returns nothing is worse than no sample. Spread across categories by
    taking them in turn, so the sample is not fifty papers about one subject.

    Ordinary-sized papers, not the biggest: ranking by passage count selects
    the longest surveys in the corpus and triples the size of the sample for
    no extra demonstration.
    """
    rows = source.execute(
        """
        SELECT p.arxiv_id, p.categories_json, COUNT(e.chunk_id) AS vectors
        FROM papers AS p
        JOIN chunks AS c ON c.arxiv_id = p.arxiv_id
        JOIN embeddings AS e ON e.chunk_id = c.chunk_id AND e.model_id = ?
        WHERE p.is_latest = 1 AND c.kind = 'body'
        GROUP BY p.arxiv_id
        HAVING vectors BETWEEN ? AND ?
        ORDER BY p.arxiv_id
        """,
        (model_id, MIN_PASSAGES, MAX_PASSAGES),
    ).fetchall()

    by_category: dict[str, list[str]] = {}
    for row in rows:
        try:
            primary = (json.loads(row["categories_json"]) or ["unknown"])[0]
        except (TypeError, ValueError):
            primary = "unknown"
        by_category.setdefault(primary, []).append(row["arxiv_id"])

    chosen: list[str] = []
    while len(chosen) < papers and any(by_category.values()):
        for category in sorted(by_category):
            if not by_category[category]:
                continue
            chosen.append(by_category[category].pop(0))
            if len(chosen) == papers:
                break
    return chosen


def _column_copier(connection: sqlite3.Connection):
    """Copy rows between two databases by column name, never by position.

    `SELECT *` moves columns in whatever order each table happens to have.
    This corpus has had its chunks table rebuilt by a migration, so the source
    and a freshly created destination do not agree on that order, and copying
    positionally would put each value in the wrong column while reporting
    success.
    """
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
    destination.unlink(missing_ok=True)

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

    # Opening the destination creates the schema, so the copy only has to move
    # rows and never has to restate what the tables look like.
    with CorpusStore(destination) as sample:
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
        # The vector index is derived, so it is built here rather than copied.
        from verbatim.vectors import SqlitePaperIndex, SqliteVecIndex

        dimensions = sample.connection.execute(
            "SELECT dimensions FROM embedding_models WHERE model_id = ?", (model_id,)
        ).fetchone()[0]
        SqliteVecIndex(sample.connection, dimensions).rebuild(model_id)
        SqlitePaperIndex(sample.connection, dimensions).rebuild(model_id)
        counts = (
            sample.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
            sample.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            sample.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0],
        )
        sample.connection.execute("VACUUM")

    return SampleReport(
        papers=counts[0], chunks=counts[1], vectors=counts[2],
        bytes_written=destination.stat().st_size,
    )
