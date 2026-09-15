"""Exporting a small, self-contained corpus that works out of the box.

Building the real corpus means 26 GB of PDFs and most of a day of embedding.
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
    # The bibliography *records* ride along - they are what the graph command
    # walks - even though bibliography *chunks* are left out below: retrieval
    # excludes those, so they would be weight the sample carries and never
    # uses. The records are a few kilobytes and make `graph` work on a clone.
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


# Papers a sample may hold: current, redistributable, embedded, and of
# ordinary length. Shared by the round-robin pick and the neighborhood pick,
# so no path into a sample can bypass the licence gate.
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
    """An eligible paper and the eligible papers that cite it.

    Chosen so the sample's citation graph demonstrates something: forty papers
    picked independently are strangers, and a `graph` command over strangers
    prints zeros. The hub is the eligible paper most cited by other eligible
    papers, and its citers fill the remaining slots.
    """
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
    """Pick papers that actually demonstrate the system.

    Only papers whose passages are embedded, since a sample where search
    returns nothing is worse than no sample. A share of the slots goes to a
    cited paper and its citers, so the citation graph has edges; the rest are
    spread across categories by taking them in turn, so the sample is not
    fifty papers about one subject.

    Ordinary-sized papers, not the biggest: ranking by passage count selects
    the longest surveys in the corpus and triples the size of the sample for
    no extra demonstration.

    Only papers under a licence that permits redistribution. A sample ships
    the papers' own words inside this repository, which is redistribution, and
    arXiv's default licence grants that right to arXiv rather than to us. The
    other classes are excluded for the same reason: `no-derivatives` forbids
    the chunking, and `unknown` means unknown rather than permitted.
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
            sample.connection.execute("SELECT COUNT(*) FROM paper_references").fetchone()[0],
        )
        sample.connection.execute("VACUUM")

    return SampleReport(
        papers=counts[0], chunks=counts[1], vectors=counts[2],
        references=counts[3],
        bytes_written=destination.stat().st_size,
    )
