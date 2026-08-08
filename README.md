# Verbatim

A local-first, citation-focused literature-review assistant for scientific papers.

## First milestone

Fetch a small set of paper records from arXiv, retain their provenance, and save
them locally. Full texts are deliberately out of scope for this first step.

## Daily corpus workflow

```bash
uv run verbatim sync --category physics.gen-ph --max-results 5
uv run verbatim download --limit 3
uv run verbatim chunk
```

`sync` writes the latest API response to `data/papers.json` for inspection and
upserts it into `data/corpus.db`, the cumulative SQLite catalogue. It tracks
full arXiv versions and marks superseded versions without deleting them.

`download` selects the newest current versions that have not yet been downloaded
and updates the cumulative PDF manifest. `chunk` regenerates a complete,
page-aware JSONL export from the cumulative manifest.

For a one-off metadata-only snapshot, use:

```bash
uv run verbatim fetch --category physics.gen-ph --max-results 5
```

This writes JSON to `data/papers.json` (which is intentionally not committed).
The legacy one-off download form remains available with `download --input`.

The PDF manifest preserves each file's
arXiv ID, source URL, reported licence, and SHA-256 checksum. Add
`--require-license` to restrict downloads to records with an explicit licence
in arXiv metadata. Cached PDFs are reused only after their checksum matches
the previous manifest; downloads and generated artifacts are written atomically.

This writes `data/chunks.jsonl`. Every line includes text plus its arXiv ID,
original source URL, PDF checksum, and page number; it is the input to the
future embedding and vector-index stage. The chunker verifies the PDF checksum
again before emitting any text.

## Principles

- Start with a small, inspectable corpus.
- Preserve source URLs, arXiv IDs, and licence metadata where available.
- Require citations for generated claims.
- Treat model training as a separate, provenance-aware track.

## Development

```bash
uv run pytest
```

The ingestion client uses arXiv's public metadata API. Before growing the corpus,
review arXiv's API and bulk-access guidance.
