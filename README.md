# Verbatim

A local-first, citation-focused literature-review assistant for scientific papers.

## First milestone

Fetch a small set of paper records from arXiv, retain their provenance, and save
them locally. Full texts are deliberately out of scope for this first step.

## Quick start

```bash
uv run verbatim fetch --category physics.gen-ph --max-results 5
```

This writes JSON to `data/papers.json` (which is intentionally not committed).

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

