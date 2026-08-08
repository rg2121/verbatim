# Verbatim

A local-first, citation-focused literature-review assistant for scientific papers.

## Architecture

`data/corpus.db` is the cumulative source of truth. Everything else — the JSON
snapshot, the PDFs, the chunk export — is derived and can be rebuilt from it.

```
arXiv API ──sync──▶ corpus.db  papers    every version seen, newest flagged current
                              downloads  local PDFs, checksummed
                              chunks     page-aware passages + char offsets
                              chunks_fts keyword index over chunk text
                    │
           download ──▶ data/pdfs/*.pdf
              chunk ──▶ data/chunks.jsonl   (export, not the store)
             search ──▶ ranked passages with citations
```

## Daily corpus workflow

```bash
uv run verbatim sync --category physics.gen-ph --max-results 5
uv run verbatim download --limit 3
uv run verbatim chunk
uv run verbatim search "magnetic flux"
```

`sync` writes the latest API response to `data/papers.json` for inspection and
upserts it into the catalogue. It tracks full arXiv versions and marks
superseded versions without deleting them.

`download` selects the newest current versions that have not yet been
downloaded. Cached PDFs are reused only after their checksum matches the
previous manifest, and downloads and generated artifacts are written
atomically. Add `--require-license` to restrict downloads to records with an
explicit licence in arXiv metadata.

`chunk` re-verifies each PDF checksum, then extracts page-aware passages for
only those files whose chunker version, chunker parameters, or bytes have
changed since they were last chunked. It also refreshes `data/chunks.jsonl`,
which is an export rather than the store.

`search` ranks passages by BM25 relevance and prints each with the paper
version, page, character span, and source URL needed to cite it. Use
`--arxiv-id` to search within one paper and `--full` for whole passages. The
index matches whole tokens and does not stem, so `field` does not match
`fields`; use `field*` for prefix matching.

For a one-off metadata-only snapshot, use `fetch --category ...`. The legacy
one-off forms `download --input` and `chunk --manifest` remain available and
bypass the catalogue.

## Principles

- Start with a small, inspectable corpus.
- Preserve source URLs, arXiv IDs, and licence metadata where available.
- Cite the paper *version* that was read; v1 and v2 can disagree.
- Require citations for generated claims.
- Treat model training as a separate, provenance-aware track.

## Licence caveats

arXiv metadata and arXiv full texts are licensed differently. The default
submission licence lets arXiv distribute a paper; it does not grant
redistribution rights to third parties. Local reading and indexing is one
thing, redistributing PDFs or chunk text is another, and training on them is a
third that needs an explicit written policy first. Licence URLs are recorded
per paper and carried through to each chunk, but nothing yet enforces a policy.

PyMuPDF is AGPL-3.0. That is fine for local personal use and is a decision to
revisit before building any hosted product on top of it.

## Development

```bash
uv run pytest
```

The ingestion client uses arXiv's public metadata API. Before growing the corpus,
review arXiv's API and bulk-access guidance.
