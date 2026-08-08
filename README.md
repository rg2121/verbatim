# Verbatim

A local-first, citation-focused literature-review assistant for scientific papers.

## Architecture

`data/corpus.db` is the cumulative source of truth. Everything else — the JSON
snapshot, the PDFs, the chunk export — is derived and can be rebuilt from it.

```
arXiv API ──sync──▶ corpus.db  papers           every version seen, newest current
                              downloads        local PDFs, checksummed
                              chunks           page-aware passages + char offsets
                              chunks_fts       BM25 keyword index
                              embeddings       one vector per chunk per model
                              vec_chunks       nearest-neighbour index
                    │
           download ──▶ data/pdfs/*.pdf
              chunk ──▶ data/chunks.jsonl   (export, not the store)
              embed ──▶ vectors in corpus.db
             search ──▶ ranked passages with citations
                ask ──▶ an answer, every claim cited and verified
```

## Daily corpus workflow

```bash
uv run verbatim sync --category physics.gen-ph --max-results 5
uv run verbatim download --limit 3
uv run verbatim chunk
uv run verbatim embed
uv run verbatim search "how does a changing field induce current in a loop"
```

`sync` pages through one or more categories, newest-updated first, and upserts
into the catalogue. It tracks full arXiv versions and marks superseded versions
without deleting them.

```bash
uv run verbatim sync --category cs.CL cs.IR --limit 2000
```

A stored watermark per category is what makes this resumable: paging stops once
results are older than the last run, so re-syncing costs one request rather than
walking the category again. `--full` ignores it. arXiv's API is variable — the
same query has taken 2s and 61s, and returns 503 under load — so requests are
retried with backoff, and a page that still fails stops the run cleanly with
everything already fetched committed and the watermark recorded.

`download` selects the newest current versions that have not yet been
downloaded. Cached PDFs are reused only after their checksum matches the
previous manifest, and downloads and generated artifacts are written
atomically. Add `--require-license` to restrict downloads to records with an
explicit licence in arXiv metadata.

`chunk` re-verifies each PDF checksum, then extracts page-aware passages for
only those files whose chunker version, chunker parameters, or bytes have
changed since they were last chunked. It also refreshes `data/chunks.jsonl`,
which is an export rather than the store.

`embed` vectorises chunks that have no vector for the chosen model, in batches,
and is a no-op on re-run. It uses `BAAI/bge-small-en-v1.5` through ONNX, so
there is no torch dependency and the weights are about 130MB.

`search` prints each passage with the paper version, page, character span, and
source URL needed to cite it. Use `--arxiv-id` to search within one paper and
`--full` for whole passages. Three modes:

| Mode | Finds | Misses |
|---|---|---|
| `hybrid` (default) | both of the below, fused by rank | — |
| `keyword` | exact terms, identifiers, notation | paraphrases |
| `vector` | meaning, without shared words | rare exact tokens |

Keyword mode requires every term by default; `--match any` ranks by how many
matched. The index matches whole tokens and does not stem, so `field` does not
match `fields` — use `field*`. Hybrid uses `any` internally, because requiring
every word of a natural-language question matches nothing at all.

`stats` shows what each stage has done and what it has left, which is how a
long ingest stays observable:

```
  stage      done                           remaining              next
  papers     59                             59 current
  licences   19 known                       40 to look up          verbatim licenses
  pdfs       35 (0.1 GB)                    24 to fetch            verbatim download
  chunks     3,184 (2,326 body / 858 refs)  0 pdfs to chunk        verbatim chunk
  vectors    3,184                          0 to embed             verbatim embed
```

Every stage is incremental and safe to re-run: nothing is redone unless its
input changed. Papers are upserted by arXiv ID; a PDF is fetched only if it has
no download record; a PDF is re-chunked only if its bytes, the chunker version,
or the chunk size changed; a chunk is embedded only if it has no vector *for
that model*; and the vector index adds and removes only what differs.
`verbatim index --rebuild` forces a full reload if you ever need one.

`ask` retrieves passages, answers strictly from them, and cites every claim:

```bash
uv run verbatim ask "can magnetic flux move through a region of zero field?"
```

```
Flux can move through zero field [1].

References
[1] Walstad, "Moving Magnetic Flux and Electromagnetic Induction",
    arXiv:2608.05440v1, p. 1. https://arxiv.org/abs/2608.05440v1
```

Every citation is checked in code before it is printed: the quoted words must
appear in the passage the claim names, or the citation is reported as rejected
rather than rendered. Citations always pin the paper *version*. At most three
passages come from any one paper by default (`--max-per-paper`), so a
corpus-wide question is not answered from a single source.

`ask` uses a local model through **Ollama** by default, so the whole pipeline
runs offline once the corpus is built. It picks `$OLLAMA_MODEL`, or the first
pulled model; override with `--llm`. Use `--backend hosted` to answer with the
hosted API instead, which needs `HOSTED_API_KEY`.

Small local models paraphrase rather than quote, so expect rejected citations —
that is the check working, not a bug. A claim whose quote could not be verified
is marked `[unverified]` in the prose rather than silently presented as sourced.
Ollama's `format` parameter and tool calling were both unreliable on realistic
multi-passage prompts, so the backend elicits a flat `ANSWER:` / `CITE` line
format and parses that.

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
third that needs an explicit written policy first.

**The query API does not return licences at all** — measured across 59 papers,
every one came back without one. Licence lives in the OAI-PMH record instead:

```bash
uv run verbatim licenses --limit 50
```

Licences are grouped by what they actually permit, because presence is not
permission:

| Class | Meaning |
|---|---|
| `open` | CC-BY / CC-BY-SA / CC0 — redistribution and training permitted, usually with attribution |
| `arxiv-nonexclusive` | arXiv's default: **arXiv** may distribute it, you may not |
| `unknown` | not looked up yet, or the paper predates the field |

Licence URLs are carried through to every chunk, and `--require-license`
restricts downloads to papers that report one — but note it accepts
`arxiv-nonexclusive` too, so it is not yet a reuse gate. Nothing enforces a
training policy.

PyMuPDF is AGPL-3.0. That is fine for local personal use and is a decision to
revisit before building any hosted product on top of it.

## Configuration

All tunables live in `.env`; copy the template and edit:

```bash
cp .env.example .env
```

`.env.example` documents every setting — storage paths, chunk size and overlap,
embedding model, retrieval limits, and which backend answers questions. `.env`
itself is gitignored because it may hold `HOSTED_API_KEY`. An exported
environment variable always overrides the file, so a one-off run needs no edit:

```bash
OLLAMA_MODEL=llama3.1 uv run verbatim ask "..."
```

## Development

```bash
uv run pytest
```

The ingestion client uses arXiv's public metadata API. Before growing the corpus,
review arXiv's API and bulk-access guidance.
