# 8. The vector index records which model's vectors it holds

**Status:** accepted, after the absence of this caused a silent fault.

## Context

`embeddings` is keyed `(chunk_id, model_id)` so two models' vectors coexist by
design — the contextual-embedding A/B depends on it. The index held one set
with no record of which, and freshness was decided by comparing counts.

## Decision

The index records its model **and the corpus generation it was built from** in
`vector_index_state`. Identity is checked first, then generation, then size. An
index that has never recorded a model is rebuilt once, because it cannot be
trusted to hold what the caller is asking for.

The generation exists because model identity alone was not enough. A chunk id
is a hash of the paper, its source checksum, the page and the chunker version,
so replacing a paper's source gives every one of its passages a new id while
leaving the count unchanged - and the count comparison read that as up to
date. Measured before the fix: after a source replacement, a vector search
returned **0 passages out of 1 embedded**, and raised nothing.
`record_chunks` therefore bumps a counter on every write, and the index stores
the counter it synced at. `ensure_current` takes it as a required argument
rather than an optional one, because a default would let a caller opt out of
the check by forgetting it, which is how the hole behaved in the first place.

A width change is also a rebuild. A vec0 table declares its dimension in its
schema and `CREATE TABLE IF NOT EXISTS` keeps the old one, so opening a
384-wide index at 768 used to succeed and then fail deep inside an insert with
a dimension mismatch the caller could do nothing about.

## Consequences

Switching embedding models can no longer search the previous model's vectors
silently — the failure this replaced returned confident nonsense and raised
nothing, because `bge-small` and `bge-small+context` have identical widths.
The cost is that only one model is resident at a time, so a genuine side-by-side
A/B now needs `model_id` as a metadata column in the index. That is a known
limit, recorded here rather than discovered again later.
