# 8. The vector index records which model's vectors it holds

**Status:** accepted, after the absence of this caused a silent fault.

## Context

`embeddings` is keyed `(chunk_id, model_id)` so two models' vectors coexist by
design — the contextual-embedding A/B depends on it. The index held one set
with no record of which, and freshness was decided by comparing counts.

## Decision

The index records its model in `vector_index_state`. Identity is checked before
size, and a mismatch rebuilds. An index that has never recorded a model is
rebuilt once, because it cannot be trusted to hold what the caller is asking
for.

## Consequences

Switching embedding models can no longer search the previous model's vectors
silently — the failure this replaced returned confident nonsense and raised
nothing, because `bge-small` and `bge-small+context` have identical widths.
The cost is that only one model is resident at a time, so a genuine side-by-side
A/B now needs `model_id` as a metadata column in the index. That is a known
limit, recorded here rather than discovered again later.
