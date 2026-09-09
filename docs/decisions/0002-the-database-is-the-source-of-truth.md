# 2. The database is the source of truth

**Status:** accepted.

## Context

The pipeline has eight stages, most expensive, over tens of thousands of files.
Any of them can be interrupted. Artifacts — JSON snapshots, the chunk export,
the vector index — invite being treated as state.

## Decision

`corpus.db` holds the state. Everything else is derived and rebuildable from
it, and nothing downstream writes back into `papers`. Every expensive stage is
incremental and keyed by a fingerprint, so re-running resumes.

## Consequences

An interrupted run costs the batch in flight and nothing more. Derived
artifacts can be deleted without loss, which is why the vector index can be
rebuilt whenever it is not trusted (see
[0008](0008-scope-the-vector-index-to-a-model.md)). The cost is that "what
does the system currently hold" is a query rather than a directory listing —
hence `stats`, and later the run ledger.
