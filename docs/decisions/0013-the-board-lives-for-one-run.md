# 13. The board lives for one run

**Status:** accepted; the tables are specified here, not built.

## Context

`ResearchState` is in memory. It is shared across the workers of a single
`team` invocation and discarded when that invocation ends. A reviewer noted
that a production-grade system would persist it — `research_runs`,
`research_tasks`, `research_evidence`, `research_decisions` — for resume, for
audit across processes, and for asynchronous workers.

That is correct as a description of production systems. It is not yet a
description of a need here, and this project has a rule about the difference.

## Decision

Keep the board in memory. Persist it when one of three things is true, and not
before:

**A run becomes long enough to want resuming.** A `team` run is two to four
minutes. The ingest, which takes days, is persisted at every stage precisely
because it must survive interruption. Resume earns its complexity against
hours, not minutes — and when the model server died mid-experiment twice in
one day, the fix that mattered was scoring what had been collected, not
restarting from a checkpoint.

**Workers become asynchronous.** Shared mutable state across processes needs a
transactional store; shared state within one process does not. No worker here
runs concurrently with another, and the dispatch loop is deliberately
sequential so that the same board always produces the same next action.

**Somebody needs to audit a run they did not watch.** The run ledger already
records each `team` invocation with its task, evidence, decision and
redirection counts. What it does not keep is the rows themselves. The trigger
is a question the counts cannot answer — "which passage did claim 4 rest on,
last Tuesday" — and until that question is asked, the schema is a guess at its
shape.

## Consequences

The board's provenance is complete within a run and gone after it. A `team`
answer is reproducible only by running it again, which for a two-minute
sequential run against a fixed corpus is close to free — and is not free at
all once the model is non-deterministic, which it is. So the honest statement
is that the *decisions* are reproducible and the *text* is not.

The schema is written down above so that building it later is transcription
rather than design. What is deliberately not written down is a migration: the
records are frozen dataclasses with an `evidence_id`, `task_id` and
`chunk_id` already, so the tables are their fields, and inventing a mapping
before there is a query to serve would be the speculative half of the work.

Related: decision 12, which defers the sync backfill cursor on the same
grounds — the trigger for building it is a decision to deepen a category, not
a review comment.
