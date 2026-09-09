# Three silent faults in the unattended ingest

**Date:** 2026-09-09. **Impact:** papers skipped permanently, stale passages
citable after their source changed, and searches answered from the wrong
model's vectors. None raised an error. **Fixed in** `fc16b3b`.

## What happened

An external review named three suspected correctness bugs. All three were
reproduced before being accepted, and all three were real.

**A failed sync advanced its watermark.** Papers are fetched newest-first and
the watermark only moves forward. A run that failed part-way still recorded the
newest paper it had seen, so every page the failure prevented it from reaching
fell permanently behind the mark. Reproduced: a 503 on page two, then a healthy
run that fetched nothing, with two papers unreachable for good.

The comment on that line asserted the opposite of what the code did — *"the
next run resumes instead of walking the category again"* — and an existing test
asserted the same belief, `assert watermark is not None  # so the next run
resumes rather than restarting`. It did not resume. It skipped.

**A re-chunk that produced nothing deleted nothing.** `record_chunks` derived
the papers to clear from the chunks it was handed, so an empty result cleared
no rows — while the stage marked the paper chunked. A PDF that stops extracting,
or exceeds the layout timeout the README documents, kept its old passages and
their vectors, searchable and citable, under a checksum the file no longer has.

**The vector index was not model-scoped.** `embeddings` is keyed
`(chunk_id, model_id)` so two models coexist, but the index held one set with
no record of which, and freshness was decided by counting rows. Two models of
the same width — `bge-small` and `bge-small+context` — produce equal counts, so
no sync ran and every search answered from the previous model's vectors.
Reproduced by reading the stored vector back: model B was asked, model A's
vector was there, nothing raised.

## Why they survived this long

Each replaced the question that mattered with a cheaper proxy that usually
agrees with it:

| the question | the proxy |
|---|---|
| did this run *finish*? | did it *advance*? |
| what did this pass *attempt*? | what did it *produce*? |
| whose vectors are these? | how many are there? |

Proxies of that shape are invisible in the common case and wrong only in the
uncommon one, which is exactly where an unattended nightly job lives. Tests
passed because the tests encoded the same assumption as the code — in the sync
case, literally the same sentence.

## What changed

A failed run leaves the watermark alone and the next run re-walks. Callers pass
the papers they attempted so a zero-yield pass still clears them. The index
records its model and rebuilds on a mismatch, or when it has never said. Each
has a regression test built from the reproduction, and `ingest.sh` — which
piped every stage into `tail` and therefore reported `tail`'s exit status —
now checks the real one.

## What it changed about how this is built

The run ledger exists because of this. Three faults ran silently for weeks in a
system whose executions left no record beyond terminal scrollback; `stats` said
what the corpus held and nothing said how it got there. Every command now
writes a row, failures included.
