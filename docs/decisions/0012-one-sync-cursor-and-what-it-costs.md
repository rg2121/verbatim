# 12. One sync cursor, and what it costs

**Status:** accepted for now; the second cursor is specified here, not built.

## Context

`sync` stores one cursor per category: `newest_updated`, the most recent
last-updated timestamp already ingested. Paging stops once results are older
than it, which is what makes a repeat sync cost one request instead of walking
the category again.

That cursor answers "what is new?" It cannot answer "how far back have I
been?", and the corpus shows the consequence:

| category | watermark | papers seen |
|---|---|---|
| cs.LG | 2026-08-06 | 2,050 |
| cs.CL | 2026-08-06 | 1,250 |
| gr-qc | 2026-08-07 | 500 |

Every category sits at the limit its first sync was given. A later
`sync --category cs.LG --limit 5000` fetches nothing beyond the newest few
days, because everything older is already behind the mark. The corpus does
hold papers back to 1997 — but those arrived through `grow`, following
citations, not through `sync`. The recency slice this project has documented
elsewhere is partly a property of this cursor.

`--full` is the escape hatch and it works: it ignores the watermark and
re-walks from the newest paper, refreshing rather than duplicating. Two costs
come with it.

The small one is bandwidth: deepening cs.LG from 2,050 to 5,000 re-requests
the 2,050 already held, about 20 wasted pages out of 50.

The real one is that **deep backfill is not resumable**, and it is the
operation most likely to need resuming. arXiv's API is documented here as
variable — the same query has taken 2s and 61s and returns 503 under load — and
a `--full` run that dies at page 30 recorded nothing about how deep it got, so
the next one starts from the newest paper again. Papers arrive newest-first, so
a failed deep walk has re-fetched only what was already held and has reached
none of what it was for.

## Decision

Keep the single cursor. Do not build the second one speculatively.

The design, when it is built, is a date cursor and not a page offset. arXiv's
`start` offset shifts as new papers are published, so a stored offset points
somewhere different tomorrow; a stored `oldest_updated` timestamp is stable.
It needs three things: an `oldest_updated` column on `sync_state`, a date range
threaded into the feed query (`cat:X AND lastUpdatedDate:[* TO <cursor>]`), and
a `sync --backfill` that pages older-first and advances the new cursor only.
The two cursors then never touch: forward runs move `newest_updated`, backfill
runs move `oldest_updated`, and a failure in either leaves the other alone —
the same rule already applied to failed forward syncs in postmortem 0001.

The trigger for building it is a decision to deepen a category, not a review
comment. Built before then it is a schema change, a second paging direction and
its tests, all defending a cost nobody is currently paying: every category here
was filled once, in August, and the pipeline has read 0 pending ever since.

## Consequences

`--full` stays the documented way to go deeper, and the README now says plainly
that a deep re-walk is not resumable rather than only that it is expensive. The
limit is asserted by test, so it is a known shape and not a surprise.

The risk accepted is that someone deepens a large category over a bad
connection and loses the run repeatedly. The mitigation available today is to
deepen in stages with increasing `--limit`, each stage committing what it
fetched — worse than a cursor, but not a wall.
