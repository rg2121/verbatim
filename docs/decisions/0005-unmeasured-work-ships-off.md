# 5. Anything unmeasured ships off by default, with its numbers written down

**Status:** accepted.

## Context

Every plausible retrieval improvement is plausible. Reranking, query rewriting,
HyDE, contextual embeddings and title matching all sound like gains.

## Decision

A technique ships enabled only with numbers behind it. Everything else stays
behind a flag with its measurement recorded, including the ones that lost.

## Consequences

Cross-encoder reranking made retrieval worse on all three local models tried
(mrr 0.940 → 0.752 / 0.701 / 0.741). Query rewriting, HyDE and always-on title
matching all raised recall@10 and lowered recall@1 — widening a pool that is
already right most of the time can only displace the right answer. Decomposition
lost on the compound questions it was built for. None of this was deleted; it is
why the README's rejected-techniques section is the longest one, and it is the
most useful part of this repository.

A later measurement qualified this rule rather than overturning it: differences
of a few hundredths of mrr at these sample sizes are noise, so "measured
better" needs an interval, not a decimal. See
[postmortems/0002](../postmortems/0002-measuring-against-a-contaminated-key.md).
