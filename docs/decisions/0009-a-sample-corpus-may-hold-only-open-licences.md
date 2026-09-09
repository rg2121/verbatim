# 9. A shipped corpus may hold only open-licensed papers

**Status:** accepted, after shipping one that did not.

## Context

A sample corpus makes a clone usable immediately. It also redistributes the
papers' text inside this repository.

## Decision

Only `open` licence classes — CC-BY, CC-BY-SA, CC0 — may appear in a shipped
sample. arXiv's default licence grants distribution rights to *arXiv*, not to
us; `no-derivatives` forbids the chunking; `unknown` means unknown.
Eligibility is one SQL fragment shared by every selection path, so no path can
bypass it.

## Consequences

The first sample violated this and held 20 non-redistributable papers; the git
history was rewritten to remove it. The same rule governs training data, where
it is written down in `TRAINING-POLICY.md` and gates that work entirely. The
licence itself is only obtainable from arXiv's OAI-PMH records — the query API
never returns it, which a naive filter silently mistakes for "no licence".
