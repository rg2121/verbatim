# 6. Gate extra retrieval legs by question shape

**Status:** accepted; the comparison gate is built but still off by default.

## Context

Definitional questions fail in a specific way: 3,152 papers mention Adam and
the paper defining it holds 53 passages, so every result is somebody's
hyperparameter section. Matching the question against paper *titles* finds it —
and, measured always-on, made retrieval worse overall, because "why does PPO
clip its objective" then matches CLIP.

## Decision

Run the extra leg only where the failure lives. The citation-graph anchor fires
only when the question is definition-shaped **and** the title-matched paper is
cited by at least two held papers, so the corpus itself testifies that this is
the paper a subject is named after. The comparison leg fires only on comparison
vocabulary. Both stand down entirely when the caller has scoped or filtered the
search, since they name papers of their own.

## Consequences

The gates work as scoped: on a held-out set containing no definition-shaped
questions the graph leg contributes exactly 0.000, which is the behaviour
claimed. But the aggregate case for either gate is weaker than it looked — the
differences are inside the noise — so a gate must be justified on the cases it
targets, where the effect is visible, rather than on a corpus-wide mean.
