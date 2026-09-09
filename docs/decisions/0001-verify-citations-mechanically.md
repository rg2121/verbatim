# 1. Verify citations mechanically, never by similarity

**Status:** accepted, and load-bearing for everything else.

## Context

Asked questions the corpus could not answer, early versions produced confident
answers whose citations were all individually genuine: "what is a tensor" cited
a tensor-slicing paper, "what is the nuclear force" once cited a German
pretraining-data paper whose page 27 contained sample text about nuclear
forces. Fluent, sourced, and wrong.

## Decision

A citation is accepted only if its quoted words appear in the passage it names.
A deterministic substring check, forgiving collapsed whitespace and wrapping
punctuation and nothing else. Never a similarity score, never a second model
call. A citation that fails is rejected and its inline marker becomes
`[unverified]`.

## Consequences

Small local models paraphrase, so rejected citations are common — that is the
check working. The guarantee is narrow and worth stating precisely: it proves
the quoted words exist in the cited passage. It does not prove the passage
answers the question, and no amount of strengthening this check would. That
second problem is [0004](0004-report-evidence-rather-than-judging-it.md).
