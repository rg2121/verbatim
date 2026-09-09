# 4. Report what an answer rests on; do not ask a model to judge it

**Status:** accepted after seven failed attempts at the alternative.

## Context

Verified citations do not prove relevance. The obvious fix is to ask whether
the retrieved passages support an answer.

## Decision

Never. Measured against questions the corpus cannot answer: no check at all
caught 2/6, a verdict inside the answering call 1/6, a separate 4B judge 2/6,
a separate 13.8GB judge **0/6**. The larger model was the worst, so this is not
a capability problem — passages are retrieved *because* they match, and every
model reads that as evidence they suffice. Requiring corroboration across
papers caught 4/6 but destroyed 6 of 14 good answers.

What is asked instead is *what these passages are about*, without mentioning
the question, compared to the question by dot product. And every answer now
carries a deterministic account of what it rests on — passages, papers,
subject, and the paper a definitional subject is named after — instead of a
threshold deciding for the reader.

## Consequences

The covered-but-unanswered case is not detected, by anything, and the project
says so rather than implying otherwise. A negative line ("no paper here is
named after X") was built, measured firing on 13 of 19 good answers, and
removed: a warning that frequent is one readers learn to skip.
