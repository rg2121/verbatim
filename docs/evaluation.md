# Evaluation

Two answer keys, and they disagree. That disagreement is the most useful thing
in this document.

## The keys

| | cases | authored | what it is for |
|---|---|---|---|
| `evals/retrieval.jsonl` | 87 | alongside the system | development; drove every design decision |
| `evals/heldout.jsonl` | 93 | blind, frozen before scoring | an unbiased estimate |
| `evals/gate.jsonl` | 12 | against the sample corpus | a CI tripwire, not a measurement |

The development key's weakness is structural: a question that retrieved badly
could be reworded until it did not, and no care rules that out from the
inside. The held-out key was built to remove that. Papers were chosen by a
fixed rule before any question existed — current papers with 25–90 body
passages, excluding every paper already in the development key and every
subject probed by hand — and each question was written from that paper's title
and abstract only, with no retrieval run against it. The file was committed
before it was scored, so the order is on the record rather than asserted.

## What they score

| mode | held-out mrr | development mrr |
|---|---|---|
| keyword-any | 0.697 | 0.705 |
| hybrid+titles | 0.737 | 0.822 |
| vector | 0.756 | 0.818 |
| hybrid | 0.758 | 0.866 |
| hybrid+graph *(default)* | 0.758 | 0.869 |

**Development figures are inflated by about an eighth.** Same corpus, same
kind of question; the difference is contamination, not retrieval.

## None of these modes is distinguishable from another

Reciprocal rank was recorded per case and the modes compared with a
10,000-round paired bootstrap over the 88 answerable held-out cases
(`evals/paired_bootstrap.py`):

| comparison | difference | 95% interval |
|---|---|---|
| graph − hybrid | +0.000 | [+0.000, +0.000] |
| graph − compare | +0.006 | [−0.014, +0.028] |
| graph − titles | +0.021 | [−0.034, +0.075] |

Every interval contains zero. At this sample size a few hundredths of mrr is
noise — and the differences this project used to rank modes are smaller than
that. `hybrid+graph` was made the default over plain `hybrid` on **0.003**,
which is unmeasurable. See
[postmortems/0002](postmortems/0002-measuring-against-a-contaminated-key.md).

The consequence is not that the gates are wrong. It is that a gate aimed at a
question shape has to be judged on that shape's cases — where "what is the
Adam optimizer" goes from nothing in the top ten to rank 3 — and not on a
corpus-wide mean that cannot see three cases move.

## What the misses are

`evals/failure_taxonomy.py` labels every held-out case from three mechanical
signals: the expected paper's rank, how much of the question's vocabulary
appears in that paper at all, and how similar the paper that took first place
is to it by abstract vector.

| | count | share |
|---|---|---|
| ranked first | 60 | 68% |
| ranked 2–5 | 15 | 17% |
| ranked 6–10 | 6 | 7% |
| missed, a near neighbour took first place | 5 | 6% |
| missed, unexplained | 2 | 2% |
| **missed on vocabulary** | **0** | **0%** |

The empty row was the expected answer. Vocabulary coverage across every
failure runs 0.67–1.00, so no miss is a lexical gap; the failures are
competition, with the winner topically adjacent at 0.62–0.95 abstract
similarity.

Which limits the metric as much as the retrieval. Each held-out question names
one acceptable paper because each was written from one abstract, so a correct
answer drawn from a neighbouring paper scores as a miss. **0.758 is a floor.**
Closing the gap needs a judgement about whether an answer is supported — an
answer-level evaluation this project does not yet have.

## Compound questions

11 development cases require *every* listed paper rather than any of them,
scored by coverage and completion instead of first hit, because ten excellent
passages about PPO answer half of "how does GRPO differ from PPO".

| mode | cov@5 | all@5 | cov@10 | all@10 |
|---|---|---|---|---|
| hybrid | 0.41 | 0.09 | 0.59 | 0.36 |
| hybrid+plan (decomposition) | 0.36 | 0.09 | 0.59 | 0.36 |
| hybrid+compare | 0.59 | 0.36 | 0.73 | 0.55 |

Decomposition lost on the questions it was built for. The comparison gate wins
here and costs an unmeasurable −0.006 on single-paper comparison questions, so
both directions have numbers and the default is still off pending a decision.

## The CI gate

Twelve questions over the sample corpus committed to this repository — the
only corpus a clean checkout has — with a floor of mrr 0.90, run on every
push. It is coarse by construction and exists to catch retrieval breaking, not
slipping. A measurement nothing enforces is a claim rather than a guarantee.

## Reproducing

```bash
uv run verbatim eval --cases evals/heldout.jsonl   # the honest number
uv run python evals/paired_bootstrap.py                # the intervals
uv run python evals/failure_taxonomy.py                # the failure labels
uv run verbatim eval --cases evals/gate.jsonl --state sample/corpus.db --min-mrr 0.90
```
