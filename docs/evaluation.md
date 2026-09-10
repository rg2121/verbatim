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

## Growing the key from 88 cases to 488

At 88 cases nothing was distinguishable from anything: every interval
contained zero, and the differences this project had used to rank modes were
smaller than the noise. `verbatim author-cases` added **400 more**,
written by a model from titles and abstracts only, frozen and committed before
being scored. The authoring rule is structural — `authoring.py` imports
nothing that can search, and a test fails if that changes.

**A quarter of what a model writes from an abstract borrows its wording.** Of
536 candidates, 134 were rejected for sharing a five-or-more-word run with the
abstract they were written from. Abstracts are passages in this corpus, so
those questions would have been retrieved for that reason alone. The threshold
came from the existing set rather than from taste: the 88 hand-authored cases
share a median run of 2 words and at most 6, with 4 of 88 at five or more.

### The batches are not equally hard, and that is the first result

| batch | cases | hybrid mrr |
|---|---|---|
| hand-authored, first | 36 | 0.675 |
| hand-authored, second | 52 | 0.816 |
| generated | 400 | 0.918 |

The generated cases are much easier — but so is the second hand-authored batch
compared with the first, by 0.141. **How a batch was written moves the score
more than any retrieval mode does.** The practical rule: an absolute mrr is
meaningless without naming the batch, and figures from different batches must
never be compared.

### What survives the batching: the comparisons

Mode *rankings* replicate across batches even though the absolute scores do
not. Default against the always-on title leg:

| batch | cases | difference | 95% interval | won–lost | sign test |
|---|---|---|---|---|---|
| hand, first | 36 | −0.052 | [−0.161, +0.056] | 6–7 | p = 1.00 |
| hand, second | 52 | +0.071 | [+0.029, +0.122] | 13–0 | p = 0.0002 |
| generated | 400 | +0.019 | [−0.002, +0.040] | 38–19 | p = 0.016 |
| **all 488** | 488 | **+0.019** | [−0.000, +0.038] | **57–26** | **p = 0.0009** |

**This is the first mode comparison in the project that says anything.** The
default beats the title leg on direction with high confidence — it wins twice
as often as it loses, replicated in the two batches that have any signal at
all, and the small first batch that disagrees shows none (6–7).

It is reported as *direction*, not as a magnitude. The mean difference is
0.019 with an interval that still touches zero, and the driver's two-test rule
— bootstrap share at or below zero **and** an exact sign test — therefore still
prints NOT distinguishable. Both halves of that are true and both are worth
saying: the title leg is reliably worse, and by an amount too small for the
mean to pin down. Keeping the stricter verdict matters more than claiming this
one, because the rule exists to stop a percentile bound being read as a result.

### The graph gate fires on none of the 488

`hybrid+graph` and plain `hybrid` return *identical* rankings on every case in
the key. The default is not beating the alternative here; it is the same
computation. The key contains no definitional questions, because deciding that
a subject is definitional requires inspecting the corpus, which authoring blind
forbids. So the shipped default remains justified by the question shape it
targets — "what is the Adam optimizer": nothing in the top ten, then rank 3 —
and by nothing in this table. See
[postmortems/0002](postmortems/0002-measuring-against-a-contaminated-key.md).

The comparison gate is in the same position for the opposite reason: it fires
on 6 of 488 cases, wins 5 of them, and 5–1 is p = 0.22.

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

## Scoring the answers, not the arXiv ids

Everything above asks whether retrieval surfaced an expected paper. That
understates the system by construction, and the failure taxonomy says by how
much: each held-out question names one acceptable paper, so a correct answer
drawn from an equally good neighbour is counted a miss.
`verbatim eval-answers` scores what the system actually produced.

Over 20 held-out questions on the full corpus:

| | |
|---|---|
| declined outright | 0% |
| at least one verified citation | 80–90% |
| every claim-carrying sentence has a marker | 70–75% |
| cited the paper the question was written from | 80–85% |

Two ranges rather than two numbers, because generation is not deterministic:
the same twenty questions on the same corpus gave 90% and 80% on consecutive
runs. Answer-level metrics need repeats before a difference means anything.

The third row is the one that settles a question left open above. Retrieval
puts the expected paper first 68% of the time and somewhere in ten results 92%
of the time; the *answer* cites it 80–85% of the time, because it can draw on
any retrieved passage rather than only the top one. **0.758 was a floor, as
suspected, and this is roughly where the ceiling sits.**

## Whether a quote supports the claim it is attached to

The remaining question cannot be counted: does a cited quote actually support
the sentence it is attached to? Asking a model that is the move this project
measured failing seven times, so the judge is never believed on its word.
Every real (claim, quote) pair is scored beside two mismatched controls - an
easy one borrowing a quote from a different question's answer, and a hard one
borrowing from the *same* answer, where the quote is on the same subject, from
the same paper, and still not what this claim rests on.

| | supported |
|---|---|
| real pairs | 89% |
| hard control — same answer, wrong claim | **62%** |
| easy control — another question entirely | 0% |

Against the easy control the judge looks excellent, separating by 89 points.
That number is worthless: telling cosmology from recommender systems is a
topic classifier's job. Against the hard control it separates by **26 points**,
which is real but modest, and it calls **62% of known mismatches supported**.

Two conclusions, and the second matters more.

**The old finding is refined rather than overturned.** Asked whether retrieved
passages are relevant to a question, models were useless - 0/6, and the
largest model was worst. Asked whether a specific quote states a specific
claim, the same kind of model separates real pairs from near-misses by 26
points. Sufficiency and entailment are different questions, and only the
second is one a model can partly answer.

**And it must not gate anything.** A judge that accepts 62% of pairs known to
be mismatched cannot be allowed to accept or reject a citation. That is
precisely why verification stays a deterministic substring check and why the
critic never judges - this measurement is the evidence for both, rather than a
reason to relax either.

One caveat against my own control: two quotes from the same paper on the same
subject may *both* support a claim, so some of that 62% are probably correct
rather than lenient. The hard control is a pessimistic adversary and the true
separation is somewhere between 26 and 89 points. Narrowing that needs
hand-labelled pairs, which is the honest next step and is not yet done.

## Is the workflow worth its cost? And the critic?

The orchestrator had been justified by argument and by one report on one
topic. `evals/three_arm.py` puts the same frozen held-out questions through
three arms - one retrieval and one answer, the ladder, and the ladder plus the
critic - scored on the answers rather than on which paper retrieval named.

The first version of this ran **20 questions**, taken as the first twenty in
file order, and reported that the workflow recovered one of three baseline
failures. Three is not a sample and the first twenty of a file authored in
batches is not a draw from the set, so it was re-run over **all 88 answerable
held-out questions**, 2 runs per arm. The critic is not in the re-run: it was
already measured worse at four times the cost, and re-deciding a decided
question would have trebled the wall clock. Its 20-question numbers are below.

| out of 88, mean of 2 runs | baseline | workflow |
|---|---|---|
| answers with a verified citation | 78.5 | **81.5** |
| every claim carries a marker | 70.5 | **72.5** |
| cited the expected paper | 72.0 | **74.5** |
| citations rejected | 10.5 | **8.5** |

The workflow is ahead on every row, and the paired difference in
verified-citation rate is **+0.034 with a 95% bootstrap CI of [+0.006,
+0.074]** - an interval that excludes zero, which nothing in this project's
comparisons had done before.

**It still does not establish anything, and the reason is worth recording.**
The arms give the same answer on 81 of the 88 questions. The entire result
rests on the 7 where they differ: the workflow is better on 6 and worse on 1.
A sign test on those 7 gives p = 0.125. And the bootstrap's lower bound is an
artefact of the RNG seed - the per-question difference takes few distinct
values, so the 2.5th percentile lands *on* zero for 7 of 10 seeds and just
above it for the other 3. The first run of this reported "distinguishable"
because seed 0 happened to fall on the right side.

The driver now prints the share of bootstrap resamples at or below zero
(0.024, and stable across seeds), the sign test, and the count of questions
the arms actually differ on - and calls a difference distinguishable only when
both tests agree. This one does not, so it reads **NOT distinguishable**.

What survives is direction, not significance: the workflow is ahead on all
four rows and on 6 of the 7 questions where anything changed. That is the
strongest evidence this project has for the orchestrator, and it is still not
enough. Settling +0.034 needs several hundred questions - the same conclusion
the retrieval comparison reached, from the other end.

Cost is **not measured in this run**. The 502 that killed it at answer 217
meant the resumed process generated only part of each arm, and the driver
withholds a cost it cannot attribute rather than reporting the fraction it
happened to see. The only cost figure remains the 20-question one below: 20
calls against 46, or 2.3×.

For the workflow that leaves the same shape of argument as before. The arms
share their first rung, so it does identical work wherever the baseline
succeeds and can only differ where the baseline fails - and on the 12
questions the baseline missed at least once, the workflow's verified-citation
rate is 0.50 against 0.21. That subset is selected on the baseline's own
failure, so the baseline's rate there is biased low and the comparison cannot
carry a headline; it is the mechanism, not the evidence. Whether 2.3× is worth
it depends on what a failure costs you: in a review, a section with no
verified citation is a hole in the document; for a single question, you can
ask again.

The 20-question run, kept because it is where the critic was measured:

| out of 20 | baseline | workflow | +critic |
|---|---|---|---|
| answers with a verified citation | 17.5 | **18.0** | 17.0 |
| every claim carries a marker | 15.0 | **16.0** | 14.0 |
| citations rejected | 2.5 | **1.5** | 3.0 |
| **cost** | **20 calls, 57k tokens** | **46 calls, 77k** | **83 calls, 87k** |

**For the critic the answer is simpler: it did not pay.** Every measure is
equal or worse than the workflow without it - fewer verified citations, fewer
fully marked answers, twice the rejected citations, and **none of the three
baseline failures recovered** - for another 1.8× on top. The mechanism is
visible in the last two rows: the critic escalates when it finds a subject no
passage covers, each escalation is another answer that may be weaker than the
one it replaced, and more attempts mean more chances to produce a quote that
fails verification.

So the critic stays off, and is now **measured worse rather than unproven**.
It remains useful as *observability* - the uncited-claim count is a real fact
about an answer and needs no model - but as a control loop it costs four times
the baseline and buys nothing measurable. If it earns its place later it will
be by reporting, not by escalating.

The caveat that applies to all of this: 20 questions and 2 runs. Settling
differences this small needs several hundred questions, which is the same
conclusion the retrieval comparison reached.

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
uv run verbatim eval-answers --cases evals/heldout.jsonl --limit 20 --judge
uv run python evals/three_arm.py 20 2                  # baseline vs workflow vs critic
uv run verbatim eval --cases evals/gate.jsonl --state sample/corpus.db --min-mrr 0.90
```
