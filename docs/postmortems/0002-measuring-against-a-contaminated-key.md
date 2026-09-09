# Measuring against a key written while watching the measurement

**Date:** 2026-09-09. **Impact:** every retrieval figure this project
published was optimistic by about an eighth, and the default retrieval mode was
chosen on a difference that cannot be distinguished from zero. **Recorded in**
`e8e2296`, `2bfa8f0`.

## What happened

Retrieval was measured on an 87-case answer key written alongside the system.
Questions were added as the system was built, which means a question that
retrieved badly could be reworded until it did not. Nothing in the process
prevented that and nothing in the numbers could reveal it.

A second key was built to be uncontaminated: papers chosen by a fixed rule
before any question existed, each question written from that paper's title and
abstract only, no retrieval run against any of them, and the file committed
before it was scored so the order is on the record rather than asserted.

The default scores **mrr 0.869** on the development key and **0.758** on the
held-out one. Same corpus, same kind of question.

Then the ranking appeared to reverse. On the first 36 held-out cases,
`hybrid+titles` — measured *worse* on the development key, and shipped off
because of it — won by 0.052. That was written up as a finding.

It did not survive. On the next 52 cases it lost by 0.071. Pooled over all 88
it loses by 0.021, and a 10,000-round paired bootstrap puts the interval at
[−0.034, +0.075].

## The actual finding

Every mode comparison on the held-out key straddles zero. At this sample size a
few hundredths of mrr is noise — and **the swing between two batches of the
same key is larger than any gap between modes.**

Which indicts the reasoning that had been driving decisions. `hybrid+graph` was
made the default over plain `hybrid` on **0.003 mrr across 63 cases**. That is
not a small effect; it is an unmeasurable one. It was reported as a win, written
into the README, and used to justify a default.

## Why it survived

Two habits, both of which looked like rigour:

- **Reporting differences to three decimals.** A number printed to three places
  invites being read as precise to three places. There was no interval anywhere,
  so nothing prompted the question "is this distinguishable from zero?"
- **Testing changes against the key that had shaped them.** Each new technique
  was scored on the same cases the previous techniques were tuned against, so
  contamination compounded quietly across decisions rather than showing up once.

The first held-out result then nearly repeated the mistake facing the other
way: a reversal on 36 cases was very nearly enough to flip a default. What
stopped it was refusing to act on a small sample, not any measurement — and the
second batch showed that refusal had been right.

## What changed

The README reports both keys and states the discount. Intervals are computed
by a committed script rather than asserted. The gates are still there, but
justified on the cases they target — "what is the Adam optimizer" going from
nothing in the top ten to rank 3 is visible in the case and invisible in a mean
over 63 — rather than on a corpus-wide aggregate. Nothing was changed as a
result of the held-out numbers: `hybrid+graph` remains the default because it
is the incumbent and nothing distinguishes the alternatives, which is a weaker
reason than was given before and a truer one.

## What is still open

Deciding whether `hybrid+titles` should be the default needs several hundred
held-out cases, not another fifty. And since each held-out question names one
acceptable paper, an answer drawn from an equally good neighbouring paper still
scores as a miss — so 0.758 is a floor, and the thing that would actually
measure this system is an answer-level evaluation it does not yet have.
