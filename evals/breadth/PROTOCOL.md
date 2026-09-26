# Does the team earn its cost on the questions it was built for?

**Status:** pre-registered on 2026-09-14. This file is committed before any
question in `evals/breadth/questions.jsonl` exists and before any arm has been
run against one. Whatever it decides is what gets reported.

## Why this exists

The three-arm run on 30 held-out questions found the team costs 5.8 model calls
per question against 1.0 for the baseline and buys nothing measurable. That key
cannot see what the team is for: every held-out question was written from one
paper's abstract, so it rewards finding one paper, and a team that gathers
limitations from many papers earns nothing for it.

Building a new key after seeing the team lose is how a benchmark gets rigged,
so the conditions below are fixed first. The team's purpose - breadth across
papers and evidence against a claim - was stated in the README before that run,
so the question type is its declared purpose, not a choice made from its
result.

## Questions

**24 questions, 8 of each kind.** Each is written by a model from the titles
and abstracts of **two papers**, and needs both to be answered well.

| kind | the question asks |
|---|---|
| comparison | how the two approaches differ, and on what |
| limitation | what limitation of one approach the other exposes or addresses, and at what cost |
| synthesis | what approaches exist to a shared problem, drawing on both |

**How the pairs are chosen, before any question exists.** A pair is two held
papers joined by an in-corpus citation edge - one cites the other - so the two
are on a related subject by the corpus's own record, not by anyone's judgement.
Both must be current versions with 25 to 90 body passages. Neither may appear
in `retrieval.jsonl`, `heldout.jsonl`, `heldout-generated.jsonl` or `gate.jsonl`.
Pairs are taken in a fixed order at a fixed stride, and no paper is used twice.

**How questions are written.** From the two titles and abstracts only. The
authoring code imports nothing that can search, and a test enforces that. A
question is rejected if it shares a run of five or more words with either
abstract (the held-out key's existing leak threshold), if it does not name its
subject, or if it asks about authors, affiliations or funding.

## Arms

Each arm is the command as it ships, with its defaults.

| arm | equivalent command |
|---|---|
| baseline | `verbatim ask` |
| ladder | `verbatim ask --deep`, with its tool-choosing agent rung live |
| team | `verbatim team`: up to 3 tasks, 2 passages per task, 12 steps |

One run per arm per question, on the full corpus, one model for all three.

## What is measured

**Mechanical, computed by code:**

| | measure |
|---|---|
| M1 | both required papers cited with a verified citation (yes/no) |
| M2 | distinct papers cited with a verified citation |
| M3 | citations rejected by verification |
| M4 | model calls |

**Human, scored blind on a 0 / 1 / 2 scale:**

| | 0 | 1 | 2 |
|---|---|---|---|
| H1 coverage | addresses at most one of the question's two subjects | both, one only in passing | both, substantively |
| H2 balance | one-sided | a second position mentioned without support | each position backed by its own citation |
| H3 limitations | none | a limitation stated without a citation | a limitation or counterpoint backed by a citation |

**Blinding.** For each question the three answers appear as A, B and C in an
order shuffled with a fixed seed. The mapping is stored apart from the scoring
sheet and read only by the script that computes the result. Declined answers
are shown with one neutral wording, because each arm's own decline message
would reveal which arm wrote it. Answer length still differs between arms and
cannot be hidden; that is a known limit of the blind.

**The scores are committed before the mapping is read.** Scoring happens
first, `evals/breadth/labels.json` is committed, and only then does the scoring
script unblind.

## Decision rule

The primary measures are **H1 coverage, H3 limitations and M1 both papers
cited**. For each, per question, the team is compared against the baseline and
against the ladder with a two-sided exact sign test.

The team **earns its cost** only if it beats **both** simpler arms on **at
least one** primary measure at p < 0.05, and is not significantly worse than
either on any primary measure. Otherwise it stays an opt-in command, and the
README says so with these numbers beside the cost.

## What this cannot show

Twenty-four questions is a small sample, and a sign test on it needs a large
and consistent effect to reach p < 0.05. A null result here means the team did
not show an effect this size, not that it has none. One run per arm means
generation noise is in every comparison. And questions written from two
abstracts are only as good as those abstracts: a pair whose papers do not
really bear on each other produces a question no arm can answer well.

## Amendment, 2026-09-26

Made before any score existed. Nothing above is edited; this section says what
changed and why.

**Blinding.** The fixed seed was written in the code, so anyone with the
repository could rebuild which arm wrote which answer - the blind was not
blind. Instead:

- Each blinding shuffles with a secret random seed, and the mapping is kept
  outside the repository.
- `evals/breadth/blind_manifest.json` holds the SHA-256 of the mapping and of
  the answers, and is committed before any score.
- The scores name the sheet they were made on. `breadth_score.py` refuses to
  unblind unless they come from that sheet, were committed after the manifest,
  and the mapping and answers still match its hashes. The commits it checked
  are recorded in the result.
- The mapping joins the repository only after unblinding, where anyone can
  check its hash against the manifest.

**Secondary statistics.** The result also reports the team's mean difference
from each simpler arm with a 95% paired bootstrap interval, and - if a second
blind rater scores the same sheet - how far the two agree (exact agreement and
weighted kappa). These describe; they do not decide. The verdict is still the
decision rule above, on the first rater's scores.

**Unchanged:** the questions, the arms, the measures, the 0 / 1 / 2 scale and
the decision rule.

**Limit.** Whoever generates the mapping can open it. The blind is strongest
when the person scoring is not the person holding the mapping.
