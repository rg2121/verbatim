# 11. A critic that describes, and code that decides

**Status:** accepted; `report --agents --critic` is off by default.

## Context

Every reference design for a multi-agent research system includes a critique
agent that judges whether the evidence supports the answer. This project had
refused to build one, because it had measured that exact thing seven times:
no check at all caught 2/6 unanswerable questions, an inline verdict 1/6, a
4B judge 2/6, a 13.8GB judge **0/6**. The larger model was the worst. Passages
are retrieved *because* they match, and a model reads that as evidence they
suffice.

Refusing outright left a real gap, though. The answering prompt has always
instructed that every claim carry a citation marker, and nothing ever checked
whether one did.

## Decision

Build the critic, and let it describe rather than judge. Everything it says is
checked before it counts:

| finding | how it is established |
|---|---|
| unsupported | sentences carrying no citation marker — pure counting, no model |
| unaddressed | the model lists what the *question* is about; any subject actually present in a passage is dropped |
| conflicts | the model names pairs of cited passages that disagree; a pair survives only if both ids were retrieved and cited |

The recommended action is derived by code from those three and never asked
for — a model that recommends its own next step is judging again under another
name. Only `unaddressed` escalates to the next retrieval strategy, because
only it is something a different retrieval could fix; an uncited sentence is a
writing problem and is reported to the reader instead.

## Consequences

The critic runs with no model at all, degraded to the half that is arithmetic,
which is also the half that fires most.

Measured over 20 answers to held-out questions on the real corpus: the
uncited-claim check fires on **20%** of answers (9% of all claim-carrying
sentences), the unaddressed-subject check on **5%**, and the conflict check on
**0%**. The first two are informative without being noise — for comparison,
the "no paper is named after X" line was removed for firing on 68% of good
answers, and the subject threshold was left at 0.65 rather than 0.79 because
0.79 would warn on 43% of correct ones.

The conflict check has never fired and costs one model call per section. It is
kept because `--critic` is opt-in, and recorded here as unproven rather than
useful. If it is still at zero after a larger sample, it should go.

**Measured 2026-09-10, and it does not pay.** Against the workflow without it,
over 20 held-out questions and 2 runs: fewer answers with a verified citation
(17.0 against 18.0), fewer fully marked (14.0 against 16.0), twice the
rejected citations (3.0 against 1.5), and **none of the three baseline
failures recovered**, for 83 model calls against 46. The mechanism is that
escalating on an uncovered subject buys another answer that may be weaker than
the one it replaced, and more attempts mean more chances to produce a quote
that fails verification.

The critic therefore stays off and is recorded as **measured worse, not
unproven**. It keeps its value as observability - the uncited-claim count is a
fact about the text and needs no model - but not as a control loop. If it
earns a place later it will be by reporting rather than by escalating.
