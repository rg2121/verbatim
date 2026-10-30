# Findings

What was measured, and what it decided. Every feature was measured against
something simpler, and most of what is worth reading here is what did not
work: query rewriting, HyDE, always-on title matching, question
decomposition, sentence and semantic chunking, two other embedding models, a
critic loop and a multi-agent team were each built, measured and left off.

Two habits run through all of it. A difference is only claimed when a sign
test or interval supports it - three decimals of mrr are not evidence. And a
negative result is recorded, not deleted.

- [How it is measured](#how-it-is-measured)
- [Retrieval](#retrieval)
- [Answers](#answers)
- [What was tried and rejected](#what-was-tried-and-rejected)
- [The ladder and the agents](#the-ladder-and-the-agents)
- [Conversation](#conversation)
- [What went wrong silently](#what-went-wrong-silently)
- [Known limitations](#known-limitations)

## How it is measured

| key | cases | written | for |
|---|---|---|---|
| `evals/retrieval.jsonl` | 87 | alongside the system | development |
| `evals/heldout.jsonl` | 93 | blind, from title and abstract, frozen before scoring | an honest estimate |
| `evals/heldout-generated.jsonl` | 400 | the same way, by a model | a larger held-out sample, always reported apart |
| `evals/conversational.jsonl` | 30 | blind, two turns | whether a follow-up still finds its paper |
| `evals/breadth/questions.jsonl` | 24 | from two linked abstracts, under [a pre-registered protocol](../evals/breadth/PROTOCOL.md) | whether `team` earns its cost |
| `evals/gate.jsonl` | 12 | against the committed sample | a CI tripwire, floor mrr 0.90 |

**The development key is inflated by about an eighth.** Its questions were
written while watching retrieval, so one that retrieved badly could be
reworded until it did not. `hybrid+graph`, then the default, scored mrr
**0.869** there and **0.758** on the held-out key: same corpus, same kind of
question. Held-out
questions are written from one paper's title and abstract by code that
imports nothing able to search, and each key is committed before it is
scored. A candidate question sharing a run of five or more words with its
abstract is rejected, because abstracts are passages here and it would be
found for that reason alone (134 of 536 generated candidates).

**How a batch was written moves its score more than any retrieval mode does:**

| batch | cases | hybrid mrr |
|---|---|---|
| hand-written, first | 36 | 0.675 |
| hand-written, second | 52 | 0.816 |
| generated | 400 | 0.918 |

So an absolute mrr means nothing without its batch. Rankings between modes do
replicate across batches; absolute scores do not.

```bash
uv run verbatim eval --cases evals/heldout.jsonl       # the honest number
uv run python evals/paired_bootstrap.py                # intervals between modes
uv run python evals/failure_taxonomy.py                # what the misses are
uv run verbatim eval-answers --cases evals/heldout.jsonl evals/heldout-generated.jsonl --sample 120 --runs 2
uv run python evals/three_arm.py --questions 20 --runs 2 --arms baseline,workflow,+critic
uv run python evals/conversation_arms.py               # follow-ups
uv run python evals/tables_ab.py --compare             # after --arm before and --arm after
```

## Retrieval

On the 88 answerable hand-written held-out cases:

| mode | r@1 | r@5 | r@10 | mrr |
|---|---|---|---|---|
| keyword-any | 0.62 | 0.76 | 0.84 | 0.697 |
| hybrid+titles | 0.65 | 0.85 | 0.92 | 0.737 |
| hybrid+compare | 0.68 | 0.85 | 0.92 | 0.752 |
| vector | 0.69 | 0.83 | 0.90 | 0.756 |
| hybrid | 0.68 | 0.85 | 0.92 | 0.758 |
| **hybrid+graph** (default until reranking) | 0.68 | 0.85 | 0.92 | 0.758 |

**At 88 cases none of these can be told apart**: every paired-bootstrap
interval contains zero, and hybrid+graph had once been chosen over plain
hybrid on 0.003 mrr - an unmeasurable difference reported as a win. The
reranked default that replaced it (below) is the first that can be: 0.75 /
0.95 / 0.98, mrr 0.829 on these 88, better on 15 and worse on 4 (p = 0.02).
Over all 488 held-out cases, the first comparison to say anything was
hybrid+graph against the always-on title leg:

| batch | difference | won-lost | sign test |
|---|---|---|---|
| hand-written, first | -0.052 | 6-7 | p = 1.00 |
| hand-written, second | +0.071 | 13-0 | p = 0.0002 |
| generated | +0.019 | 38-19 | p = 0.016 |
| **all 488** | **+0.019** | **57-26** | **p = 0.0009** |

The title leg is reliably worse on *direction*; the mean difference is too
small for its interval to exclude zero, so it is not claimed as a magnitude.

**The graph gate moves none of the 488.** Blind authoring cannot produce
definitional questions, so `hybrid+graph` and `hybrid` rank every held-out
case identically. The gate is justified on the shape it targets, where the
effect is visible case by case: on the development key, "what is the Adam
optimizer" goes from nothing in the top ten to rank 3 and BERT's definitional
question from rank 2 to 1, while one case ("what is artificial intelligence")
moves from 1 to 3. The comparison gate fires on 6 of 488 and wins 5 (p = 0.22).

**The misses are competition, not vocabulary.** Labelled mechanically:

| | cases |
|---|---|
| ranked first | 60 (68%) |
| ranked 2-10 | 21 (24%) |
| missed, a near neighbour took first place | 5 (6%) |
| missed, unexplained | 2 (2%) |
| missed because the question's words are not in the paper | **0** |

The paper that beat the expected one is topically adjacent in almost every
case (abstract similarity 0.62-0.95). Each held-out question names one
acceptable paper, so an answer from an equally good neighbour scores as a
miss: **0.758 is a floor**, which the answer-level scores below confirm.

**Compound questions** need every listed paper, not any one - ten excellent
passages about PPO answer half of "how does GRPO differ from PPO". Scored by
coverage over 11 development cases:

| mode | cov@5 | all@5 | cov@10 | all@10 |
|---|---|---|---|---|
| hybrid | 0.41 | 0.09 | 0.59 | 0.36 |
| hybrid+plan (decomposition) | 0.36 | 0.09 | 0.59 | 0.36 |
| hybrid+titles | 0.64 | 0.45 | 0.68 | 0.45 |
| **hybrid+compare** | 0.59 | 0.36 | **0.73** | **0.55** |

Decomposition lost on the questions it was built for. The comparison-gated
title leg wins at depth and cannot touch single-subject questions, but stays
off: its cost on a comparison question with a *single*-paper answer has no
number yet, and unmeasured risk ships off.

**Tables were never searched until 2026-09-25.** The filter keeping
bibliographies out had been written as "body passages only" before tables
existed, so all 64,742 table passages were excluded. Letting them in was
measured first, under a rule fixed before either arm ran (ship unless worse at
p < 0.05), on all 488 held-out cases:

| | r@1 | r@5 | r@10 | mrr |
|---|---|---|---|---|
| tables excluded | 0.85 | 0.94 | 0.97 | 0.889 |
| tables searched | 0.85 | 0.94 | 0.97 | 0.889 |

Better on none, worse on one (rank 2 to 3), p = 1.0, so tables ship. They gain
nothing measurable either, as expected: questions written from abstracts do
not ask about the numbers in a table.

**ColBERT reranking pays on the held-out key, and gated it costs a tenth of
a second (2026-10-28).** ColBERT (answerai-colbert-small-v1: late interaction, 96
numbers per token, each question token matched to its best passage token)
reranked the top 50 fused passages, either on every question or only when the
keyword and vector legs put different papers first. The rule was written into
`evals/retrieval_ab.py` before either arm ran, though committed only with the
result: beat hybrid+graph at p < 0.025 by sign test, with under 2.5% of
bootstrap resamples at or below zero. On all 488 held-out cases:

| | r@1 | r@5 | r@10 | mrr | won-lost | median s |
|---|---|---|---|---|---|---|
| hybrid+graph (then the default) | 0.85 | 0.94 | 0.97 | 0.889 | | 1.18 |
| + ColBERT, every question | 0.92 | 0.98 | 0.99 | 0.946 | 59-14 | 2.20 |
| **+ ColBERT, legs disagree** | **0.92** | **0.98** | **0.99** | **0.943** | **48-6** | **1.31** |
| + cross-encoder, every question | 0.89 | 0.98 | 0.98 | 0.923 | 52-27 | |
| + cross-encoder, legs disagree | 0.89 | 0.98 | 0.98 | 0.925 | 42-16 | |

Both ColBERT arms pass (p < 0.0001, no resample at or below zero). The
cross-encoder rows were added afterwards, to ask what did the work, and decide
nothing.

**The gate keeps the gain where it is.** The legs disagree on 111 of the 488
(23%), and those hold 59 of hybrid+graph's 73 first-place misses. There ColBERT
lifts r@1 from 52 to 86 of 111. Where they agree hybrid+graph is already first
on 363 of 377, and reranking those too is a coin toss: 11 better, 8 worse
(p = 0.65). So the gated arm keeps all but 0.003 of the gain, loses 6 cases
instead of 14, and spends its second only where it fires. Where ColBERT wins,
fusion had sided with the wrong leg; its six losses are near neighbours, such
as Humanity's Last Exam for a question written about MMLU.

**On the development key every reranker loses again, on definitions.** The
MiniLM cross-encoder that [lowered mrr there](#what-was-tried-and-rejected)
raises it on the held-out key, to 0.923 on every question (p = 0.007) and
0.925 gated; ColBERT beats it by about 0.02 either way (intervals +0.008 to
+0.039 and +0.007 to +0.029), though neither sign test clears (p = 0.08,
0.06). But over the 63 development cases gated ColBERT falls from 0.878 to
0.789 (2-11, p = 0.02) and the cross-encoder to 0.761. Ten of the eleven
cases gated ColBERT moves down are definitional - "what is the Adam optimizer"
leaves the top ten - because a reranker rewards the passages that use a term
most, not the paper that introduced it, undoing what the graph leg fixed. The
held-out key holds no such question, so it could not see this. Standing the
reranker down on definition-shaped questions restores the development key
exactly (1-1) and keeps 0.939 on the held-out one (46-6), computed from the
saved ranks, which is exact because standing down returns hybrid+graph's order.

**Sparing definitions, it is now the default.** That arm then ran for real,
under a rule written after the prediction, so the run confirms rather than
tests: beat hybrid+graph on the held-out key as before, and be worse on no more
development-key cases than better. Every rank matched the prediction, 488 of
488 and 63 of 63:

| | held-out mrr | won-lost | development mrr | won-lost | median s |
|---|---|---|---|---|---|
| hybrid+graph | 0.889 | | 0.878 | | 1.18 |
| **+ ColBERT, legs disagree, not on definitions** | **0.939** | **46-6** | **0.878** | **1-1** | **1.27** |

It passes (p < 0.0001, no resample at or below zero), so every front end now
searches this way: `default_search_options` in `retrieval.py`, scored as
`hybrid+graph+rerank` by `verbatim eval` and enforced by the CI gate. It reranks
106 of the 488, at 1.89 s median there against 1.19 s elsewhere. It also stands
down wherever a title leg ran, so `--use-compare`, `--use-titles` and the
ladder's title rungs keep the behaviour they were measured with. `--no-rerank`
or `VERBATIM_RERANKER=off` turns it off. The CI gate cannot see it work: on the
committed sample's 12 questions the legs never disagree.

**By batch the ColBERT arms split, noticed after the fact and not claimed.**
On the 88 hand-written cases reranking every question scores 0.865 against the
gate's 0.829 (default 0.756): it also fixes 6 cases where both legs agreed on
the wrong paper, and breaks none. On the 400 generated it demotes 8 right
answers both legs agreed on, and fixes 5. Hand-written questions paraphrase;
generated ones repeat the paper's own terms, and then agreement is usually
right. The legs disagree on 40% of hand-written questions against 19% of
generated, so on questions people write the gate fires twice as often. Gated
against ungated needs a fresh hand-written batch to settle.

**Batching cost more than the model.** fastembed encodes passages in batches
padded to the longest. Fifty at once took 2.1 s, and its working memory pushed
the 4.9 GB corpus out of the page cache on a 16 GB machine, so the searches
after a rerank slowed even where none ran (median 1.21 to 1.71 s in a
controlled run). One passage at a time takes 1.0 s, leaves the next search
alone and ranks all 488 cases identically. The gated arm then adds 0.13 s at
the median, about a second on the questions it fires on. The cross-encoder,
still batched, shows the same symptom (p90 16 s), so it has no time above.

## Answers

`eval-answers` scores what the system actually produced. Over **120 held-out
questions drawn evenly from all three batches, answered twice**:

| | |
|---|---|
| declined outright | 0% |
| at least one verified citation | **92%** |
| every claim-carrying sentence has a marker | **88%** |
| cited the paper the question was written from | **88%** |

Retrieval puts the expected paper first 68% of the time; the answer cites it
88% of the time, because it can draw on any retrieved passage. Both runs
scored 111/120 exactly - an earlier "10-point run-to-run swing" was a 20-case
denominator, not the model.

**Does a cited quote support its claim?** That cannot be counted, so a model
judge was scored beside controls it cannot pass by topic alone:

| pairs judged "supported" | |
|---|---|
| real claim and its quote | 89% |
| hard control: a quote from the *same* answer, wrong claim | **62%** |
| easy control: a quote from another question | 0% |

Against the easy control the judge looks excellent; against the hard one it
separates by 26 points and accepts 62% of known mismatches. That is real
signal - entailment is a question a model can partly answer, unlike
relevance - and far too weak to gate anything. It is the evidence for keeping
verification a substring check.

## What was tried and rejected

**Asking a model whether passages are relevant.** Measured against questions
the corpus cannot answer:

| approach | declines correctly | keeps real answers |
|---|---|---|
| no check at all | 2/6 | 14/14 |
| verdict inside the answering call | 1/6 | 13/14 |
| separate judge, 4B model | 2/6 | 14/14 |
| separate judge, 13.8 GB model | **0/6** | 14/14 |
| require two supporting papers | 4/6 | 8/14 |

The larger model was worse, so this is not about size: passages are retrieved
*because* they match, and every model reads that as relevance. What works is
not asking - the model names what the passages are about, and that is
compared to the question arithmetically.

**That subject check works on one kind of unanswerable question and is blind
to the other.** "No paper is about this" and "the subject is covered and the
question still has no answer" were one label until split:

| kind | cases | caught |
|---|---|---|
| no coverage ("the farthest star from the earth") | 9 | **7** |
| covered, unanswered ("what is a tensor" - 177 abstracts mention one) | 4 | **0** |

Raising the warning threshold to catch more costs more than it buys:

| threshold | coverage gaps caught | false warnings on 63 good answers |
|---|---|---|
| **0.65** (default) | 7/9 | 1 (2%) |
| 0.70 | 7/9 | 5 (8%) |
| 0.75 | 8/9 | 16 (25%) |
| 0.79 | 9/9 | 27 (43%) |

Abstract-level similarity and lexical coverage separate the two kinds no
better than passages do. A line saying "no paper here is named after X" was
built and removed: it fired on 13 of 19 good definitional answers, and a
warning that frequent is one readers learn to skip.

**Everything that widens the candidate pool trades top precision for depth.**
On the 63 answerable development cases:

| mode | r@1 | r@5 | r@10 | mrr |
|---|---|---|---|---|
| **hybrid** | **0.81** | **0.94** | 0.95 | **0.866** |
| hybrid + titles | 0.75 | 0.94 | 0.97 | 0.822 |
| hybrid + 3 rewrites | 0.75 | 0.90 | **0.98** | 0.826 |
| hybrid + HyDE | 0.70 | 0.92 | 0.97 | 0.799 |
| hybrid + decomposition | 0.76 | 0.87 | 0.92 | 0.816 |

With the right passage already first 81% of the time, new candidates can only
displace it. Cross-encoder reranking is the same trade, worse: mrr 0.940 fell
to 0.752, 0.701 and 0.741 on the three rerankers tried (against the 42-case
set of the time). On the held-out key, over 50 candidates, it helps instead;
on today's development key 14 of its 16 losses are definitional questions
([Retrieval](#retrieval)). The graph leg is the one widener that pays no
recall@1, because it widens only the questions the baseline was already losing.

**Structure and prompting.** Tool calling for the answer works on `gemma4:31b`,
where it failed on a 4B model, and is no better than the line protocol:

| out of 20, 3 runs | line protocol | tool calling |
|---|---|---|
| answers with a verified citation | 17.0 | 16.0 |
| every claim carries a marker | 15.0 | 13.7 |
| cited the expected paper | 16.7 | 16.0 |

Telling the model more forcefully to cite every sentence made things worse:

| out of 20 | current prompt | stricter prompt |
|---|---|---|
| every claim carries a marker | 16.0 | 15.7 |
| answers with a verified citation | 17.3 | 16.3 |
| citations rejected | 1.3 | **2.7** |

Pressed to cite more, the model reaches for quotes it cannot reproduce
exactly, and verification throws them out.

**Embedding each passage with its title and section** is unmeasured rather
than rejected: an A/B scored mrr 0.982 for both arms on a subset too easy to
show a difference (r@5 was 1.000). It stays off until measured on a harder set.

**Cutting at sentence ends, or where the subject changes, does not beat
windows (2026-10-29).** Re-cutting means re-embedding, about 12 hours for
the whole corpus, so this ran on an eval corpus of 1,500 papers: the 589 the
keys name, each one's nearest paper by abstract, and a seeded random fill
(`sampling.export_eval_corpus`, kept under `data/`). Three copies were cut from
the same source files. Windows is today's way: 1,200 characters with a
200-character overlap, cut at the last word that fits. Sentences packs as many
whole sentences as fit. Semantic ends each passage, somewhere between 600 and
1,200 characters, at the sentence boundary where the sentences on either side
are least alike to bge-small (two a side, each sentence's first 400
characters). Both new ways overlap by whole sentences, and tables stay whole
in all three. The rule was written into `evals/chunking_ab.py` before any copy
was cut, though committed only with the result: beat windows on the 488
held-out cases, searched with the shipped default, at p < 0.025 by sign test
with under 2.5% of bootstrap resamples at or below zero, and be worse on no
more development-key cases than better.

| | passages | median chars | end a sentence | held-out mrr | won-lost | development mrr | won-lost |
|---|---|---|---|---|---|---|---|
| **windows** | 128,619 | 1,125 | 38% | **0.969** | | **0.951** | |
| sentences | 132,556 | 942 | 80% | 0.970 | 15-12 | 0.909 | 1-5 |
| semantic | 140,608 | 805 | 82% | 0.967 | 13-14 | 0.907 | 0-6 |

Neither passes (p = 0.70 and 1.0), and both lose on the development key, so
windows stays. Each moves 27 held-out cases, about as many down as up. On these
1,500 papers windows misses first place on only 23 of 488, so the comparison
was also run with plain hybrid search (no graph leg, no reranker), which misses
41. That run was added after windows scored and before either other copy did,
and decides nothing; it agrees: 0.941 for windows, 0.935 for sentences (15-23),
0.944 for semantic (21-22). On the development key most of the losses are
definitions windows already ranked first: "what is the Penrose diagram" falls
to 7th with sentences and 3rd with semantic, and three questions are lost by
both.

**Semantic cuts are also the expensive ones.** Finding where the subject
changes means embedding every sentence: 4.7 s a paper, against about 1 ms for
windows or sentences, timed alone on the same 60 papers (parsing the HTML
takes 18 ms). Across the 7,887 papers of the full corpus that is about 10
more hours of chunking, then 9% more passages to embed. Built side by side on
the eval corpus, sentences took 13 minutes to cut and 3.9 hours to embed;
semantic 3.6 hours and 3.0 hours.

**Answers do not separate them either.** Sixty held-out questions, drawn once
with a fixed seed, were answered on each copy by a local model
(`gemma4:e4b-mlx`) and counted as `eval-answers` counts them, with no judge:

| out of 60 | windows | sentences | semantic |
|---|---|---|---|
| answers with a verified citation | 56 | 56 | 56 |
| every claim carries a marker | 44 | 45 | 44 |
| cited the expected paper | 50 | 52 | 54 |
| citations rejected | 12 | 9 | 13 |

Paired, semantic cites the expected paper on 8 questions windows misses and
misses it on 4 (p = 0.39); sentences on 8 and 6 (p = 0.79). Generation is not
seeded, so some of those flips are the model's own, and the counts were set
beforehand to decide nothing. `VERBATIM_CHUNKER` stays, set to windows.

**A wider or newer small encoder does not beat bge-small either
(2026-10-29).** The same eval corpus, cut into windows, was embedded again by
two other models, each given the query and passage prefixes it was trained
with: bge-base-en-v1.5 (768 numbers a passage, twice today's) and
snowflake-arctic-embed-s (384, like today's). The rule, in
`evals/encoder_ab.py` before any arm was scored, is the chunking one with the
threshold split between two arms. Two larger candidates, arctic-embed-m and
nomic-embed-text-v1.5, were dropped on timing alone, before any run: about 12
hours and 16 days to embed this corpus here. Giving each model its own
prefixes leaves bge-small unchanged; its ranks match the chunking run case for
case.

| | numbers a passage | ms a passage | held-out mrr | won-lost | development mrr | won-lost |
|---|---|---|---|---|---|---|
| **bge-small-en-v1.5** | 384 | 51 | **0.969** | | **0.951** | |
| bge-base-en-v1.5 | 768 | 111 | 0.976 | 9-4 | 0.955 | 1-3 |
| snowflake-arctic-embed-s | 384 | 57 | 0.967 | 9-12 | 0.933 | 1-5 |

bge-base comes closest and still fails twice: p = 0.27, with 3.4% of
resamples at or below zero against the 2.5% allowed, and on the development
key it is worse on 3 questions and better on 1. That one gain is large (a
question with the right paper outside the top 10 now ranks it first), so its
development mrr rises while the count goes against it. The losses are
concept questions a place or two lower: "what is entropy?" 2nd to 3rd, "how
does Adam adapt its learning rate per parameter?" 3rd to 5th. arctic-s loses
definitions too ("what is the einstein field equation in flat space
approximation?" 1st to 4th). Plain hybrid search, reported to decide nothing,
agrees: 0.938 for bge-base (19-22) and 0.936 for arctic-s (19-23) against
0.941, and on the development key bge-base is better on none and worse on 6.

Timed alone on the same 512 passages in the batches of 64 that `embed` uses,
bge-base costs twice as much: about 26 hours to embed the full corpus against
12, a vector index twice the size (2.6 GB against 1.3 GB), and a median
search on the eval corpus of 0.26 s against 0.21. arctic-s costs about what
bge-small does and buys nothing. bge-small stays.

Four arms in two experiments, two cuts and two encoders, have now failed to
separate from today's on this key. The shipped search misses first place on only 23 of the 488
held-out cases, so that is all any arm can gain, and any change to the
vectors or the passages moves about as many cases down as up. What the next
retrieval experiment needs is a harder key, not another arm: hand-written
questions that name methods, and more definitions.

## The ladder and the agents

**The ladder (`ask --deep`) against one retrieval and one answer**, over all 88
answerable held-out questions, two runs each:

| out of 88 | plain | ladder |
|---|---|---|
| answers with a verified citation | 78.5 | **81.5** |
| every claim carries a marker | 70.5 | **72.5** |
| cited the expected paper | 72.0 | **74.5** |
| citations rejected | 10.5 | **8.5** |

Ahead on every row, and **not distinguishable**: the arms agree on 81 of 88
questions, and the 7 that differ split 6-1 (p = 0.125). A bootstrap interval
that seemed to exclude zero did so only for some random seeds. It costs about
2.3x the model calls (measured on the 20-question run below), and only where
the plain path was going to fail - its first rung is the plain path. So it
ships behind a flag.

**The critic made things worse** (20 questions, 2 runs):

| out of 20 | plain | ladder | ladder + critic |
|---|---|---|---|
| answers with a verified citation | 17.5 | **18.0** | 17.0 |
| every claim carries a marker | 15.0 | **16.0** | 14.0 |
| citations rejected | 2.5 | **1.5** | 3.0 |
| cost | 20 calls | 46 calls | 83 calls |

Escalating on an uncovered subject buys another answer that may be weaker than
the one it replaced. The critic stays off, useful as a report, not as a loop.

**The ladder does not catch the covered-but-unanswered case either.** It
declared a gap on 0 of 4 such questions, 6 of 9 coverage gaps, and 0 of 25
answerable ones - a subset of what the one-call subject check already catches.

**Two supervisors, the same roles.** The built-in loop and the LangGraph
version gave the same route on 57 of 60 held-out questions; the built-in one
run twice agreed with *itself* on only 55. The difference between the
implementations is below the model's own run-to-run variation, so the roles
really are separable from the loop that drives them. Running sections in parallel buys
nothing against one Ollama server: a four-section review took 424 s with one
worker and 423 s with three.

**`team` costs 5.8x and buys nothing measurable** (30 held-out questions):

| out of 30 | plain | ladder | team |
|---|---|---|---|
| answers with a verified citation | 26 | **28** | 26 |
| cited the expected paper | 24 | **25** | 23 |
| model calls per question | **1.0** | 2.3 | 5.8 |

The held-out key cannot see what a team is for - 63 of its 144 citations
rested on passages only its adversarial search found - so a second test was
pre-registered for two-paper questions before any question existed. On its one
mechanical primary measure the team is ahead and not significantly (cited both
required papers on 7 of 24, against 3 for plain and ladder; p = 0.125 and
0.219). **Status: awaiting blind human scoring; no verdict yet.**

**`gaps`, measured by planting gaps whose answer is known** - a claim a real
paper refutes, run with that paper reachable and then hidden:

| out of 25 planted gaps | |
|---|---|
| refuted when the paper was reachable | **23** |
| missed evidence that was there | 2 |
| still refuted with the paper hidden, by a neighbour | 4 |

Falsifier recall is **92% (95% interval 75-98%)**, which is what licenses
reading a gap that stands as "nothing found by an opponent who tried" - not as
absence. Both misses cleared the effort floor and were wrong anyway, so about
**one standing gap in twelve is false**. The first run measured 100%, an
artefact of claims that borrowed the wording of the paper that refuted them.

## Conversation

A follow-up does not carry its own subject, so the page resolves it against
the conversation before retrieval. Over 30 two-turn cases:

| | r@1 | r@5 | mrr |
|---|---|---|---|
| the opening question | 0.97 | | 0.973 |
| the follow-up as typed | 0.30 | 0.43 | 0.357 |
| **resolved against the conversation** (ships) | **0.97** | **1.00** | **0.983** |
| resolved, fused with the follow-up as typed | 0.50 | 1.00 | 0.704 |
| resolved, searched within the previous turn's papers | 0.97 | 1.00 | 0.983 |

**Resolving is the largest effect measured in this project** - better on 21 of
30 cases, worse on none. Fusing the raw follow-up back in is worse, the
widening trade again. Scoping to the previous turn's papers changed nothing,
and cannot be judged on this key: every case is answered by the paper the
opening turn found. Follow-ups whose answer is in a different paper, and
conversations longer than two turns, are unmeasured.

## What went wrong silently

Each of these ran for weeks without raising an error, and most hid behind a
cheaper question that usually has the same answer as the real one:

| fault | the proxy it hid behind |
|---|---|
| a failed sync advanced its watermark, skipping pages for good | did the run *advance*? - not did it *finish* |
| a re-chunk that yielded nothing deleted nothing, leaving stale passages citable | what did the pass *produce*? - not what did it *attempt* |
| the vector index served another model's vectors | how many vectors? - not whose |
| retrieval was tuned on a key written while watching it | three decimals of mrr - not an interval |
| every table passage was excluded from search | a filter naming what to keep, written before tables existed |
| a re-sync erased licences already found (50 papers) | "no licence reported" read as "no licence" |
| every licence lookup failed after arXiv moved its OAI-PMH service | a failed lookup is retried, so nothing looked wrong |

Proxies like these agree with the real question in the common case and fail
only in the uncommon one - which is where an unattended job lives. The run
ledger exists because of the first three: nothing recorded how the corpus got
into its state.

## Known limitations

- **It will answer a question the corpus cannot answer.** Every quote is real,
  but a real quote does not prove the passage answers the question. The subject
  line catches 7 of 9 questions on uncovered subjects and 0 of 4 whose subject
  is covered but unanswered - the second kind is not a retrieval problem, and
  nothing tried detects it.
- **The corpus starts as a slice of recent work.** `grow` fills in what it
  cites - GPT-4, Adam, BERT and PPO were each cited dozens of times and absent
  until it ran - but "what is a tensor" still has no defining paper to anchor to.
- **Scanned PDFs yield nothing**; there is no OCR. 32 papers have no PDF at
  arXiv and are recorded as such.
- **Retrieval numbers are corpus- and batch-specific**, and most mode
  differences are inside the noise at these sample sizes.
- **Licences are only as good as arXiv's records.** `unknown` means unknown,
  not permitted, and arXiv's current OAI-PMH service has no record for some
  papers that exist, so their licences stay unknown until it does.
- **A question about what is new lists what is new**, filtered by date and
  category but not by topic.
- **Vector search is brute force** - exact, and about 0.2 s over 835,743
  vectors. A corpus tens of times larger would want an approximate index
  behind the same interface.
