# Verbatim

A local-first, citation-focused literature-review assistant for scientific papers.

**[What it looks like](demo/transcript.md)** · **[Architecture](docs/architecture.md)**
· **[Evaluation](docs/evaluation.md)** · **[Decisions](docs/decisions)**
· **[Postmortems](docs/postmortems)**

Every claim in an answer carries a quote, and every quote is checked against
the passage it cites before the answer is shown. What the project has learned
is mostly what did *not* work: reranking, query rewriting, HyDE, always-on
title matching and question decomposition were each built, measured, and left
off, and a held-out answer key later showed that the differences used to rank
several of them were inside the noise. Those measurements are in
[docs/evaluation.md](docs/evaluation.md) and
[docs/postmortems](docs/postmortems).

## Architecture

`data/corpus.db` is the cumulative source of truth. Everything else — the JSON
snapshot, the PDFs, the chunk export — is derived and can be rebuilt from it.

```
arXiv API ──sync──▶ corpus.db  papers            every version seen, newest current
                              downloads         local PDFs, checksummed
                              download_failures the ones arXiv has no PDF for
                              html_sources      local HTML renderings, checksummed
                              paper_references  bibliographies, read from the HTML
                              chunks            passages + char offsets, from either
                              chunks_fts        BM25 index over passages
                              papers_fts        BM25 index over titles
                              embeddings        one vector per chunk per model
                              paper_embeddings  one vector per paper's abstract
                              vec_chunks        nearest-neighbour index, passages
                              vec_papers        nearest-neighbour index, papers
                    │
           download ──▶ data/pdfs/*.pdf
               html ──▶ data/html/*.html
         references ──▶ what the corpus cites and does not hold
               grow ──▶ adds those papers by identifier
              graph ──▶ what cites a paper, what it builds on, who is cited most
              chunk ──▶ data/chunks.jsonl   (export, not the store)
              embed ──▶ vectors in corpus.db
           pipeline ──▶ every stage above, in rounds, until nothing is pending
               runs ──▶ what each command did, and whether it finished
             search ──▶ ranked passages with citations
                ask ──▶ an answer, every claim cited and verified
             report ──▶ a sectioned literature review, one verified bibliography
```

## Try it without building a corpus

The repository ships a small corpus so the tool works on a fresh clone, with
nothing to fetch and nothing to embed:

```bash
uv run verbatim search "how do transformers handle long context" --state sample/corpus.db
uv run verbatim ask "what is a neural turing machine?" --state sample/corpus.db
uv run verbatim graph --state sample/corpus.db
```

Forty papers, 3,139 passages and their vectors, 2,230 bibliography entries,
about 23 MB. A share of the papers is picked as a citation neighborhood - a
hub and the papers that cite it - so the `graph` command demonstrates edges
on a clone rather than printing zeros. It carries no PDFs or renderings, only
the database, so `stats` correctly reports zero PDFs rather than claiming
files it does not have. Every passage still carries its own arXiv id,
section, offsets, checksum and licence, so citations are complete. `ask`
needs a model through Ollama; `search` and `graph` need nothing.

Regenerate it from a full corpus with `verbatim sample --papers 40`.

## Daily corpus workflow

```bash
uv run verbatim sync --category physics.gen-ph --max-results 5
uv run verbatim download --limit 3
uv run verbatim chunk
uv run verbatim embed
uv run verbatim search "how does a changing field induce current in a loop"
```

`sync` pages through one or more categories, newest-updated first, and upserts
into the catalogue. It tracks full arXiv versions and marks superseded versions
without deleting them.

```bash
uv run verbatim sync --category cs.CL cs.IR --limit 2000
```

A stored watermark per category is what makes this resumable: paging stops once
results are older than the last run, so re-syncing costs one request rather than
walking the category again.

The watermark only looks **forward in time**. A category first synced with a
small `--limit` stays at that size, because everything older is already "behind"
the mark — a later `sync --limit 5000` on it fetches nothing. To go *deeper*
into a category's history, use `--full`, which ignores the watermark and pages
from the newest paper again:

```bash
uv run verbatim sync --category cs.LG --limit 5000 --full
```

Already-known papers are refreshed rather than duplicated, so the only cost is
re-requesting their metadata. arXiv's API is variable — the
same query has taken 2s and 61s, and returns 503 under load — so requests are
retried with backoff, and a page that still fails stops the run cleanly with
everything already fetched committed and the watermark recorded.

`download` selects the newest current versions that have not yet been
downloaded. Cached PDFs are reused only after their checksum matches the
previous manifest, and downloads and generated artifacts are written
atomically. Add `--require-license` to restrict downloads to records with an
explicit licence in arXiv metadata, or `--license open` to restrict to papers
whose licence actually permits redistribution and derivatives. The two are not
the same: arXiv's default licence grants **arXiv** the right to distribute and
grants you nothing, and CC terms carrying NC or ND rule out commercial use and
training respectively. Of the 5,453 papers looked up so far, 49% are open, 41%
arXiv-nonexclusive, 5% no-derivatives and 4% noncommercial.

`html` fetches arXiv's HTML rendering of each paper, which is generated from
the author's TeX rather than from a printed page. A paper with no rendering is
recorded as such, so it is asked for once and never again.

Tables are read as tables. LaTeXML marks up every cell, so a results table
becomes a Markdown pipe table carried as its own passage rather than being
flattened into prose - "Person Corp Time Position Musk SpaceX 2002 CEO" gives a
reader no way to see which four values are one row. Cells spanning rows or
columns are placed in their real columns, which matters because 19% of the
tables here use rowspan and 29% colspan, and ignoring a span shifts every later
cell left so the table states something the paper does not. Display equations
are laid out as tables too and are deliberately not read as such: their LaTeX
already comes from each `<math>` element. Measured over 1,547 renderings: 9,343
tables, 84,829 data rows.

Layout extraction runs in a child process with a timeout. It is C code with no
internal limit, and one 7.7 MB paper in this corpus held a core for 46 minutes
without returning, which blocked its batch and through that the whole ingest -
twice, before it was diagnosed. A paper that exceeds the limit is skipped and
marked chunked so it is not retried for ever; it contributes nothing rather
than stopping everything.

`references` reads each paper's bibliography out of its HTML rendering, where
LaTeXML marks every entry as its own element rather than leaving a run of text
to be guessed apart. An entry is resolved to an arXiv identifier or DOI only
when it states one; resolving by title similarity would be a guess recorded as
a fact.

Measured over all 398,802 entries, 80,469 carry a resolvable arXiv identifier
and 12,882 point at a paper this corpus holds - 12,781 distinct edges, 43% of
papers connected. An earlier version of this file said there was no graph here
at all; that was measured over 908 papers which happened to be the oldest in
the corpus, and it was wrong.

Its first use is outward, and `grow` follows it - the
papers most cited by this corpus and absent from it are GPT-4 (52 citing
papers), Llama 3 (38), DeepSeek-R1 (32), Adam (28), PPO (27), BERT (17). The
corpus leans on Adam and BERT constantly and holds neither, which is also the
likeliest reason a definitional question has no answer here: there is no
foundational literature in it to answer from.

`grow` adds those papers by identifier. It only fetches metadata; their PDFs,
renderings, chunks and vectors follow from the ordinary stages.

`graph` is the inward use of the same bibliographies: what a held paper builds
on, what cites it, and - with no identifier - the papers this corpus itself
cites most, which is an authority ranking no similarity score can produce.
Authority here means in-corpus citers, counted; at an average degree this low
a PageRank over the same edges is in-degree with extra steps. The current
top of that list is Qwen3 (371 citing papers), Llama 3 (306), DeepSeekMath
(290), GPT-4 (221), DeepSeek-R1 (217), PPO (188) and Adam (129).

`chunk` re-verifies each file's checksum, then extracts passages for only those
whose chunker version, chunker parameters, or bytes have changed. **A paper
with an HTML rendering is read from it in preference to its PDF**, because the
rendering is better input: maths survives as the author's own LaTeX rather than
the character soup a PDF extractor produces, sectioning comes from the author's
commands rather than a guess at font sizes, and none of the ligature or
reading-order damage of PDF extraction arises. Measured against PDF extraction
of the same papers, the words that HTML "loses" are almost entirely words the
PDF broke across lines — `asafunctionoftheeffective`, `characterlevel`. HTML
has no pages, so such a passage is cited by its section instead. A paper is
held as exactly one chunking, from one source. `chunk` also refreshes
`data/chunks.jsonl`, which is an export rather than the store.

`embed` vectorises chunks that have no vector for the chosen model, in batches,
and is a no-op on re-run. It uses `BAAI/bge-small-en-v1.5` through ONNX, so
there is no torch dependency and the weights are about 130MB.

`search` prints each passage with the paper version, page, character span, and
source URL needed to cite it. Use `--arxiv-id` to search within one paper and
`--full` for whole passages. Three modes:

| Mode | Finds | Misses |
|---|---|---|
| `hybrid` (default) | both of the below, fused by rank | — |
| `keyword` | exact terms, identifiers, notation | paraphrases |
| `vector` | meaning, without shared words | rare exact tokens |

Keyword mode ranks by how many terms matched; `--match all` requires every one.
Requiring every term scores mrr 0.389 against 0.866 on the eval set, which is
why it is no longer the default. The index matches whole tokens and does not stem, so `field` does not
match `fields` — use `field*`. Hybrid uses `any` internally, because requiring
every word of a natural-language question matches nothing at all.

`pipeline` runs every stage in rounds until nothing is pending. Running them
once in order does not finish the job: each stage creates work for the ones
after it, so growing the corpus leaves PDFs to download, downloading leaves
renderings to fetch, and chunking leaves passages to embed. Done by hand here,
that stranded 81 papers unchecked, 350 unchunked and 69 bibliographies unread.

It stops when a round changes nothing rather than when the counts reach zero,
because some work never completes — 33 papers in this corpus return a
truncated PDF every time they are asked for, and waiting for them would loop
for ever. Stages run as separate processes so that one failing does not end
the ingest: a DNS failure took out a licence run here and everything after it
should have carried on. A failed stage is retried while other stages are still
making progress, and the summary names what failed rather than hiding it.

`search --since` and `search --category` narrow retrieval to papers by date or
subject, which neither leg can do alone: similarity has no notion of recency
and a keyword index has no notion of subject. The keyword leg filters before
ranking; the vector leg over-fetches and filters afterwards, because the
vector index carries only chunk id, paper id and kind as metadata and
filtering on publication date would mean rebuilding it. A very narrow filter
can therefore return fewer results than asked for, rather than quietly ranking
papers it was told to exclude.

`stats` shows what each stage has done and what it has left, which is how a
long ingest stays observable:

```
  stage      done                           remaining              next
  papers     59                             59 current
  licences   19 known                       40 to look up          verbatim licenses
  pdfs       35 (0.1 GB)                    24 to fetch            verbatim download
  html       33 (2 have none)               24 to check            verbatim html
  chunks     3,184 (2,326 body / 858 refs)  0 pdfs to chunk        verbatim chunk
  vectors    3,184                          0 to embed             verbatim embed
```

Every stage is incremental and safe to re-run: nothing is redone unless its
input changed. Papers are upserted by arXiv ID; a PDF is fetched only if it has
no download record; a PDF is re-chunked only if its bytes, the chunker version,
or the chunk size changed; a chunk is embedded only if it has no vector *for
that model*; and the vector index adds and removes only what differs.
`verbatim index --rebuild` forces a full reload if you ever need one.

`ask` retrieves passages, answers strictly from them, and cites every claim:

```bash
uv run verbatim ask "can magnetic flux move through a region of zero field?"
```

```
Flux can move through zero field [1].

References
[1] Walstad, "Moving Magnetic Flux and Electromagnetic Induction",
    arXiv:2608.05440v1, p. 1. https://arxiv.org/abs/2608.05440v1
```

Every citation is checked in code before it is printed: the quoted words must
appear in the passage the claim names, or the citation is reported as rejected
rather than rendered.

That proves a quote is real. It does not prove the source is *about* your
question, and search always returns its closest passages even when the corpus
covers nothing on the subject. So `ask` also names what the passages actually
discuss, and says so when that is far from what you asked:

```
The name of the farthest star is not mentioned in the provided passages.

  ! these passages are about: Neutron star properties and distribution
    which may not be what you asked (0.53 similarity)
```

It warns rather than refuses, because the measurement does not support
refusing — see "What was tried and rejected" below. `--no-subject-check` skips
the extra call. Citations always pin the paper *version*. At most three
passages come from any one paper by default (`--max-per-paper`), so a
corpus-wide question is not answered from a single source.

Pass `--tools` and the model routes the question itself, which is what lets one
command serve three different kinds of question:

```bash
uv run verbatim ask "what came out in the last 10 days?" --tools
uv run verbatim ask "what is arXiv:2608.05440 about?" --tools
uv run verbatim ask "what do papers say about hybrid retrieval?" --tools
```

Recency and completeness are SQL questions — similarity search returns things
that *resemble* a query and has no notion of "this week" — so those go to the
catalogue rather than the vector index. The route taken is printed above each
answer. Tools only gather evidence: the answer still goes through the same
citation verification, so a claim whose quote is not in its passage is still
rejected.

`ask` answers with a local model through **Ollama**, so the whole pipeline runs
offline once the corpus is built. It picks `$OLLAMA_MODEL`, or the first pulled
model; override with `--llm`. Ollama is the only backend - there is no second
client, no API key and no hosted path around it - but which model it names is a
decision with a boundary in it. A local model keeps the corpus on this machine.
A model whose name ends `-cloud` is served by Ollama's own infrastructure, so
the passages in each prompt leave the machine, and the corpus is 26 GB of other
people's papers under mixed licences. Neither is wrong; the point is that the
choice is the user's and is visible in `OLLAMA_MODEL` rather than buried in a
default.

Small local models paraphrase rather than quote, so expect rejected citations —
that is the check working, not a bug. A claim whose quote could not be verified
is marked `[unverified]` in the prose rather than silently presented as sourced.
Ollama's `format` parameter and tool calling were both unreliable on realistic
multi-passage prompts, so the backend elicits a flat `ANSWER:` / `CITE` line
format and parses that.

Two shapes of that line are accepted, because models disagree about it:
`CITE [1] <chunk_id> :: <quote>` as the prompt asks, and
`CITE [<chunk_id>] :: <quote>`, which `gemma4:31b` writes every time. Reading
only the first threw away every citation that model made and marked a
correctly-quoted answer `[unverified]` throughout. Accepting both changes no
verification: the quote is still checked character-for-character against the
passage it names, and only the bookkeeping differs, since prose that cites by
chunk id was already resolved to the bibliography's numbering.

For a one-off metadata-only snapshot, use `fetch --category ...`. The legacy
one-off forms `download --input` and `chunk --manifest` remain available and
bypass the catalogue.

`eval --rewrites N` and `eval --hyde` score two query-side treatments beside
the existing modes. Rewriting asks a local model for other phrasings of the
question and fuses a ranking from each, so a passage found by several
phrasings rises; HyDE adds the sentence a paper answering the question would
contain, which puts the query in the register of the passages rather than of
the question. Both are off by default and unmeasured on this corpus, for the
same reason the reranker is: hybrid retrieval scores mrr 0.866 here, and an
unmeasured change is as likely to cost as to gain. The user's own words are
always retrieved for and are never displaced by a rewrite, and if the model is
unreachable both fall back to the original question rather than failing.

`ask --plan` answers a compound question in parts. One retrieval serves a
single-subject question well and a two-subject one badly: for "how do MOND and
dark matter differ in clusters", the passages matching the whole sentence come
from whichever half the wording leans towards. The question is split, each part
is retrieved for with its own quota rather than fused - fusion is what lets one
half crowd out the other - and the answer is written once by the same
verified-citation path. The split is printed, because a reader who disagrees
with how a question was divided needs to see the division.

It is a multi-step loop, not a swarm of agents, and it is off by default:
splitting a question that never needed splitting buys three mediocre
retrievals in place of one good one. Measured on the compound cases it was
built for, it lost - see *What was tried and rejected*.

`report` composes the measured pieces into a literature review. One model
call plans three to five section questions; each section then runs the same
retrieval `ask` uses (graph gate on, comparison gate behind `--use-compare`)
and is written by the same verified-citation path, so a claim cannot appear
anywhere in the review unless its quoted words are in the passage it names.
The assembly is what `ask` run several times would not give: one bibliography
numbered across sections with the inline markers rewritten to match, each
section's subject check printed beside it as a warning, and a section the
corpus cannot support saying so in place. `--out review.md` writes it as
markdown with version-pinned arXiv links. A hosted "deep research" product
can show neither its retrieval nor a verification of its quotes; this one is
built from parts whose numbers are all in this file.

**A definitional question loses to the papers that merely use the thing.**
`grow` added the Adam paper, and "what is the Adam optimizer?" still returns
nothing from it in the top ten: 3,152 papers here mention Adam, the paper that
defines it holds 53 passages, and every result is somebody's "we trained with
Adam" hyperparameter section. Adding the foundational paper was necessary and
not sufficient.

The paper-abstract index does not fix it either - the Adam paper is not in its
top four for that question, because an abstract about stochastic optimization
sits near every other paper about optimization. Matching the question's terms
against paper *titles* does find it at rank 1, and fails on "why does PPO clip
its objective?", where `clip` matches CLIP the model. So title matching is a
real signal and a partial one - measured always-on it cost more than it earned
(see below), and gated by the citation graph it is the fix: the leg fires only
on definition-shaped questions, and only for a title match that held papers
actually cite, so the corpus itself distinguishes the paper a subject is named
after from one that shares a word with it. "What is the Adam optimizer?" now
answers from the Adam paper.

## Asking a follow-up

The page is a conversation. Questions and answers stack in a transcript, so
what was asked three turns ago is still on screen with the answer it got, and
a follow-up like "why does it clip?" resolves against what came before without
restating the subject. "New topic" clears the thread and starts fresh.

The history is used for exactly one thing: turning a follow-up into a question
that stands on its own, before retrieval sees it. It is never given to the
model that writes the answer. That model may state only what a retrieved
passage supports, because every claim it makes is checked against the passage
it cites - and prose from an earlier turn cannot be checked against any paper,
so letting it back in would let a claim enter an answer without ever having
come from one. History decides what is looked up; it never decides what may be
said.

The resolved question is shown above the answer, because a reader who
disagrees with how their follow-up was interpreted needs to see the
interpretation. If the model is unreachable or replies with nothing usable,
the original question is used unchanged: resolving a follow-up is an
improvement to retrieval, and an improvement must not be able to stop a
question being asked.

## Orchestrating specialist roles

`report --agents` is the multi-agent version of the review, and it was
designed from what the 2026 literature agrees on rather than from enthusiasm.
Published orchestrator-workers systems and the LangGraph benchmark both find that an
orchestrator delegating to parallel specialists beats a single agent on
breadth-first work, at a cost of 3 to 15 times the tokens; one such system also
keeps citation checking in a separate agent because one agent both writing
and checking plays "a game of telephone". Magentic-One's orchestrator keeps a
ledger of progress and retries on error, and OrchestraBench (arXiv:2608.05263,
held in this corpus) shows why the ledger matters: blind retry reproduces
latent faults, and containment needs detection *with attribution*. The
MAST taxonomy finds most multi-agent failures are design failures - weak
specification, misaligned roles, missing verification - not model failures.
And the guidance on when to split at all is to divide by *context*, not by
problem type: facets that share context belong to one agent.

That last rule explains a result in this file. Splitting one question into
sub-questions lost on its own compound cases, because the parts share the
question's context and divide its retrieval budget. A review's *sections*
share a topic and nothing else, which is exactly the split the literature
says parallel workers are for.

So the roles here differ in kind, not in subject, and the supervisor is code:

| role | who | what it does |
|---|---|---|
| planner | model | one call, section questions as flat lines |
| router | code | the section's shape picks a ladder of strategies |
| retrievers | code, and the tool-choosing agent as the last rung | `hybrid` (graph gate inside), `compare`, `titles`, `agent` |
| writer | model | the existing verified-citation path, unchanged |
| verifier | code | the quote check, plus the describe-then-compare subject check |
| critic | code, model-assisted | uncited claims by counting; uncovered subjects and conflicts proposed by a model and then checked |
| discloser | code | what the section rests on: passages, papers, subject, named-after paper |
| supervisor | code | ledger, retries, containment, parallel sections, assembly |

The verifier never asks a model whether a section is good: asked that, every
model tested here says yes. A section is accepted on numbers - at least one
citation survived verification and the passages are on-subject - and
otherwise retried with the *next* strategy on its ladder, so a retry is never
the same retrieval twice. The best attempt is kept, not the last, since an
escalation can retrieve worse than what it replaced. The supervisor forwards
each verified answer untouched, because the LangGraph benchmark's dominant
failure mode was the supervisor's own paraphrasing. Every attempt goes into
a ledger that the review prints as its final section, so a reader can see
which strategy each section took, what verified, and what was retried. The
agent rung records the tools it actually called (`via search_chunks(...) ->
get_paper(...)`), because it is the only rung whose behaviour varies question
to question, and a ledger that says a rung ran without saying what it did is
the attribution OrchestraBench found containment needs.

A rung that *errors* - an unreachable model, a corpus fault - is recorded as a
failed attempt and the ladder carries on to the next strategy. One dead
section no longer discards a review that took ten minutes to write, and the
outcome distinguishes the two cases that matter: a section nothing could
support is a **gap**, which is a claim about the corpus, while a section whose
strategies fell over is a **failure**, which is a claim about the machinery
and says nothing about the corpus at all.

Every section also prints what it rests on - passages, papers, what those
passages are about - by the same rule the single-answer path follows: always,
not on a threshold, because the case a threshold would catch is the one no
threshold sees.

The plain `report` is now this same loop with a single rung and no
escalation, rather than a second copy of it. The two had already drifted once:
the orchestrated path gained containment and disclosure and the deterministic
one did not. `--use-compare` is refused with `--agents` rather than silently
ignored, since with a shape router there is no section-independent strategy to
force.

Sections run in parallel with `--workers N`, each thread on its own SQLite
connection. **Measured, it buys nothing here:** a four-section review took
424s with one worker and 423s with three. Ollama serves one request to a model
at a time unless `OLLAMA_NUM_PARALLEL` is raised, so the workers queue behind
the same server and the wall clock is set by generation either way. The
orchestration costs nothing measurable, and it earns nothing either.

Both `ask` and `report` now close with what the run cost - model calls, tokens
in and out, and seconds spent inside the model, which Ollama returns on every
call and this project was discarding. The orchestrator's entire case is a
cost/benefit argument, and it was being made without the cost.

That is worth stating plainly because the literature's headline claim for
multi-agent systems is throughput through parallelism. On a single local model
server that claim does not transfer: the parallelism has nowhere to go. What
does transfer here is the *other* half - routing by shape, retrying with a
different strategy, and refusing to accept a section on a model's say-so -
and those are measured in the tables above. Raise `OLLAMA_NUM_PARALLEL`, or
point the writer at a server that batches, and `--workers` becomes worth
having; until then it is off by default at one worker.

**Measured, three more ways.** First the router: `hybrid+routed` - both
shape gates on, which is every section's first rung - scores exactly
`hybrid+graph` on the single-subject table (0.81 / 0.95 / 0.97 / mrr 0.869)
and exactly `hybrid+compare` on the compound table (cov@10 0.73, all@10
0.55). The gates compose without touching each other's questions; that was
predicted, and is now a row.

Second, the orchestrated review against the deterministic one on a topic the
corpus covers well ("reinforcement learning for reasoning in large language
models", four sections each): `report` took 343 s; `--agents` took 411 s
with one worker and 395 s with two, and **every section was accepted on its
first rung** in both runs. Zero escalations, so on this topic the
orchestrator bought a ledger and nothing else, at about 20% more time - the
same conclusion as the timing above, reached again. (The planner is a model,
so the sections differ between runs; these are not controlled pairs.)

Third, the measurement the ladder exists for: a topic the corpus covers
unevenly ("comparing MOND and dark matter in galaxy clusters, and what Planck
2018 constrains"). The deterministic review took 360 s and produced a fluent,
fully cited section on Planck constraints that never mentions MOND. The
passages are about Planck, so the subject check passed it at 0.77 - it
answers a question the section did not ask. The orchestrated review took
841 s. Its Planck-constrains-MOND section was retrieved three ways -
`hybrid`, `titles`, then the tool-choosing `agent` - and the writer declined
to cite anything all three times, so the verifier read zero verified
citations, the ladder ran out, and the section says so:

> The passages do not contain information regarding the extent to which
> Planck 2018 cosmological constraints limit the viability of MOND when
> applied to cluster dynamics.

That reads like the `no_answer` case - subject covered, question unanswered -
finally being caught. **Measured against the answer key, it is not.** Every
one of the four labelled `no_answer` questions was *accepted* by the ladder:

| class | gap declared |
|---|---|
| `no_answer` | **0 / 4** |
| `no_coverage` | 6 / 9 |
| answerable | **0 / 25** |

The four are accepted because the corpus really does hold passages about
tensors and about cosmology, the writer really can quote them, and the quotes
really do verify. Verification asks whether the words are there, not whether
they answer, so a covered-but-unanswered question sails through exactly as it
always did.

What the verdict does detect is a *coverage* gap, at 6 of 9 with no false
declaration in 25 answerable questions. That precision is real and the recall
is not new: all six are cases the subject check already catches, and the three
it misses - Venus, cuprates, solar panels - are a superset of the subject
check's two misses. As a detector it is a strict subset of one that costs a
single call instead of three retrievals and three generations.

So the Planck-and-MOND section above is a narrower thing than it appears: no
passage supported that specific intersection, which is a local absence rather
than the `no_answer` class. Worth having, because a review that says "the
corpus does not answer this" beats one that answers something else - but it is
a gap declared about one claim, not a solution to the open problem.

So `--agents` stays off by default and is worth turning on when the topic is
one the corpus might not fully cover - which is when the ledger earns its
appendix. The keyword router is the known weak point: OrchestraBench found
flag routers score zero on adversarially phrased tasks, and the tool-choosing
agent on the last rung is the intent-reasoning fallback for exactly that.

## Saying what an answer rests on

Seven approaches have now tried to *detect* the covered-but-unanswered case,
and all seven measured 0 on the labelled cases. The reason does not change:
the passages really are on the subject and the quotes really do verify, so
nothing in retrieval or verification can separate a question that was answered
from one that was not.

So `ask` stops trying to decide, and reports instead. Every answer carries a
short, deterministic account of what it was written from - how many passages,
in how many papers, what those passages are about, and, for a question asking
what something *is*, the paper the subject is named after when the corpus holds
one:

```
The passages provided do not define what a tensor is.
  Answered from 8 passages in 8 papers.
  Those passages are about: Tensor properties and their architectural implications.
```

The subject line used to appear only when similarity fell below 0.65. It is
now always shown, because the case it would catch is exactly the case no
threshold sees.

One line was built, measured and removed: the negative of the named-after
fact - "no paper here is named after X". It fired on all 4 `no_answer` cases,
and on **13 of the 19** definition-shaped answerable ones. True every time and
useless: a line that fires on two thirds of good answers is one readers learn
to skip, which is the same reason the subject threshold was not raised to
0.79. Its absence is left to speak for itself.

## What retrieval measures

Over 87 cases - 63 answerable, 9 coverage gaps, 4 covered but unanswered,
11 compound (scored in their own table below) - across 7,921 papers and
835,743 vectors:

| mode | r@1 | r@5 | r@10 | mrr |
|---|---|---|---|---|
| keyword-all | 0.38 | 0.40 | 0.40 | 0.389 |
| keyword-any | 0.62 | 0.81 | 0.86 | 0.705 |
| vector | 0.76 | 0.89 | 0.92 | 0.818 |
| hybrid | **0.81** | 0.94 | 0.95 | 0.866 |
| **hybrid+graph** (default) | **0.81** | **0.95** | **0.97** | **0.869** |

`hybrid+graph` adds a citation-graph leg that fires only on definition-shaped
questions ("what is X", "define X"), and only when a title-matched paper is
cited by at least two held papers - the corpus itself testifying that this is
the paper the subject is named after. Across the 63 answerable cases it
changes exactly three: "what is the Adam optimizer" goes from found-nothing to
rank 3, BERT's definitional question from rank 2 to 1, and "what is artificial
intelligence" from rank 1 to 3. Every other case is untouched, which is what
separates it from the always-on title leg below.

Eleven further cases are **compound**: "compare the post-training pipelines of
Qwen3 and Llama 3" is not answered by ten excellent passages about Qwen3.
These are scored by coverage - the fraction of a question's required papers
found - and completion, never by first hit, and they are kept out of the table
above so its numbers stay comparable across measurements.

| mode | cov@5 | all@5 | cov@10 | all@10 |
|---|---|---|---|---|
| hybrid | 0.41 | 0.09 | 0.59 | 0.36 |
| hybrid+graph | 0.41 | 0.09 | 0.59 | 0.36 |
| hybrid+plan | 0.36 | 0.09 | 0.59 | 0.36 |
| hybrid+titles | **0.64** | **0.45** | 0.68 | 0.45 |
| **hybrid+compare** | 0.59 | 0.36 | **0.73** | **0.55** |

Three findings. The default mode fully covers a multi-paper question 36% of
the time - a failure first-hit metrics cannot see, since ten perfect passages
about PPO score 1.0 on "how does GRPO differ from PPO". The mode that rescues
most of it is the title leg that single-subject questions measured off. And
gating that leg to comparison-shaped questions (`--use-compare`) beats both
parents at depth while provably not touching the single-subject table -
though it stays off by default, because the key holds no comparison-shaped
single-paper questions to measure its cost on. Details, and decomposition's
loss on its home turf, under *What was tried and rejected*.

An earlier version of this file reported mrr 0.940. That was a 42-case set,
mostly paper titles, over a corpus chunked from PDFs. Scoring those same 42
cases on today's corpus gives **mrr 0.917, r@5 0.98** - so reading papers from
HTML, carrying tables as their own passages and growing the corpus by 370
papers left retrieval where it was. The headline number is lower because the
answer key is harder: the 21 cases added since score mrr 0.764, and eight of
them are definitional questions of the kind this corpus is worst at.

## What a held-out set measures

Everything above was measured on an answer key written alongside the system.
That is a real weakness and not a small one: a question that retrieved badly
could be reworded until it did not, and no amount of care rules that out from
the inside. So a second key was built to be uncontaminated -
`evals/heldout.jsonl`, 93 cases authored from each paper's title and abstract
only, with no retrieval run against any question, committed before it was
scored, and reported at whatever it scored. Papers were picked by a fixed rule
before any question existed, excluding every paper already in the development
key and every subject probed by hand while building the system. 88 answerable
cases over 88 distinct papers in 29 categories.

| mode | r@1 | r@5 | r@10 | mrr | mrr on the development key |
|---|---|---|---|---|---|
| keyword-any | 0.62 | 0.76 | 0.84 | 0.697 | 0.705 |
| hybrid+titles | 0.65 | 0.85 | 0.92 | 0.737 | 0.822 |
| hybrid+compare | 0.68 | 0.85 | 0.92 | 0.752 | 0.866 |
| vector | 0.69 | 0.83 | 0.90 | 0.756 | 0.818 |
| hybrid | 0.68 | 0.85 | 0.92 | 0.758 | 0.866 |
| hybrid+graph *(default)* | 0.68 | 0.85 | 0.92 | 0.758 | 0.869 |

**The development numbers are inflated by about an eighth.** The shipped
default scores mrr 0.869 there and **0.758** here, on questions of the same
kind over the same corpus. The difference is not the retrieval; it is that one
set of questions was written while watching the retrieval and the other was
not. Every figure in the section above should be read with that discount.

**And none of these modes can be told apart.** Reciprocal rank was recorded
per case and the modes compared pairwise, with a 10,000-round paired bootstrap
over the 88 cases:

| comparison | difference | 95% interval | |
|---|---|---|---|
| graph − hybrid | +0.000 | [+0.000, +0.000] | the gate never fires here |
| graph − compare | +0.006 | [−0.014, +0.028] | not distinguishable |
| graph − titles | +0.021 | [−0.034, +0.075] | not distinguishable |

Every interval straddles zero. At this sample size a difference of a few
hundredths of mrr is indistinguishable from noise, and the differences this
file has been reporting are smaller than that: `hybrid+graph` was made the
default over `hybrid` on **0.003** mrr across 63 cases, which the interval
above says is unmeasurable. That is a fair criticism of the reasoning, and it
stands whatever the gate's other merits.

The gates are still defensible - just not on aggregate mrr. The graph gate
exists to fix definitional questions, and it does: "what is the Adam
optimizer" goes from nothing in the top ten to rank 3, which is visible in the
case and invisible in a mean over 63. An aggregate cannot see three cases
move. The honest statement is that a gate targeted at a question shape should
be judged on that shape, and that the corpus-wide number was never evidence
either way.

Two things this set cannot decide. It is **silent on the graph gate** by
construction - it contains no definition-shaped questions, so the leg is inert
and the difference is exactly 0.000, which at least confirms the scoping does
what it claims. And it cannot settle whether `hybrid+titles` should be the
default: the first 36 cases had it winning by 0.052, the next 52 had it losing
by 0.071, and the pooled interval contains zero. That swing between batches is
itself the most useful number here, because it is far larger than any
difference between modes. Deciding this needs several hundred cases, not
another fifty.

So nothing was changed. `hybrid+graph` remains the default because it is the
incumbent and no evidence distinguishes the alternatives, which is a weaker
reason than this file used to give and a truer one.

### What the misses actually are

A mean says how often the right paper is found and never what happened the
rest of the time. `evals/failure_taxonomy.py` labels every held-out case from
three mechanical signals - where the expected paper ranked, how much of the
question's vocabulary appears anywhere in that paper, and how similar the
paper that took first place is to the expected one by abstract vector:

| | share | |
|---|---|---|
| ranked first | 60 | 68% |
| ranked 2-5 | 15 | 17% |
| ranked 6-10 | 6 | 7% |
| missed, a near neighbour took first place | 5 | 6% |
| missed, unexplained | 2 | 2% |
| **missed because the question's words are not in the paper** | **0** | **0%** |

That last row was the expected answer and it is empty. The guess going in was
that authoring questions without the title's phrasing had produced questions
too abstract to match - and it is wrong. Vocabulary coverage across every
failure runs from 0.67 to 1.00: the words the question uses are in the paper
already. Nothing here is a lexical gap.

What the failures are instead is competition. The paper that took first place
is topically adjacent in almost every case - abstract similarity 0.62 to 0.95
to the paper the question was written from - so the corpus holds several
papers on the subject and the ranking cannot separate near-equals. For "how do
graph-based item embeddings compare with identifier-based ones", several
papers here are about item embeddings, and only one of them is the one the
question was authored against.

Which exposes a limit of the metric rather than of the retrieval. Each
held-out question names exactly one acceptable paper, because each was written
from one abstract, so a genuinely correct answer drawn from a neighbouring
paper is scored as a miss. The 0.758 is therefore a floor on how often the
system retrieves something that answers the question, and the gap between that
floor and the truth is exactly the 6% of cases where a near neighbour won.
Closing it needs a judgement about whether an answer is supported, not a
comparison of arXiv ids - the answer-level evaluation this project does not
have.

The two unexplained misses are worth naming rather than rounding away:
a cleaning-robot path-planning question and one about the analytic
continuation of a power series, where the top result is unrelated by abstract
similarity (0.67 and 0.72) and no signal here accounts for it.

A twelve-question tripwire over the shipped sample corpus runs in CI on every
push and fails the build below mrr 0.90 (`evals/gate.jsonl`). It is coarse by
construction - twelve plainly distinct papers out of forty - and it exists to
catch retrieval breaking, not retrieval slipping. A measurement nothing
enforces is a claim rather than a guarantee.

## What was tried and rejected

Kept here because the negative results are the useful part.

**Asking a model whether passages are relevant does not work.** Four variants
were measured against questions the corpus cannot answer. Every one failed the
same way: the passages were retrieved *because* they matched, and the model
reads that as evidence they are relevant.

| approach | declines correctly | keeps real answers |
|---|---|---|
| no check at all | 2/6 | 14/14 |
| verdict inside the answering call | 1/6 | 13/14 |
| separate judging call, 4B model | 2/6 | 14/14 |
| separate judging call, 13.8GB model | 0/6 | 14/14 |
| require two supporting papers | 4/6 | 8/14 |

Corroboration also fails as a *warning*, not just as a gate: it fires on 5 of 14
good answers while catching 2 of 6 bad ones, and catches nothing the subject
check misses. Combining the two is strictly worse than the subject check alone.

The bigger model was worse, so this is not a matter of model size. What does
work is not asking the question: ask only what the passages discuss, and compare
that to the question arithmetically.

**Splitting the label showed the subject check was being scored against a
question it cannot answer.** Once `expect_none` was separated into "no paper
is about this" and "the subject is covered and the question still has no
answer", the same subject check that had scored 3/6 against the mixed label
turned out to work on one kind and be blind to the other:

| kind | cases | caught | similarity range |
|---|---|---|---|
| `no_coverage` | 9 | **7** | 0.431 - 0.782 |
| `no_answer` | 4 | **0** | 0.769 - 0.853 |

Raising the threshold to close the gap is not worth it, and that is measured
rather than assumed. Against the 63 answerable cases:

| threshold | gaps caught | false warnings |
|---|---|---|
| **0.65** (current) | **7/9** | **1/63 (2%)** |
| 0.70 | 7/9 | 5/63 (8%) |
| 0.75 | 8/9 | 16/63 (25%) |
| 0.79 | 9/9 | 27/63 (43%) |

Catching the last two costs a warning on 43% of correct answers, and a warning
that fires that often is one people learn to ignore - which would cost the
seven it already catches. 0.70 buys nothing at four times the false-warning
rate. The default stays at 0.65 because it was measured, not because it was
chosen first.

The distributions barely overlap. For six attempts the detector had been asked
to find something no retrieval signal can see, half the time, and the mixed
label made a partial success look like a flat failure. The two misses are the
deliberately hard cases - "surface temperature of Venus" and
"superconductivity in cuprates", off-domain subjects that sound in-domain -
while the plainly off-domain gaps warn at 0.43 to 0.48.

**Neither abstract-level nor lexical matching detects an unanswerable
question.** The reasoning was that a passage says what it mentions while an
abstract says what a paper is *about*, so a paper that quotes sample text about
nuclear forces on page 27 would be excluded by its abstract. Measured over
7,551 abstracts and the 22 natural-language eval questions, it separates no
better than passages do.

| signal | answerable | unanswerable | separated |
|---|---|---|---|
| passage similarity | 0.666-0.799 | 0.628-0.798 | no |
| abstract similarity | 0.598-0.810 | 0.603-0.785 | no |
| abstracts containing every question word | 0-235 | 0-177 | no |

The measurement did explain why every attempt fails. The `expect_none` cases
are two different things wearing one label. "What is the farthest star from
the earth" has no paper on the topic at all - zero abstracts contain its words.
"What is a tensor" has **177**: the corpus is saturated with tensors and
defines none of them. Only the first kind is a retrieval problem. The second is
a question about whether passages *answer*, which no retrieval signal can see,
and it sits next to "what is entropy?" - 235 matching papers, labelled
answerable - with nothing measurable between them.

So the next work here is the answer key, not another detector: the two kinds
need separating before a detector for either can be honestly scored.

**Embedding a passage with its title and section is still unmeasured.** An A/B
over 21,414 passages returned mrr 0.982 for both arms, identical to three
decimals. The inputs did differ; the subset of 259 papers was simply too easy,
and r@5 saturated at 1.000 with no headroom for a difference to appear in. It
is recorded here as a failed experiment rather than a negative result, and
`with_context` stays off because it is unmeasured. Redoing it needs a
distractor set large enough that r@5 sits well below 1.0.

**Everything that widens the candidate pool trades top precision for deep
recall.** Four techniques, measured across the same 63 answerable cases:

| mode | r@1 | r@5 | r@10 | mrr |
|---|---|---|---|---|
| **hybrid** | **0.81** | **0.94** | 0.95 | **0.866** |
| hybrid + titles | 0.75 | 0.94 | 0.97 | 0.822 |
| hybrid + 3 rewrites | 0.75 | 0.90 | **0.98** | 0.826 |
| hybrid + HyDE | 0.70 | 0.92 | 0.97 | 0.799 |
| hybrid + decomposition | 0.76 | 0.87 | 0.92 | 0.816 |

The first three raise recall@10 and lower recall@1. That is not four coincidences,
it is what widening does to a ranking that is already right most of the time:
at r@1 0.81 the passage that answers the question is usually first already, and
new candidates can only displace it. The gain is confined to the 5% of
questions where hybrid finds nothing in ten, and it is paid for out of the 81%
where it was already first.

Decomposition is the exception, and it is worse on every measure including
recall@10. It does not widen the pool, it divides it: each sub-question gets a
fraction of the result slots, and a fragment of a question retrieves less
precisely than the whole one. An earlier version of this paragraph noted the
caveat that nothing in the key was the compound question `ask --plan` was
built for, and that measuring that needed cases whose answer genuinely spans
two papers plus a coverage metric. Both now exist - eleven compound cases,
scored by coverage and completion - and **decomposition lost on its home
turf**: cov@5 0.36 against plain hybrid's 0.41, identical at depth (all@10
0.36 both). Splitting still divides the slots, and the sub-questions a model
writes do not name the papers the way the question's own words already do.

What wins the compound cases instead is the title leg (cov@5 0.64, all@5 and
all@10 0.45, against hybrid's 0.09 and 0.36): "compare the GPT-4 and Llama 3
technical reports" names its papers almost by title, and passage search finds
neither while the title index finds both at ranks 1-2. So the same technique
has now been measured three ways - harmful always-on, decisive behind the
definitional gate, and best-in-class on compound questions - and the pattern
is that question shape decides the tool.

That pattern made a prediction, and the prediction was measured true. A titles
leg gated to comparison-shaped questions ("compare", "differ from", "versus")
should keep the title leg's compound wins, skip its one compound loss - the
DeepSeek-R1 case, where "build on" is not comparison vocabulary and the title
leg had displaced a right answer - and be provably unable to touch the
single-subject table, which contains no comparison-shaped question. Measured
as `hybrid+compare`: cov@10 0.73 and all@10 0.55, better than either parent,
with the single-subject table byte-identical to hybrid. It ships behind
`--use-compare`, off by default, for one honest reason: the eval key holds no
comparison-shaped questions with a *single*-paper answer, so the gate's cost
on that shape - a "how does X differ from vanilla Y" whose title match is
wrong - has no number yet. Authoring those cases is the recorded next step;
until then the leg's benefit is measured and its risk is not, and unmeasured
risk ships off.

Two of the eleven cases also turned out not to be compound on this corpus at
all: DeepSeek-R1's appendix compares GRPO and PPO in one place, and RoBERTa
restates BERT's recipe as its very subject. A corpus that holds surveys and
technical reports has already done the decomposition for many comparison
questions, which is both a caveat on these numbers and a finding in its own
right. The labels were left as the original papers rather than retrofitted to
what retrieval happened to find.

Which matters depends on what reads the results. An answer is written from the
first few passages, so `ask` wants r@1 and keeps plain hybrid. A person
scanning ten results might prefer the wider net, which is why each remains
available behind a flag rather than being deleted.

**Adding a title-matching leg makes retrieval worse, for the same reason
reranking does.** It was built to answer the definitional questions passage
search cannot: "what is the Adam optimizer" finds nothing from the Adam paper
in ten results, and a title leg puts it at rank 4. Measured across all 63
answerable cases it costs more than it earns:

| mode | r@1 | r@5 | r@10 | mrr |
|---|---|---|---|---|
| hybrid | **0.81** | 0.94 | 0.95 | **0.866** |
| hybrid+titles | 0.75 | 0.94 | **0.97** | 0.822 |

It rescues a few deep misses and displaces better passage matches at the top,
which is a bad trade when the answer is written from the first few passages.
Available behind `--use-titles`, off by default. An earlier version of this
paragraph guessed that "a version that applies titles only to questions that
look definitional might get both" - that version is now measured, and it does.
`hybrid+graph` gates the title leg twice: the question must be shaped like a
definition, and the matched paper must be cited by held papers, so "why does
PPO clip its objective" never sees the leg that would hand it CLIP. It is the
only widening technique measured here that pays no recall@1 for its deep
recall (0.81 / 0.95 / 0.97 / mrr 0.869), because it widens only the questions
the baseline was already losing - and it is on by default for exactly that
reason. The exception proves the widening rule rather than breaking it: the
cost of widening is paid where the ranking was already right, so a widener
that can tell those questions apart keeps the gain without the bill.

**Cross-encoder reranking makes retrieval worse here**, on all three local
models tried (mrr 0.940 → 0.752 / 0.701 / 0.741, measured against the 42-case
set of the time). Reranking pays when the right passage sits at rank 8; at
r@1 0.86 there is nothing to promote and much to
demote. Available behind `--rerank` for when that changes.

## What each run did

`stats` says what the corpus holds. Until recently nothing said how it got
there - an ingest that runs unattended could fail, half-finish, or embed with
a different model than yesterday, and the only evidence was whatever scrolled
past in a terminal. Every command that touches the corpus now writes a row:
when it started and finished, whether it succeeded, what it was asked to do,
what it produced, and the corpus counts before and after.

```
uv run verbatim runs

  when                 command              took  what it did
  2026-09-09T17:28:53  ask          ok       3.2s  passages=8, papers=4, citations=1, rejected=1
  2026-09-09T17:28:52  index        ok       0.2s  added=0, removed=0, vectors=3139
  2026-09-09T14:02:11  embed        FAIL    12.4s  RuntimeError: Cannot reach Ollama
```

Failures are recorded, which is the point: a ledger that holds only successes
is a ledger of the wrong half. `runs --json` emits the same rows for a driver
to read, and exits non-zero if any of them failed, so `ingest.sh` ends by
writing a summary and checking it rather than by printing DONE and hoping.

Recording is best-effort by construction. A command's job is the corpus, not
the bookkeeping, so a failure to write a row is swallowed rather than allowed
to fail work that otherwise succeeded.

`GET /api/health` reads the same ledger:

```json
{
  "ok": false,
  "corpus": {"papers": 7921, "passages": 835743, "vectors": 835743,
             "pending": {"downloads": 0, "chunking": 0, "embedding": 0}},
  "last_run": {"embed": {"at": "...", "ok": false, "error": "RuntimeError: ..."}},
  "failing": ["embed"],
  "answers": {"runs": 41, "citations_verified": 118, "citations_rejected": 39,
              "rejection_rate": 0.248}
}
```

It is deliberately more than a 200. An empty corpus and a corpus whose last
ingest failed both serve pages perfectly well, and both are what a check is
actually asking about. The citation rejection rate comes from the ledger
rather than being recomputed, so it describes what happened rather than what
would happen now - and it is a property of the model, not a fault: the check
exists because models paraphrase.

## Serving it over HTTP

`verbatim serve` runs a small FastAPI service with a search page at `/`,
a JSON API under `/api`, and generated OpenAPI docs at `/docs`.

```
verbatim serve --port 8000
```

| route | returns |
|---|---|
| `GET /api/search?q=&mode=&limit=&arxiv_id=` | passages, each with its locator, licence and source URL |
| `GET /api/ask?q=&arxiv_id=&history=` | an answer as server-sent events, drafted live and then verified |
| `GET /api/report?topic=&sections=&passages=&agents=` | a sectioned review as server-sent events, each section streamed as it lands |
| `GET /api/papers/{arxiv_id}` | one paper's metadata, or 404 |
| `GET /api/stats` | the same pipeline counts `stats` prints |
| `GET /api/health` | corpus state, the last run of each command, and the citation rejection rate |

The ask route streams because a local model takes twenty seconds to write and
a silent page for twenty seconds reads as broken. The event order is the
honesty order: `passages` first (what was retrieved, before a word is
generated), `delta` fragments as the model drafts, then an `answer` event
whose text has been through the same citation verification as everywhere
else - the draft is what the model said, the answer is what survived checking
- and finally a `subject` event carrying the abstention warning when it
fires. Streaming changes when text is shown, and nothing about what is
accepted. The page's ask box renders all of this: live draft, then verified
text, references, rejected-citation count, and the subject warning.

The review route streams for the same reason the answer route does, and its
events are the same vocabulary extended by one: `working` when a section
starts, `section` when it has been written and checked - carrying its
citations, its evidence account, and the ladder's attempts including which
tools the agent rung called - then `report` with the assembled markdown and
the ledger, then `done` with what the run cost. The orchestration runs on a
thread of its own with its own corpus connection, because the work is a
blocking loop and the events have to leave the generator while it is still
going. `agents=false` runs the single-rung version. Measured live over HTTP:
a two-section review arrived as `working, section, working, section, report,
done` - sections landing as they were written rather than batched at the end.

The page has a review panel over it, and sections appear one at a time with
their strategy trace under each, which is the orchestrator's argument made
visible: you can watch a section be retried with a different strategy.

**`report` is deliberately not an MCP tool.** A review is minutes of
generation, MCP calls are request/response with client timeouts, and a tool
that blocks for ten minutes and cannot stream its progress would be a worse
version of what the CLI and this route already do well.

It binds to loopback unless `--host` says otherwise: the corpus is 26 GB of
other people's papers under mixed licences, and putting that on a network is a
decision rather than a default. The page is one file with no bundler and no
external requests, since the pipeline runs offline once the corpus is built and
a page that reached a CDN would quietly undo that.

## Using the corpus from an MCP client

`verbatim mcp` serves the corpus over the Model Context Protocol on stdio,
so a desktop assistant, a coding agent or an editor can search it directly. It offers
the same five tools the built-in agent uses - `search_chunks`, `get_paper`,
`list_papers`, `get_citations`, `read_pages` - because both are front ends onto
one set of operations rather than two implementations of it. All five are
marked read-only.

Results come back as JSON rather than prose, so the arXiv id, page or section
and chunk id survive intact and the client can cite what it read.

```json
{
  "mcpServers": {
    "verbatim": {
      "command": "uv",
      "args": ["run", "verbatim", "mcp"],
      "cwd": "/path/to/verbatim"
    }
  }
}
```

The corpus is opened inside a single worker thread and every call is dispatched
to it: SQLite connections belong to the thread that made them, and the tools
block on SQLite and ONNX, so they must not run on the event loop serving the
client.

## Known limitations

**It will answer a question the corpus cannot answer.** This is the largest
open defect. Retrieval always returns its best matches, and those matches are
passed to the model whether or not they bear on the question; asked about a
subject the corpus does not cover, the system produces a fluent answer whose
citations are all individually genuine. Every quote is verified against the
stored passage, so nothing is fabricated — but a verified quote proves the
words exist, not that they answer the question. The subject check catches
roughly half of these.

"Unanswerable" was one label over two different problems, and separating them
changed the picture. A question with no paper on its subject is visible to
retrieval; a question whose subject the corpus covers heavily and answers
nowhere is not - "what is a tensor" matches 177 abstracts here and sits
indistinguishably beside "what is entropy", which matches 235 and is
answerable.

Measured against the split label, the subject check catches **7 of 9** coverage
gaps and **0 of 4** covered-but-unanswered questions. So the defect is now two
defects with different prospects: the first is largely handled and its
remaining misses are subjects that sound in-domain, and the second is not a
retrieval problem at all. See *What was tried and rejected*.

**Scanned PDFs yield nothing.** Text extraction assumes a text layer; OCR is
not implemented. Affected papers chunk to little or nothing rather than
failing loudly.

**The corpus is a slice of recent work.** It was built from the newest papers
in each category, so it began holding very little of the literature those
papers rest on. After `grow`, 12,882 of 80,469 resolvable references now point
at a paper the corpus holds and 43% of papers have at least one such link. `grow` closes this along the citation edges — GPT-4, Adam, BERT, PPO,
ViT and Planck 2018 were all cited dozens of times and absent until it ran —
but a corpus assembled this way starts without foundations, and definitional
questions are the first thing to suffer. Adding the defining paper turned out
to be necessary and not sufficient: 3,152 papers here mention Adam, so the
paper that defines it was drowned roughly 60:1 by hyperparameter sections
until the citation-graph leg learned to ask which title-matched paper the
corpus itself cites. That handles subjects a held paper is named after;
"what is a tensor" still has no defining paper to anchor to, and remains
unanswerable for reasons no ranking can fix.

**Retrieval quality is measured on two sets, and they disagree.** The
87-case development key was written alongside the system - 63 answerable by
one paper, 11 compound, 13 unanswerable - and twenty of the answerable cases
are an auto-seeded floor of paper titles, which retrieve their own paper for
reasons a real question does not. The 93-case held-out key was authored blind
and frozen before scoring, and the default scores mrr 0.758 on it against
0.869 on the development key. Neither set transfers to another corpus without
re-measuring, and a paired bootstrap over the held-out cases cannot
distinguish any of the retrieval modes from any other. See *What a held-out
set measures*.

**Vector search is brute force.** sqlite-vec scans every vector, which is
measured at about 94 ms for a k=50 query over 455,000 vectors and is fine at
this scale. A corpus several times larger wants a real index behind the
`VectorIndex` protocol rather than a faster machine.

**Some papers have no PDF at arXiv.** 32 of the papers here return 404 for
their versioned PDF, most of them also unversioned. They are recorded as
settled rather than retried, and `stats` reports them apart from real
outstanding work. No unversioned fallback is attempted: fetching whatever the
current version happens to be, while the citation claims v2, would break the
version pinning the rest of the project depends on.

**Licence classification is only as good as arXiv's metadata**, and the
`unknown` class is large. `unknown` means unknown, not permissive.

## Principles

- Start with a small, inspectable corpus.
- Preserve source URLs, arXiv IDs, and licence metadata where available.
- Cite the paper *version* that was read; v1 and v2 can disagree.
- Require citations for generated claims.
- Treat model training as a separate, provenance-aware track.

## Licence caveats

arXiv metadata and arXiv full texts are licensed differently. The default
submission licence lets arXiv distribute a paper; it does not grant
redistribution rights to third parties. Local reading and indexing is one
thing, redistributing PDFs or chunk text is another, and training on them is a
third that needs an explicit written policy first.

**The query API does not return licences at all** — measured across 59 papers,
every one came back without one. Licence lives in the OAI-PMH record instead:

```bash
uv run verbatim licenses --limit 50
```

Licences are grouped by what they actually permit, because presence is not
permission:

| Class | Meaning |
|---|---|
| `open` | CC-BY / CC-BY-SA / CC0 — redistribution and training permitted, usually with attribution |
| `arxiv-nonexclusive` | arXiv's default: **arXiv** may distribute it, you may not |
| `unknown` | not looked up yet, or the paper predates the field |

Licence URLs are carried through to every chunk, and `--require-license`
restricts downloads to papers that report one — but note it accepts
`arxiv-nonexclusive` too, so it is not yet a reuse gate. Nothing enforces a
training policy.

PyMuPDF is AGPL-3.0. That is fine for local personal use and is a decision to
revisit before building any hosted product on top of it.

## Running it in a container

`docker compose run --rm verbatim search "how do transformers handle long context" --state sample/corpus.db`
works without installing Python, uv or an ONNX runtime, and
`docker compose up verbatim-web` serves the page on `127.0.0.1:8000`. Retrieval
needs no model, so the default profile starts nothing that downloads weights;
`--profile ollama` adds one for `ask` and `report`.

The corpus is a mounted volume rather than part of the image - it is 26 GB of
other people's papers - and the web port is bound to loopback for the same
reason the CLI's `serve` is. The image is a reproducible environment for the
tool, not a production deployment: this is a local-first assistant whose corpus
lives on one machine.

## Configuration

All tunables live in `.env`; copy the template and edit:

```bash
cp .env.example .env
```

`.env.example` documents every setting — storage paths, chunk size and overlap,
embedding model, retrieval limits, and the Ollama host and model. `.env` itself
is gitignored so local paths and hosts stay out of the repository. An exported
environment variable always overrides the file, so a one-off run needs no edit:

```bash
OLLAMA_MODEL=llama3.1 uv run verbatim ask "..."
```

## Development

```bash
uv run pytest
```

The ingestion client uses arXiv's public metadata API. Before growing the corpus,
review arXiv's API and bulk-access guidance.

## Licence

**AGPL-3.0-or-later.** Copyright (c) 2026 Rishab Gupta.

This is not a default choice. The layout-aware PDF parsing depends on
[PyMuPDF](https://pymupdf.readthedocs.io/), which its authors dual-license as
AGPL-3.0 or commercial. Copyleft is sticky: a work that imports AGPL code and
is then distributed must itself be AGPL, so a permissive licence here would
have been incorrect rather than merely generous.

What that means in practice:

- Read it, run it, fork it, learn from it, modify it privately — freely.
- Publish a modified version, or run it as a network service — then your source
  must be available to your users under the same terms.

Replacing PyMuPDF with a permissively licensed extractor (`pypdfium2`,
`pdfminer.six`) would allow MIT, at the cost of section paths, reference
detection and reading-order reconstruction. That trade was not worth making.
