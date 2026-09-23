# Using Verbatim in depth

The [README](../README.md) gets you running and lists every command. This is the
longer version: how each ingest stage behaves and why, the options behind
`ask`, the web and MCP front ends, the run ledger, generating training data and
the container. What was measured along the way, and what was tried and
rejected, is in [findings.md](findings.md) and [evaluation.md](evaluation.md).

- [Architecture](#architecture)
- [The sample corpus](#the-sample-corpus)
- [Daily corpus workflow](#daily-corpus-workflow)
- [Asking a follow-up](#asking-a-follow-up)
- [Escalating a single question](#escalating-a-single-question)
- [What each run did](#what-each-run-did)
- [Serving it over HTTP](#serving-it-over-http)
- [Using the corpus from an MCP client](#using-the-corpus-from-an-mcp-client)
- [Licence caveats](#licence-caveats)
- [Generating training data](#generating-training-data)
- [Running it in a container](#running-it-in-a-container)

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

## The sample corpus

Forty papers, 3,139 passages and their vectors, 2,230 bibliography entries,
about 23 MB. A share of the papers is picked as a citation neighborhood - a
hub and the papers that cite it - so the `graph` command demonstrates edges
on a clone rather than printing zeros. It carries no PDFs or renderings, only
the database, so `stats` correctly reports zero PDFs rather than claiming
files it does not have. Every passage still carries its own arXiv id,
section, offsets, checksum and licence, so citations are complete. `ask`
needs a model through Ollama; `search` and `graph` need nothing.

Answer quality varies with the model and the project says where — the ladder
recovers sections on a strong model that it can only report as gaps on a weak
one — but the citation verification is arithmetic and does not vary at all. A
quote either appears in the passage it names or it is thrown out, whichever
model wrote it.

Regenerate it from a full corpus with `verbatim sample --papers 40`.

## Daily corpus workflow

```bash
uv run verbatim sync --category physics.gen-ph --limit 5
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
re-requesting their metadata — and a `--full` run is **not resumable**: it
records nothing about how deep it reached, so one that fails part-way starts
from the newest paper again. Deepen a large category in stages of increasing
`--limit`, each of which commits what it fetched. Why there is no backfill
cursor is recorded in [decision 12](decisions/0012-one-sync-cursor-and-what-it-costs.md). arXiv's API is variable — the
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
refusing — see [What was tried and rejected](findings.md#what-was-tried-and-rejected). `--no-subject-check` skips
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
built for, it lost - see [What was tried and rejected](findings.md#what-was-tried-and-rejected).

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

Measured over 30 two-turn cases, this is the largest effect in the project:
retrieved as typed, a follow-up finds the right paper 30% of the time; resolved
against the conversation, 97% - as well as the question that preceded it. See
[findings.md](findings.md#what-a-conversation-costs), which also records why
fusing the two rankings is worse and why scoping to the previous turn's papers
cannot be judged on that key.

A question that already stands on its own is not sent for rewriting at all: it
costs a model call, and a model asked to rewrite a complete question can narrow
it. Only a question that points back - a pronoun, "that approach", or too few
words to carry a subject - consults the conversation.

The resolved question is shown above the answer, because a reader who
disagrees with how their follow-up was interpreted needs to see the
interpretation, and can correct it: "ask it differently" puts it in the box and
sends it exactly as written, with the conversation left out. If the model is
unreachable or replies with nothing usable,
the original question is used unchanged: resolving a follow-up is an
improvement to retrieval, and an improvement must not be able to stop a
question being asked.

## Escalating a single question

`ask` answers once and stops. When that answer has no verified citation, or
the passages turn out to be about something else, it says so and leaves it
there — it computes both of those facts and uses them only to print a warning.

`ask --deep` acts on them instead. It routes the question by shape and climbs
the same ladder a review section climbs, stopping as soon as a citation
verifies and the passages are on-subject:

```bash
uv run verbatim ask "how can reinforcement learning plan a route that covers an entire floor?" --deep
```

```
  open question, 2 attempt(s):
    hybrid   retry
    titles   accepted
```

Plain `ask` on that question declines with no citations and an off-subject
warning at 0.54. The ladder retries with the title leg and comes back with two
verified citations — and still says plainly that the passages do not fully
settle the question, because escalating changes what is retrieved, not what
may be claimed.

**It costs nothing extra when the first attempt succeeds**, because it *is* the
first attempt: the first rung is the same search plain `ask` runs. A
comparison-shaped question that the `compare` rung answers straight away makes
exactly the same calls either way. The cost is paid only where the plain path
was going to fail.

Measured over all 88 answerable held-out questions, two runs each — the
"workflow" arm of `evals/three_arm.py` is precisely this, one question through
the ladder:

| out of 88 | plain | `--deep` |
|---|---|---|
| answers with a verified citation | 78.5 | **81.5** |
| every claim carries a marker | 70.5 | **72.5** |
| citations rejected | 10.5 | **8.5** |

Ahead on every row, and **not distinguishable from noise**: the two paths give
the same answer on 81 of the 88 questions, and the 7 that differ split 6–1, a
sign test at p = 0.125. It ships behind a flag on that basis rather than as a
default. See [docs/evaluation.md](evaluation.md) for why the confidence
interval alone was misleading.

`--deep` subsumes `--tools` (the agent is its last rung), `--use-compare` (the
router chooses) and `--plan` (the ladder answers one question rather than
splitting it), and refuses each rather than ignoring it. It refuses
`--arxiv-id`, `--include-references` and `--no-graph` too, since each rung sets
its own retrieval, and `--tools` refuses the retrieval flags its routing model
decides for itself.

The web page has the same thing as a **dig deeper** checkbox beside the ask
box, off by default. Unchecked, nothing changes: the answer streams token by
token as before. Checked, `GET /api/ask?deep=true` streams the rungs instead —

```
event: working   {"strategy": "hybrid", "rung": 1, "rungs": 3}
event: working   {"strategy": "titles", "rung": 2, "rungs": 3}
event: answer    {...}
event: ladder    {"shape": "open", "outcome": "accepted", "attempts": [...]}
```

— because the ladder may write an answer, find that no citation survived, and
retry, and a draft shown to a reader and then withdrawn is worse than a slower
page. Deep mode also reports the subject rather than warning about it: plain
`ask` warns because it cannot act, while the ladder already retried on exactly
that signal, so complaining afterwards would report a failure that did not
happen.

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
so a desktop assistant, a coding agent or an editor can search it directly. It
offers
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

Every passage and citation reports its paper's licence URL, read from the
paper's catalogue record whenever the passage is loaded - so a licence looked
up after chunking still reaches it. `--require-license` restricts downloads to
papers that report one — but note it accepts
`arxiv-nonexclusive` too, so it is not yet a reuse gate. The one place the
policy *is* enforced in code is `distill`, below.

PyMuPDF is AGPL-3.0. That is fine for local personal use and is a decision to
revisit before building any hosted product on top of it.

## Generating training data

The long-term goal is a small model that answers the way this pipeline
answers — the flat `ANSWER`/`CITE` protocol, the abstention wording, the
citation discipline. `distill` writes the data to teach it:

```bash
uv run verbatim distill --examples 200
```

Each example is a `(question, retrieved passages, verified answer)` triple.
Questions are seeded from real passages, because a question the corpus cannot
answer costs a generation call to discover; retrieval then runs for real and
may or may not find the paper the question came from, which is the task the
student has to learn rather than a lookup.

Three rules come from [TRAINING-POLICY.md](../TRAINING-POLICY.md) and are enforced
rather than trusted:

| rule | how |
|---|---|
| only licence-eligible papers | a `PaperFilter` pushed into **both** retrieval legs, so an ineligible passage never reaches the prompt |
| only verified answers | one failed quote rejects the whole example |
| provenance per example | chunk ids, papers, licence URLs and classes, source checksums |

The licence gate is on the way in, not a filter on the way out. Discarding
examples afterwards would still leave the ineligible text in the prompt that
produced the kept ones. A final check runs over the citations anyway, and if
it ever fires the run reports a **fault** rather than an ordinary rejection —
it means the gate leaked, and the dataset should not be trained on.

Rejections are counted and printed, because the rejection mix is the only
signal that the generator is working: a run that keeps everything has stopped
checking.

```
  kept 300 of 328 candidate example(s)
    rejected, abstains while citing            14
    rejected, an unverified marker in the text  9
    rejected, a quote failed verification       5
```

Measured over a real 300-example run against the full corpus: 300 examples
drawn from 225 papers in 20 minutes, a 91.5% keep rate, zero faults.

**Reading the first run's output was worth more than building the generator.**
It found four defects, one of them in the answering path everything uses:
grouped chunk-id markers — `[a826c26dad1a785511b3, 180d5845e4ce3b24a60a]` —
survived into prose, because the renumbering rewrote a single id and left a
group alone. Nine of 300 examples carried one, and so did every `ask` answer
where the model grouped its markers.

The other three became the acceptance rules above. Two of them are now the top
rejection reasons, catching 23 of 28 rejections — every one of which had gone
silently into the first dataset:

- **`[unverified]` with nothing rejected.** Checking the rejected-citation list
  is necessary and not sufficient: a model that writes an inline marker for a
  passage it never cited leaves a dangling reference, rewritten to
  `[unverified]` with an empty list behind it.
- **Abstain-and-cite.** "The provided passages do not mention X" *carrying
  citations* teaches a student to refuse and cite at once. A real abstention
  has no citations and is rejected anyway; this is the half-abstention. The
  pattern is anchored on references to the supplied evidence, so "the authors
  do not provide a proof" — a finding about a paper — is kept.
- **Bibliographic trivia.** 83 of the first 300 asked about affiliations,
  emails or funding, because a paper's front matter is a passage like any
  other. The check is on the *question*, not the passage: front matter shares
  `chunk_index` 0 with 48% of the corpus and with most of the good seeds,
  since an HTML paper's first chunk usually holds the abstract too.

**What this dataset cannot teach: abstention.** Questions are seeded from
passages that are answerable by construction, and a declined answer is
rejected, so there are zero abstention examples in it. That is a real gap —
"unanswerable questions get answered" is this project's founding flaw, and
the policy names abstention wording as part of what is being taught. Teaching
it needs a generator for questions the corpus *cannot* answer, and a
measurement that the generated declines are real. Not built.

An `ATTRIBUTIONS` file is written beside the dataset, listing every source
paper with its authors, versioned id and licence. The policy requires it of
any released artifact; it is written whether or not anything is released,
because it costs nothing now and cannot be reconstructed if the dataset is
regenerated. Both land under `data/`, which is gitignored — the policy makes
the dataset private by default, since it quotes the papers directly and is
more clearly a derivative than trained weights are.

Of the corpus's 7,921 papers, **3,806 are eligible** (`open` class, 410,265
passages) — about half. The rest are `arxiv-nonexclusive`, `no-derivatives`,
`noncommercial` or `unknown`, and `unknown` means unknown, not permitted.

Runs resume: examples are keyed by a hash of their question and passages, so
re-running with the same `--seed` skips what is already on disk instead of
regenerating it.

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

**Built and smoke-tested on 2026-09-15**, on an arm64 Mac through Colima: the
compose file validates, the image builds, and `--help`, `stats` and a real
`search` against the sample all succeed - the search downloading the embedding
model inside the container, which is the first-run experience. A pre-publish
audit had already fixed two things a first build would have hit: the
container installing dev dependencies on its first command, and a build
context that would have sent the entire corpus to the daemon. The `container`
workflow repeats the same steps on every push, on amd64, so the path stays
checked rather than having been checked once.
