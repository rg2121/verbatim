# Architecture

Everything runs on one machine: corpus, index and generation. The property the
whole design serves is that **a claim cannot appear in an answer unless its
quoted words are in the passage it cites**, checked mechanically rather than
trusted.

One rule governs what ships: **anything unmeasured is off by default**, with
its numbers written down - including the ideas that lost. Those numbers are in
[findings.md](findings.md).

## The store

`data/corpus.db` is one SQLite file and the only source of truth. PDFs, HTML
files, the chunk export and the vector index are all derived from it and can be
rebuilt, and nothing downstream writes back into `papers`. An interrupted stage
costs the batch in flight and nothing more.

| table | holds |
|---|---|
| `papers` | every version seen; `is_latest` marks the current one |
| `downloads` / `download_failures` | local PDFs with checksums; the papers arXiv has no PDF for |
| `html_sources` | local HTML renderings, and the papers arXiv renders nothing for |
| `chunks` | passages with character offsets, from HTML or PDF, never both |
| `chunks_fts` / `papers_fts` | BM25 over passages and over titles |
| `embeddings` / `paper_embeddings` | one vector per passage and per abstract, keyed by model |
| `vec_chunks` / `vec_papers` | nearest-neighbour indexes, one model at a time |
| `paper_references` | bibliographies read from the HTML: the citation graph |
| `runs` | one row per command, failures included |

## Ingest

`sync → licenses → download → html → references → chunk → embed → index`.
`pipeline` runs them in rounds, because each creates work for the next, and
stops when a round changes nothing rather than when every count reaches zero -
some work never completes, such as the 33 papers that return a truncated PDF
every time. Each stage runs as its own process, so one failure does not end the
ingest. Every stage is incremental: chunking is keyed on
`version:chunk_size:overlap`, embedding on `(chunk_id, model_id)`.

**`sync`** pages a category newest first and stops at a stored watermark, so a
repeat costs one request. A failed run leaves the watermark where it was, so
the pages it missed are not skipped for good. The watermark only looks forward:
`--full` re-walks a category to go deeper, and cannot resume if it fails. A
second, backward cursor is designed (a stored `oldest_updated` date, not a page
offset, since arXiv's offsets shift) and deliberately not built until someone
needs to deepen a category.

**`licenses`** reads each paper's licence from arXiv's OAI-PMH service
(`oaipmh.arxiv.org`), because the query API never returns one. Licences are
grouped by what they permit, not by whether one exists:

| class | meaning |
|---|---|
| `open` | CC-BY, CC-BY-SA, CC0 - redistribution and derivatives permitted |
| `arxiv-nonexclusive` | arXiv's default: **arXiv** may distribute it, you may not |
| `no-derivatives`, `noncommercial` | CC terms that rule out training or commercial use |
| `unknown` | not looked up, or no licence recorded - unknown, not permitted |

A sync that reports no licence never overwrites one already found, and a
failed lookup is never marked done, so it is retried.

**`html` before the PDF.** arXiv renders most recent papers from the author's
TeX, so maths survives as LaTeX, headings are the author's own and a citation
can name a real section instead of an invented page. A paper is chunked from
exactly one source; 94% of the corpus is read from HTML. A rendering that
covers too little of its PDF is set aside, because a truncated rendering is
worse than the PDF. Tables are carried as passages of their own, with spanned
cells placed in their real columns (9,343 tables over 1,547 renderings).
Layout extraction of a PDF runs in a child process with a timeout, because one
paper held a core for 46 minutes; a paper that times out or crashes is skipped
and marked, not retried for ever.

**`references`** reads each bibliography from the HTML and resolves an entry to
an arXiv id only when it states one - a match by title similarity would be a
guess recorded as a fact. The edges between held papers form the citation
graph: `grow` follows them outward to the papers the corpus cites and lacks,
and `graph` inward to what cites what.

## Retrieval

Keyword (BM25) and vector search, fused by Reciprocal Rank Fusion, which
combines rankings by position so two incomparable scores never have to be
reconciled. At most three passages come from one paper by default, so a
corpus-wide question is not answered from a single source. Bibliographies are
kept out unless `--include-references` asks for them; every other kind of
passage, tables included, is searched. `--since` and `--category` narrow the
papers first.

Two more legs run **only on the question shape they were measured to help**,
because each measured worse always-on:

| leg | fires when | why |
|---|---|---|
| citation-graph anchor (on) | the question is definition-shaped *and* a title-matched paper is cited by at least two held papers | 3,152 papers mention Adam; the paper defining it holds 53 passages, so definitions drown in usage |
| comparison titles (off, `--use-compare`) | the question uses comparison vocabulary | "compare A and B" names its papers almost by title |

Both stand down whenever the caller has scoped or filtered the search, since
they name papers of their own.

The vector index records **which model's vectors it holds and which corpus
generation it was built from**, and is rebuilt on a mismatch. Two embedding
models of the same width once searched each other's vectors silently, and a
source replacement once left every passage of a paper unsearchable while the
row count still matched. Search over the index is brute force and exact: about
0.2 s over 835,743 vectors, which is not the bottleneck.

## Answering

Passages go to a model through **Ollama**, the only backend - no API key, no
second client. A local model keeps the corpus on the machine; a model whose
name ends `-cloud` sends each prompt's passages to Ollama's servers. The choice
is the user's, visible in `OLLAMA_MODEL`. The model replies in a flat
`ANSWER:` / `CITE` line protocol, because JSON output and tool calling were
unreliable on realistic prompts.

**Every citation is then verified**: the quoted words must appear in the
passage named, forgiving only whitespace and the quotation marks a model wraps
around a quote. A citation that fails is rejected and its marker becomes
`[unverified]`. This is a substring check, never a similarity score and never
a second model call. It proves the words exist in the source. It does not
prove the passage answers the question, and nothing strengthens it into that.

For that second problem the system **reports rather than judges**. Asking a
model whether passages support an answer was measured seven ways and failed
every time - the largest model tried did worst - because passages are
retrieved *because* they match and a model reads that as sufficiency. So each
answer says instead what it rests on: how many passages, from how many papers,
what those passages are about (a model names their subject without seeing the
question, and the two are compared arithmetically), and, for a definitional
question, the paper the subject is named after. Nothing in that account gates
or edits the answer.

**A conversation** is used for one thing only: resolving a follow-up ("why
does it clip?") into a question that stands alone, before retrieval. It never
reaches the model that writes the answer, because prose from an earlier turn
cannot be checked against any paper.

## The ladder

`ask --deep`, `report --agents` and the page's "try harder" route a question by
its shape to a ladder of retrieval strategies, write an answer, judge it by
numbers alone, and retry with the *next* strategy when it fails. The supervisor
is code, not a model.

| role | kind | does |
|---|---|---|
| planner | model | writes a review's section questions |
| router | code | question shape → ladder (definitional, comparison, recency, open) |
| retriever | code; the last rung is a tool-choosing agent | `hybrid`, `compare`, `titles`, `agent` |
| writer | model | the verified-citation path above, unchanged |
| verifier | code | accept if a citation verified and the passages are on-subject |
| supervisor | code | ledger, retries, budget, assembly |

A retry is never the same retrieval twice, the best attempt is kept rather than
the last, and a rung that errors is recorded and passed over. A section nothing
could support is a **gap** - a claim about the corpus - while one whose
machinery fell over is a **failure**, which says nothing about the corpus. A
recency question is answered from the catalogue on the agent rung: the rows
are the answer, so nothing is generated or needs verifying. Every attempt goes
into a ledger printed with the review, including which tools the agent called,
which papers it stood on and what it cost; a budget caps a run by tokens,
seconds or attempts. `--supervisor langgraph` runs the same ladder as a
LangGraph state machine behind the same role contracts (`contracts.py`).

Three designs around it are off by default, each with its measurement in
[findings.md](findings.md):

- **The critic** (`--critic`) describes rather than judges: it counts sentences
  with no citation marker, and lets a model list what the *question* is about,
  dropping any subject a passage actually contains. Measured, escalating on it
  made answers worse.
- **`team`** puts workers around a shared evidence board: they append typed
  records, and code reads the board for what is missing to decide who runs
  next. The board lives for one run; persisting it waits for a run long enough
  to resume, workers that run asynchronously, or an audit the run counts
  cannot answer.
- **`gaps`** sets a Prospector that asserts an absence against a Falsifier
  rewarded for refuting it with a verified quote, with an effort floor deciding
  whether a failure to refute counts.

## Front ends

The CLI, a FastAPI page (`serve`) and an MCP server (`mcp`) share one
retrieval and answering path, so their behaviour cannot drift; the defaults
each passes are tested too. The page streams: first what was retrieved, then
the draft as the model writes it, then the verified answer that replaces it -
streaming changes when text is shown, never what is accepted. `report` is not
an MCP tool, because a ten-minute blocking call is a worse version of the
streaming route. The server binds to loopback, and the corpus is never served
to a network unless someone says so.

Every command writes a row to the run ledger - when, what it was asked, what it
produced, and failures - and `/api/health` reads it, so a corpus whose last
ingest failed reports that rather than a bare 200.

## Licences, the sample and training data

A shipped corpus redistributes papers' text, so the committed sample may hold
only `open` papers; one SQL rule gates every path into it. (The first sample
broke this and the history was rewritten to remove it.) Opening a corpus never
changes it, so the committed sample stays byte-identical after use.

`distill` generates training examples under
[TRAINING-POLICY.md](../TRAINING-POLICY.md): the licence filter is pushed into
both retrieval legs so an ineligible passage never reaches a prompt, one failed
quote rejects a whole example, and every example carries its provenance. A
final check reports a leak as a fault rather than a rejection.

## Testing

The suite runs offline in seconds with no network, no Ollama and no API key:
every backend sits behind a protocol (`Embedder`, `Completion`, `VectorIndex`,
`ToolCaller`) and tests inject stand-ins. The web page's script runs in node
against a stand-in DOM where node is installed. CI adds a retrieval regression
gate over the committed sample and a container build.
