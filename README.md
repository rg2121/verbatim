# Verbatim

A local-first, citation-focused literature-review assistant for scientific papers,
named for the one thing it guarantees: a quote either appears word for word in
the passage it cites, or it is thrown out.

![Sixty seconds against the sample corpus: stats, a search, an answer whose every quote is verified, the citation graph, and the run ledger](demo/demo.gif)

Verbatim builds a searchable corpus of arXiv papers on your own machine, answers
questions from it with every claim quoted and every quote checked against its
source, and writes whole literature reviews the same way. What was measured
along the way - including the many ideas that did not work - is written up in
[docs/](docs).

## Features

- **Verified citations.** Every quoted span is checked character for character
  against the passage it cites. A quote that fails is rejected and its marker
  becomes `[unverified]`. No model is asked whether a citation is good.
- **Traceable sources.** Citations pin the exact arXiv version, with the page or
  section, character offsets, source URL and licence.
- **Hybrid retrieval.** Keyword and vector search fused by rank, plus a
  citation-graph leg that finds the paper a definitional question is named
  after ("what is the Adam optimizer?"), on by default.
- **HTML-first ingest.** Papers are read from arXiv's HTML rendering where one
  exists - maths stays LaTeX, tables stay tables - and from the PDF otherwise.
- **Answers that say what they rest on.** Every answer reports how many passages
  and papers it used and what those passages are actually about.
- **Several ways to answer.** `ask` for a question, `ask --deep` to retry with a
  different retrieval when nothing verifies, `report` for a sectioned review,
  `team` for workers around a shared evidence board, and `gaps` for claims of
  absence that another agent tries to refute.
- **Three front ends on one core.** The CLI, a web page with streaming answers
  and follow-up questions, and an MCP server for other assistants.
- **Observable.** Every command writes a row to a run ledger, and
  `/api/health` reports corpus state and failed runs.
- **Licence-aware.** Licences come from arXiv's OAI-PMH records and are grouped
  by what they permit; training data is generated only from papers whose
  licence allows it.
- **Local-first.** Models run through Ollama, retrieval needs no model at all,
  and the test suite runs offline.

## Quick start

You need Python 3.11+, [uv](https://docs.astral.sh/uv/), and - for answering
questions - [Ollama](https://ollama.com) with any model pulled.

```bash
git clone https://github.com/rg2121/verbatim.git
cd verbatim
uv run verbatim search "how do transformers handle long context" --state sample/corpus.db
uv run verbatim ask "what problem does QLoRA solve?" --state sample/corpus.db
uv run verbatim graph --state sample/corpus.db
```

The repository ships a sample corpus - 40 open-licence papers, 3,139 passages
with their vectors, 2,230 bibliography entries, about 23 MB - so this works on a
fresh clone. The first search downloads the embedding model (about 130 MB).
`search` and `graph` need no language model; `ask` needs Ollama. How the sample
was chosen and how to regenerate it:
[docs/guide.md](docs/guide.md#the-sample-corpus).

Then ask something the sample does not cover:

```bash
uv run verbatim ask "what is a neural turing machine?" --state sample/corpus.db
```

None of the forty papers is about that, and the answer says so rather than
assembling something plausible from the nearest passages.

Any Ollama model works. Leave `OLLAMA_MODEL` unset to use the first pulled
model, or name one per command:

```bash
OLLAMA_MODEL=llama3.1 uv run verbatim ask "what problem does QLoRA solve?" --state sample/corpus.db
```

## Building your own corpus

```bash
uv run verbatim sync --category cs.CL cs.IR --limit 2000   # paper metadata, newest first
uv run verbatim pipeline                                   # licences, PDFs, HTML, bibliographies, passages, vectors
uv run verbatim stats                                      # what is done and what is left
```

`pipeline` runs every ingest stage in rounds until nothing is pending. Each
stage is incremental, so an interrupted run resumes rather than restarts, and
each can be run on its own: `licenses`, `download`, `html`, `references`,
`chunk`, `embed`, `index`. After `references`, `grow` adds the papers your
corpus cites most and does not yet hold.

Worth knowing before a large ingest:

- `sync` stops at a stored watermark per category, so re-running it costs one
  request. A category first synced with a small `--limit` does not get deeper
  later unless you pass `--full`.
- `download --license open` keeps only papers whose licence permits
  redistribution. arXiv's default licence grants rights to arXiv, not to you.
- The whole corpus is one SQLite file, `data/corpus.db` by default. Everything
  else - PDFs, HTML, the chunk export, the vector index - is derived from it.

Every stage in detail: [docs/guide.md](docs/guide.md#daily-corpus-workflow).

## Using it

### Search

```bash
uv run verbatim search "hybrid retrieval" --since 2026-06-01 --category cs.IR
uv run verbatim search "spin-orbit coupling" --mode keyword --full
```

Hybrid by default; `--mode keyword` for exact terms and identifiers, `--mode
vector` for meaning without shared words. `--arxiv-id` searches within one
paper, `--since` and `--category` filter by date and subject, and `--full`
prints whole passages. Keyword search matches whole tokens, so use `field*` for
a prefix.

### Ask a question

```bash
uv run verbatim ask "can magnetic flux move through a region of zero field?"
```

```
Flux can move through zero field [1].

References
[1] Walstad, "Moving Magnetic Flux and Electromagnetic Induction",
    arXiv:2608.05440v1, p. 1. https://arxiv.org/abs/2608.05440v1
```

Below the references each answer says what it rests on - how many passages,
from how many papers, and what those passages are about - and what the model
calls cost ([real output](demo/transcript.md)). When the
passages are about something other than what you asked, it warns rather than
refusing. Useful options:

| option | what it does |
|---|---|
| `--deep` | when no citation verifies or the passages drift off-subject, retry with the next retrieval strategy for the question's shape |
| `--tools` | let the model route the question: recency goes to the catalogue, a named paper to a scoped search |
| `--arxiv-id ID` | answer from one paper version |
| `--use-compare` | on comparison questions, also retrieve from papers whose titles match |
| `--plan` | split a compound question and retrieve for each part (measured no better; kept for comparison) |
| `--max-per-paper N` | cap the passages taken from any one paper (default 3) |
| `--llm MODEL` | choose the Ollama model |

`--deep` and `--tools` choose their own retrieval, so they refuse search
options they would otherwise ignore. More on answering, including the options
above: [docs/guide.md](docs/guide.md#escalating-a-single-question).

### Write a literature review

```bash
uv run verbatim report "sparse attention for long context" --out review.md
uv run verbatim report "sparse attention for long context" --agents --out review.md
```

`report` plans three to five section questions, answers each through the same
verified path, and assembles one bibliography numbered across sections.
`--agents` routes each section by its shape, retries a section that fails
verification with a different strategy, and appends a ledger of what was tried.
`--max-tokens`, `--max-seconds` and `--max-attempts` bound a run, and
`--supervisor langgraph` runs the same ladder as a LangGraph state machine
(install it with `uv sync --extra langgraph`).

### Work a question as a team

```bash
uv run verbatim team "how do sparse attention methods trade accuracy for memory?"
```

A coordinator turns the question into tasks, a retrieval worker files
supporting passages, an adversarial worker searches for limitations and
criticism, and code reads the shared board to decide which worker runs next.
It is opt-in: on held-out questions it cost about six times a plain `ask` for no
measured gain, and a pre-registered test on the two-paper questions it was built
for is awaiting blind scoring. The numbers:
[docs/findings.md](docs/findings.md#a-team-around-a-shared-evidence-board).

### Look for gaps in the literature

```bash
uv run verbatim gaps "sparse attention for long context"
```

One agent proposes things the corpus lacks; another searches for a paper that
refutes each claim and has to quote it verifiably. A gap that stands means
"nothing found by an opponent who tried", not proof of absence - measured, about
one standing gap in twelve is false.

### Web page and HTTP API

```bash
uv run verbatim serve        # http://127.0.0.1:8000
```

A search box, a conversation that streams answers and resolves follow-up
questions, a "dig deeper" toggle, and a review panel that shows each section as
it is written. The JSON routes are `/api/search`, `/api/ask` and `/api/report`
(server-sent events), `/api/papers/{arxiv_id}`, `/api/stats` and `/api/health`,
with OpenAPI docs at `/docs`. It binds to loopback unless `--host` says
otherwise. [Details](docs/guide.md#serving-it-over-http).

### Use it from an MCP client

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

Five read-only tools - `search_chunks`, `get_paper`, `list_papers`,
`get_citations`, `read_pages` - returning JSON with the provenance intact.

### See what each run did

```bash
uv run verbatim runs
uv run verbatim runs --json    # exits non-zero if any listed run failed
```

Every command that touches the corpus records when it ran, what it was asked,
what it produced, and whether it failed.

### Evaluate

```bash
uv run verbatim eval --cases evals/heldout.jsonl evals/heldout-generated.jsonl
uv run verbatim eval-answers --cases evals/heldout.jsonl --sample 60 --runs 2
```

`eval` scores every retrieval mode on an answer key; `eval-answers` scores the
answers themselves. The keys, the results and how to reproduce them are in
[docs/evaluation.md](docs/evaluation.md).

### Generate training data

```bash
uv run verbatim distill --examples 200
```

Writes (question, retrieved passages, verified answer) examples under `data/`
from open-licence papers only, each with its provenance, plus an `ATTRIBUTIONS`
file - under the rules in [TRAINING-POLICY.md](TRAINING-POLICY.md).
[Details](docs/guide.md#generating-training-data).

### Run it in a container

```bash
docker compose run --rm verbatim search "how do transformers handle long context" --state sample/corpus.db
docker compose up verbatim-web                  # the page on 127.0.0.1:8000
docker compose --profile ollama up              # add a model for ask and report
```

The corpus is a mounted volume, not part of the image. The image is built and
smoke-tested in CI on every push.

## Every command

Each takes `--help` for its full options: `uv run verbatim <command> --help`.

**Build and inspect a corpus**

| command | what it does |
|---|---|
| `sync` | Fetch metadata and upsert it into the cumulative corpus catalogue |
| `fetch` | Fetch paper metadata from one arXiv category |
| `licenses` | Look up licences via OAI-PMH, which the query API omits |
| `download` | Download a bounded set of PDFs from a metadata snapshot |
| `html` | Fetch arXiv HTML renderings, which keep maths as LaTeX |
| `references` | Read bibliographies from HTML renderings and rank what the corpus lacks |
| `chunk` | Extract passages from HTML renderings, or from the PDF where there is none |
| `embed` | Embed chunks that have no vector yet |
| `index` | Reconcile the vector index |
| `grow` | Add the papers this corpus cites most and does not hold |
| `pipeline` | Run every ingest stage in rounds until nothing is pending |
| `stats` | Show corpus size and work remaining |
| `runs` | What each command did, and whether it finished |
| `graph` | Walk the internal citation graph: what cites a paper, what it builds on |

**Search and answer**

| command | what it does |
|---|---|
| `search` | Search the corpus: hybrid by default, or keyword or vector only |
| `ask` | Answer a question from the corpus, with citations |
| `report` | Write a sectioned literature review with verified citations |
| `team` | Answer through a shared evidence board that decides what runs next |
| `gaps` | Propose what a topic's literature is missing, then try to disprove it |

**Evaluate**

| command | what it does |
|---|---|
| `eval` | Score every retrieval mode on an answer key |
| `eval-answers` | Score the answers themselves, not whether retrieval named the expected paper |
| `author-cases` | Write held-out evaluation cases from titles and abstracts only |

**Serve and share**

| command | what it does |
|---|---|
| `serve` | Serve the corpus over HTTP, with a search page |
| `mcp` | Serve the corpus over the Model Context Protocol, on stdio |
| `sample` | Export a small self-contained corpus that works without an ingest |

**Training data**

| command | what it does |
|---|---|
| `distill` | Generate training examples from licence-eligible papers |

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

A local model keeps the corpus on your machine. A model whose name ends
`-cloud` is served by Ollama's infrastructure, so the passages in each prompt
leave it - which one you use is your choice, and it is visible in
`OLLAMA_MODEL`.

## Known limitations

- **It can answer a question the corpus cannot answer.** Every quote is real,
  but a real quote does not prove the passage answers the question. The line
  saying what the passages are about is there so you can judge: it flags most
  questions on subjects the corpus does not cover, and none where the subject is
  covered but the specific question is not answered.
- **Scanned PDFs yield nothing.** There is no OCR.
- **A corpus built by `sync` starts as a slice of recent papers.** `grow` adds
  the older work it cites.
- **Retrieval quality is corpus-specific**, and at the sample sizes measured most
  retrieval modes cannot be told apart; the evaluation reports intervals rather
  than a single score.
- **Licence data is only as good as arXiv's records.** `unknown` means unknown,
  not permitted.

The full list, with the measurements behind each:
[docs/findings.md](docs/findings.md#known-limitations).

## Documentation

- [docs/guide.md](docs/guide.md) — every ingest stage and front end in depth
- [docs/findings.md](docs/findings.md) — what was built, measured and rejected
- [docs/evaluation.md](docs/evaluation.md) — the answer keys, the results, and how to reproduce them
- [docs/architecture.md](docs/architecture.md) — what the pieces are and how they fit
- [docs/decisions](docs/decisions) — why the load-bearing choices are what they are
- [docs/postmortems](docs/postmortems) — what went wrong, and what it changed
- [demo/transcript.md](demo/transcript.md) — real output from the sample corpus
- [TRAINING-POLICY.md](TRAINING-POLICY.md) — what may be trained on, and on what terms

## Development

```bash
uv run --extra langgraph pytest    # the whole suite, offline, in a few seconds
uv run verbatim eval --cases evals/gate.jsonl --state sample/corpus.db --min-mrr 0.90
```

The suite needs no network, no Ollama and no API key: every model and network
call sits behind an interface the tests replace. Without the `langgraph` extra,
the LangGraph supervisor's tests skip. CI runs the suite, a retrieval regression
gate over the sample corpus, and a container build on every push.

The ingestion client uses arXiv's public metadata API. Before growing a corpus,
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
