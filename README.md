# Verbatim

A local-first research assistant for arXiv papers, named for the one thing it
guarantees: a quote either appears word for word in the passage it cites, or it
is thrown out.

![Sixty seconds against the sample corpus: stats, a search, an answer whose every quote is verified, the citation graph, and the run ledger](demo/demo.gif)

Verbatim builds a searchable corpus of arXiv papers on your own machine,
answers questions from it with every claim quoted and every quote checked
against its source, and writes literature reviews the same way.

## Features

- **Verified citations.** Every quote is checked character for character
  against the passage it cites. One that fails is rejected and its marker
  becomes `[unverified]`. No model is asked whether a citation is good.
- **Traceable sources.** Citations pin the exact arXiv version, the page or
  section, character offsets, source URL and licence.
- **Hybrid retrieval.** Keyword and vector search fused by rank, plus a
  citation-graph leg for definitional questions ("what is the Adam optimizer?").
- **HTML-first ingest.** Papers are read from arXiv's HTML rendering where one
  exists - maths stays LaTeX, tables stay tables - and from the PDF otherwise.
- **Answers that say what they rest on.** Every answer reports how many
  passages and papers it used and what those passages are about.
- **Several ways to answer.** `ask`, `ask --deep` to retry with a different
  retrieval, `report` for a sectioned review, `team` and `gaps`.
- **Three front ends.** A CLI, a web page with streaming answers and follow-up
  questions, and an MCP server for other assistants.
- **Local-first and licence-aware.** Models run through Ollama and retrieval
  needs none. Training data is generated only from openly licensed papers.

## Quick start

You need Python 3.11+, [uv](https://docs.astral.sh/uv/), and - for answering -
[Ollama](https://ollama.com) with any model pulled.

```bash
git clone https://github.com/rg2121/verbatim.git
cd verbatim
uv run verbatim search "how can a model be finetuned on one GPU" --state sample/corpus.db
uv run verbatim ask "what is QLoRA?" --state sample/corpus.db
uv run verbatim graph --state sample/corpus.db
```

The repository ships a sample corpus - 40 openly licensed papers, 3,139
passages with their vectors, 2,230 bibliography entries, 23 MB - so this works
on a fresh clone. The first search downloads the embedding model (about
130 MB). `search` and `graph` need no language model; `ask` needs Ollama, and
uses `OLLAMA_MODEL` or the first model you have pulled.

Real output from the `ask` above, with the answer shortened:

```
QLoRA is a method for "high-fidelity 4-bit finetuning" [1] that utilizes "Double Quantization" and "4-bit NormalFloat (NF4) quantization" [1]. It is designed to "significantly reduce the required memory for finetuning models" [2] and is the first method that would allow for the finetuning of 7B models on phones and other low resource settings [3]. ...

References
[1] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314v1, section 3 QLoRA Finetuning. https://arxiv.org/abs/2305.14314v1
[2] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314v1, section 4 QLoRA vs. Standard Finetuning. https://arxiv.org/abs/2305.14314v1
[3] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314v1, section 9 Broader Impacts. https://arxiv.org/abs/2305.14314v1
  Answered from 8 passages in 4 papers.
  Those passages are about: QLoRA and algorithmic search neutrality

  2 model calls, 2,528 tokens in, 585 out, 2.3s in the model
```

Ask it something the forty papers do not cover - "what is a neural turing
machine?" - and it says so rather than assembling an answer from the nearest
passages.

## Building your own corpus

```bash
uv run verbatim sync --category cs.CL cs.IR --limit 2000   # paper metadata, newest first
uv run verbatim pipeline                                   # licences, PDFs, HTML, bibliographies, passages, vectors
uv run verbatim stats                                      # what is done and what is left
```

`pipeline` runs every ingest stage in rounds until nothing is left to do. Each
stage is incremental, so an interrupted run resumes, and each can be run on its
own. After `references`, `grow` adds the papers your corpus cites most and
does not yet hold.

- The whole corpus is one SQLite file, `data/corpus.db`. Everything else is
  derived from it.
- `sync` stops at a stored watermark per category, so re-running it is cheap.
  To go deeper into a category than its first sync did, pass `--full`; a
  `--full` run cannot resume, so deepen a large category in stages.
- `download --license open` keeps only papers whose licence permits
  redistribution. arXiv's default licence grants rights to arXiv, not to you.

## Commands

Every command takes `--help`: `uv run verbatim <command> --help`.

| | command | what it does |
|---|---|---|
| **Build** | `sync` | fetch paper metadata into the corpus |
| | `licenses` | look up licences, which arXiv's query API omits |
| | `download`, `html` | fetch PDFs and arXiv HTML renderings |
| | `references` | read bibliographies, and list what the corpus cites and lacks |
| | `chunk`, `embed`, `index` | cut passages, embed them, reconcile the vector index |
| | `grow` | add the papers the corpus cites most and does not hold |
| | `pipeline` | run every stage above until nothing is pending |
| | `fetch` | a one-off metadata snapshot, outside the corpus |
| **Inspect** | `stats` | corpus size and work remaining |
| | `runs` | what each command did, and whether it failed |
| | `graph` | what cites a paper, what it builds on, what the corpus cites most |
| **Answer** | `search` | hybrid by default; `--mode keyword` or `vector` |
| | `ask` | an answer with verified citations |
| | `report` | a sectioned literature review with one bibliography |
| | `team` | workers around a shared evidence board |
| | `gaps` | claims of what the literature lacks, which another agent tries to refute |
| **Evaluate** | `eval`, `eval-answers`, `author-cases` | score retrieval, score answers, write held-out questions |
| **Share** | `serve` | the web page and JSON API on `127.0.0.1:8000` |
| | `mcp` | the corpus as MCP tools, over stdio |
| | `sample` | export a small self-contained corpus |
| | `distill` | training examples from openly licensed papers only |

**Options for `ask`:** `--deep` retries with the next retrieval strategy when
no citation verifies; `--tools` lets the model route the question (recency to
the catalogue, a named paper to a scoped search); `--arxiv-id` answers from one
paper; `--max-per-paper` caps passages per paper (default 3); `--llm` picks the
Ollama model. `--deep` and `--tools` choose their own retrieval, so they refuse
search options they would otherwise ignore.

**`report`** plans three to five section questions and answers each through the
same verified path. `--agents` retries a section that fails verification with a
different strategy and appends a ledger of what was tried; `--max-tokens`,
`--max-seconds` and `--max-attempts` bound a run.

**`serve`** gives a search box, a conversation that resolves follow-up
questions, and a review panel that shows each section as it is written. The
JSON routes are documented at `/docs` once it runs. It binds to loopback unless
`--host` says otherwise.

**`mcp`**, from an MCP client's config:

```json
{"mcpServers": {"verbatim": {"command": "uv", "args": ["run", "verbatim", "mcp"], "cwd": "/path/to/verbatim"}}}
```

**In a container:** `docker compose run --rm verbatim search "..." --state
sample/corpus.db`, `docker compose up verbatim-web` for the page, and
`--profile ollama` to add a model. The corpus is a mounted volume, not part of
the image.

## Configuration

Copy `.env.example` to `.env`; it documents every setting. An exported
variable overrides the file, so a one-off needs no edit:

```bash
OLLAMA_MODEL=llama3.1 uv run verbatim ask "..."
```

A local model keeps the corpus on your machine. A model whose name ends
`-cloud` is served by Ollama's infrastructure, so the passages in each prompt
leave it - which one you use is your choice, visible in `OLLAMA_MODEL`.

## Limitations

- **It can answer a question the corpus cannot answer.** Every quote is real,
  but a real quote does not prove the passage answers the question. The line
  saying what the passages are about is there so you can judge.
- **Scanned PDFs yield nothing.** There is no OCR.
- **A corpus built by `sync` starts as a slice of recent papers.** `grow` adds
  the older work it cites.
- **Retrieval quality is corpus-specific**, and at the sample sizes measured
  most retrieval modes cannot be told apart.
- **Licence data is only as good as arXiv's records.** `unknown` means unknown,
  not permitted.

## Documentation

- [docs/architecture.md](docs/architecture.md) - how it works, and why it is
  built this way
- [docs/findings.md](docs/findings.md) - what was measured, including
  everything that did not work
- [TRAINING-POLICY.md](TRAINING-POLICY.md) - what may be trained on, and on
  what terms

## Development

```bash
uv run --extra langgraph pytest    # the whole suite, offline, in a few seconds
uv run verbatim eval --cases evals/gate.jsonl --state sample/corpus.db --min-mrr 0.90
```

The suite needs no network, no Ollama and no API key: every model and network
call sits behind an interface the tests replace. CI runs the suite, a
retrieval regression gate over the sample corpus, and a container build on
every push. Before growing a large corpus, read arXiv's API and bulk-access
guidance.

## Licence

**AGPL-3.0-or-later.** Copyright (c) 2026 Rishab Gupta.

The PDF parsing depends on [PyMuPDF](https://pymupdf.readthedocs.io/), which is
dual-licensed AGPL-3.0 or commercial, so this project must be AGPL too. You may
read, run, fork and modify it freely; if you publish a modified version or run
it as a network service, your source must be available under the same terms.
