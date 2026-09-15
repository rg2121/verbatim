# Architecture

A local-first literature-review assistant over arXiv. Everything runs on one
machine: corpus, index, and generation. The property the whole design serves
is that **a claim cannot appear in an answer unless its quoted words are in
the passage it cites**, checked mechanically rather than trusted.

## The store is the source of truth

`data/corpus.db` is one SQLite file. Everything else — the JSON snapshot, the
PDFs, the chunk export, the vector index — is derived and rebuildable from it.
Nothing downstream writes back into `papers`.

| table | holds |
|---|---|
| `papers` | every version seen; `is_latest` marks the current one |
| `downloads` / `download_failures` | local PDFs with checksums; the ones arXiv has no PDF for |
| `html_sources` | local HTML renderings, and the papers arXiv renders nothing for |
| `chunks` | passages with character offsets, from HTML or PDF, never both |
| `chunks_fts` / `papers_fts` | BM25 over passages and over titles |
| `embeddings` / `paper_embeddings` | one vector per chunk and per abstract, keyed by model |
| `vec_chunks` / `vec_papers` | nearest-neighbour indexes, one model at a time |
| `paper_references` | bibliographies read from the HTML; the citation graph |
| `runs` | one row per command invocation |
| `vector_index_state` | which model's vectors each index currently holds |

## Stages

`sync → licenses → download → html → references → chunk → embed → index`, then
`search`, `ask`, `report`. `pipeline` runs the ingest stages in rounds until
nothing is pending; `grow` adds papers the corpus cites and does not hold.

Every expensive stage is incremental, keyed by a fingerprint: chunking on
`version:chunk_size:overlap`, embedding on `(chunk_id, model_id)`. Re-running
after an interruption resumes rather than restarts.

## Retrieval

Hybrid: BM25 and vector search fused by Reciprocal Rank Fusion, which combines
rankings by position so two incomparable scores never have to be reconciled.
Two further legs are **gated by question shape** rather than always on:

| leg | fires when | why |
|---|---|---|
| citation-graph anchor | the question is definition-shaped *and* a title-matched paper is cited by ≥2 held papers | definitional questions drown in usage: 3,152 papers mention Adam, the paper defining it holds 53 passages |
| comparison titles | the question uses comparison vocabulary | a "compare A and B" question names its papers almost by title |

Gating is the point. The same title leg measured always-on costs more than it
earns; measured behind a gate it does not. See
[decisions/0006](decisions/0006-gate-retrieval-legs-by-question-shape.md).

## Answering

Retrieved passages go to a model through Ollama - a local one, or a `-cloud`
one whose prompts leave the machine
([decisions/0010](decisions/0010-local-first-and-where-that-boundary-moved.md)) -
which replies in a flat
`ANSWER:` / `CITE` line protocol — documented structured-output mechanisms were
measured unreliable on realistic prompts. Every citation is then checked: the
quoted words must appear in the passage named, or the citation is rejected and
its marker becomes `[unverified]`. This is a deterministic substring check and
never a second model call.

Around that sit three things that do not decide, only report: the subject
check (what are these passages about, compared arithmetically to the
question), the evidence account under every answer, and the run ledger.

## Front ends

CLI, FastAPI web and an MCP server share one retrieval and answering path
(`retrieval.py`, `answers.py`); the MCP server and the tool-choosing agent also
share the tool layer in `tools.py`. Sharing functions is not sharing defaults -
the web search route once ran without the graph leg every other front end used -
so the defaults each route passes are tested too. The web page adds streaming (`/api/ask`,
`/api/report` over server-sent events) and a conversation whose history is used
only to resolve a follow-up into a standalone question **before retrieval** —
never given to the model that writes the answer, because prose from an earlier
turn cannot be checked against any paper.

## Orchestration

`report --agents` routes each review section by its shape to a ladder of
retrieval strategies, has the section written by the model, judges it by
numbers alone, and retries with the *next* strategy when the verifier rejects
it. The supervisor is code, not a model.

The roles are contracts (`contracts.py`), not just functions that fit
together, so a role can be swapped or tested against its contract alone:

| role | kind | contract |
|---|---|---|
| planner | model | `(topic, max_sections) -> Plan` |
| router | code | `question -> shape -> ladder` |
| retriever | code, and one tool-using agent on the last rung | `(question, strategy, limit) -> Gathered` |
| writer | model | `(question, Gathered) -> Answer` |
| verifier | code | `(Answer, closeness) -> Verdict` |
| critic | code, model-assisted | `(question, Answer, Gathered) -> Critique` |
| supervisor | code | ladder, budget, ledger, assembly |

Every attempt is recorded in a ledger printed with the review: the strategy,
the tools an agent rung called, **which papers it stood on**, what it cost in
calls and tokens, the verdict, and what the critic found. A `Budget` caps a
whole run by tokens, seconds or attempts; exhausting it stops further work
without discarding work already done. See
[decisions/0007](decisions/0007-a-deterministic-supervisor.md) and
[decisions/0011](decisions/0011-a-critic-that-describes-rather-than-judges.md).

The same ladder runs in two more places. `ask --deep` climbs it for a single
question, and `--supervisor langgraph` runs it as a LangGraph state machine
behind the same contracts, sharing every role and the selection rule.

Two further designs are different in kind, and both are off by default:

- `team` puts workers around a shared evidence board. Workers append typed
  records; code reads the board for what is structurally missing and that
  decides which worker runs next, so one worker's findings redirect another's
  work ([decisions/0013](decisions/0013-the-board-lives-for-one-run.md)).
- `gaps` sets a Prospector that asserts an absence against a Falsifier rewarded
  for refuting it with a verified quote, with an effort floor deciding whether a
  failure to refute counts.

Their measurements, including where they do not pay, are in the README.

## Testing

The suite runs offline in about two seconds with no network, no Ollama and no
API key, because every backend sits behind a protocol (`Embedder`,
`Completion`, `VectorIndex`, `ToolCaller`) and tests inject stand-ins. CI runs
that suite plus a retrieval regression gate over the sample corpus committed to
the repository.
