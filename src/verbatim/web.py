"""An HTTP front end onto the corpus, and a page that uses it.

A third front end, alongside the CLI and the MCP server, onto the same tools.
Nothing here searches, resolves or cites: it translates HTTP into the calls
that already exist, so behaviour cannot diverge between the three.

The page is served from this module rather than a build directory. It is one
file with no dependencies, no bundler and no external requests, because the
project runs offline once the corpus is built and a page that reaches a CDN
would quietly undo that.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import queue
import threading
from pathlib import Path
from typing import Any

from verbatim import config
from verbatim.answers import (
    SYSTEM_PROMPT,
    Answer,
    answer_from_payload,
    build_prompt,
    describe_subject,
    parse_line_protocol,
    subject_similarity,
)
from verbatim.citations import render_locator
from verbatim.conversation import Turn, standalone_question
from verbatim.corpus import ChunkMatch, CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.retrieval import hybrid_search
from verbatim.runs import last_run_per_command, recent_runs

MAX_LIMIT = 50
# A review is minutes of generation per section, so the ceiling is low on
# purpose: this endpoint is the most expensive thing the server can be asked
# to do, and the planner is told to write three to five sections anyway.
MAX_SECTIONS = 6


def _event(name: str, payload: dict[str, Any]) -> str:
    """One server-sent event. The name routes it to the page's handler."""
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def _section_payload(section: Any, entry: Any) -> dict[str, Any]:
    """One finished section, with the trace of how it was arrived at."""
    return {
        "heading": section.heading,
        "text": section.answer.text,
        "declined": section.answer.declined,
        "evidence": list(section.evidence_lines),
        "citations": [
            {
                "arxiv_id": citation.arxiv_id,
                "title": citation.title,
                "locator": render_locator(
                    citation.source_kind, citation.page_number, citation.section_path
                ),
                "source_url": citation.source_url,
            }
            for citation in section.answer.citations
        ],
        "rejected": len(section.answer.rejected),
        "shape": entry.shape,
        "outcome": entry.outcome,
        "attempts": [
            {
                "strategy": attempt.strategy,
                "verified": attempt.verified,
                "rejected": attempt.rejected,
                "verdict": attempt.verdict,
                "route": attempt.route,
                "error": attempt.error,
            }
            for attempt in entry.attempts
        ],
    }


def _usage_totals(*models: Any) -> tuple[int, int, int, float] | None:
    """A snapshot of what these clients have counted so far.

    None when nothing counts, which is every scripted stand-in in the tests.
    """
    counted = [getattr(model, "usage", None) for model in models]
    present = [usage for usage in counted if usage is not None]
    if not present:
        return None
    return (
        sum(u.calls for u in present),
        sum(u.prompt_tokens for u in present),
        sum(u.output_tokens for u in present),
        sum(u.seconds for u in present),
    )


def _cost_since(
    before: tuple[int, int, int, float] | None, *models: Any
) -> str:
    """What this run cost, as the difference from a snapshot.

    A served client lives as long as the process, so its totals are the
    server's lifetime rather than this review's - which is what the footer
    was reporting.
    """
    after = _usage_totals(*models)
    if before is None or after is None:
        return "cost not reported"
    calls, prompt, output, seconds = (a - b for a, b in zip(after, before))
    return (
        f"{calls} model call{'s' if calls != 1 else ''}, "
        f"{prompt:,} tokens in, {output:,} out, {seconds:.1f}s in the model"
    )


def _turns(history: str | None) -> list[Turn]:
    """Read the client's conversation, ignoring anything malformed.

    The history is a convenience for retrieval, so a client that sends
    nonsense gets a single-turn answer rather than an error.
    """
    if not history:
        return []
    try:
        payload = json.loads(history)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    turns = []
    for item in payload:
        if isinstance(item, dict) and str(item.get("question", "")).strip():
            turns.append(
                Turn(str(item["question"]), str(item.get("answer", "")))
            )
    return turns


def match_to_dict(match: ChunkMatch) -> dict[str, Any]:
    """Render one hit with everything needed to check it.

    The locator, licence and source URL travel with the passage: a result the
    reader cannot trace back is the failure mode this project exists to avoid.
    """
    chunk = match.chunk
    return {
        "chunk_id": chunk.chunk_id,
        "arxiv_id": chunk.arxiv_id,
        "locator": render_locator(chunk.source_kind, chunk.page_number, chunk.section_path),
        "section": chunk.section_path.split(" > ")[-1] if chunk.section_path else "",
        "source_kind": chunk.source_kind,
        "source_url": chunk.source_url,
        "license_url": chunk.license_url,
        "score": round(match.score, 4),
        "text": chunk.text,
        "snippet": match.snippet,
    }


def create_app(
    state: Path | None = None,
    embedding_model: str | None = None,
    completion: Any = None,
    encoder: Any = None,
    tool_caller: Any = None,
) -> Any:
    """Build the application. Kept a factory so tests can point it at a fixture.

    `completion` is the answering model and `encoder` the embedder; tests
    inject scripted ones, so the suite never loads ONNX weights or needs
    Ollama running. By default Ollama is constructed lazily on the first
    question, so serving the search page never requires a model at all.
    """
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, StreamingResponse

    database = state or config.STATE_PATH
    model_name = embedding_model or config.EMBEDDING_MODEL
    resources: dict[str, Any] = {}
    # The lazily built model and embedder are shared by every request thread
    # and by the review worker, and building one takes seconds; without this
    # two first callers can build two and race inside the ONNX loader.
    building = threading.Lock()
    # A SQLite connection may only be used by the thread that opened it, and
    # FastAPI serves synchronous endpoints from a pool, so consecutive requests
    # arrive on different threads. One shared connection therefore fails as
    # soon as a second thread is used - which is intermittent, and looked like
    # a dropped connection in the browser. Each thread gets its own instead;
    # the pool is small and long-lived, so this is a handful of readers.
    stores = threading.local()

    @asynccontextmanager
    async def lifespan(_: Any) -> Any:
        yield
        # Only this thread's connection can be closed from here, and the rest
        # are released when the process exits. Closing another thread's is the
        # same error this arrangement exists to avoid.
        store = getattr(stores, "corpus", None)
        if store is not None:
            store.close()
            stores.corpus = None

    app = FastAPI(
        title="verbatim",
        description="Citation-checked search over a local arXiv corpus.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def corpus() -> CorpusStore:
        store = getattr(stores, "corpus", None)
        if store is None:
            store = CorpusStore(database)
            stores.corpus = store
        return store

    def embedder() -> Any:
        with building:
            if "embedder" not in resources:
                resources["embedder"] = encoder or FastEmbedEmbedder(model_name=model_name)
            return resources["embedder"]

    def answering() -> Any:
        with building:
            if "completion" not in resources:
                if completion is not None:
                    resources["completion"] = completion
                else:
                    from verbatim.answers import OllamaCompletion

                    resources["completion"] = OllamaCompletion(config.OLLAMA_MODEL)
            return resources["completion"]

    def routing() -> Any:
        """The tool-choosing model the orchestrator's last rung needs.

        Without one that rung retrieves nothing and the ladder's escalation
        stops a step early, which is how it shipped: the route built an
        orchestrator with no caller, so every streamed review had a dead rung.
        """
        with building:
            if "caller" not in resources:
                if tool_caller is not None:
                    resources["caller"] = tool_caller
                elif completion is not None:
                    # An injected model is the whole backend for this app, so
                    # a caller is not built behind its back: the suite must
                    # not reach Ollama, and it may itself choose tools.
                    resources["caller"] = (
                        completion if hasattr(completion, "choose_tool") else None
                    )
                else:
                    from verbatim.answers import OllamaToolCaller

                    resources["caller"] = OllamaToolCaller(config.OLLAMA_MODEL)
            return resources["caller"]

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return PAGE

    @app.get("/api/stats")
    def stats() -> dict[str, int]:
        # The model is named, not loaded: the page asks for this on every visit
        # and ONNX startup costs seconds.
        return corpus().pipeline_stats(_fingerprint(), embedder().model_id)

    @app.get("/api/search")
    def search(
        q: str = Query(min_length=1, description="What to search for"),
        limit: int = Query(10, ge=1, le=MAX_LIMIT),
        arxiv_id: str | None = Query(None, description="Restrict to one paper version"),
        mode: str = Query("hybrid", pattern="^(hybrid|keyword)$"),
    ) -> dict[str, Any]:
        if mode == "keyword":
            matches = corpus().search_chunks(q, limit=limit, arxiv_id=arxiv_id)
        else:
            matches = hybrid_search(
                corpus(), embedder(), q, limit=limit, arxiv_id=arxiv_id,
                max_per_paper=3 if arxiv_id is None else None,
            )
        return {
            "query": q,
            "mode": mode,
            "results": [match_to_dict(match) for match in matches],
        }

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Whether this corpus is usable, and what its recent runs did.

        Deliberately more than a 200: an empty corpus and a corpus whose last
        ingest failed both serve pages perfectly well, and both are the thing
        a check is asking about. The rates come from the run ledger rather
        than being recomputed, so they describe what actually happened.
        """
        store = corpus()
        counts = store.pipeline_stats(_fingerprint(), embedder().model_id)
        last = last_run_per_command(store)
        answered = [
            run for run in recent_runs(store, limit=200)
            if run["command"] in ("ask", "report") and run["ok"]
        ]
        cited = sum(run["outputs"].get("citations", 0) for run in answered)
        refused = sum(run["outputs"].get("rejected", 0) for run in answered)
        failing = sorted(name for name, run in last.items() if run["ok"] is False)
        return {
            "ok": bool(counts["chunks"]) and not failing,
            "corpus": {
                "papers": counts["papers"],
                "passages": counts["chunks"],
                "vectors": counts["vectors"],
                "pending": {
                    "downloads": counts["downloads_pending"],
                    "chunking": counts["chunking_pending"],
                    "embedding": counts["embedding_pending"],
                },
            },
            "last_run": {
                name: {
                    "at": run["started_at"], "ok": run["ok"],
                    "seconds": run["seconds"], "error": run["error"],
                }
                for name, run in sorted(last.items())
            },
            "failing": failing,
            "answers": {
                "runs": len(answered),
                "citations_verified": cited,
                "citations_rejected": refused,
                # The share of citations a model produced that did not survive
                # checking. It is a property of the model, not a fault: the
                # check exists because models paraphrase.
                "rejection_rate": round(refused / (cited + refused), 3) if cited + refused else None,
            },
        }

    @app.get("/api/report")
    def report(
        topic: str = Query(min_length=1, description="What the review is about"),
        sections: int = Query(4, ge=1, le=MAX_SECTIONS),
        passages: int = Query(6, ge=1, le=MAX_LIMIT),
        agents: bool = Query(
            True,
            description=(
                "Route each section by shape and escalate through strategies "
                "when the verifier rejects one. False runs a single rung."
            ),
        ),
    ) -> Any:
        """Write a sectioned review, streaming each section as it lands.

        A review is minutes of generation, so it streams for the same reason
        an answer does. The events are the ask route's vocabulary extended by
        one: `working` when a section starts, `section` when it is written and
        checked, then `report` carrying the assembled markdown and the ledger,
        then `done`.

        The orchestration runs on a thread of its own with its own corpus
        connection, and reaches this generator through a queue. It cannot run
        inline: the work is a blocking loop and the events have to leave the
        generator while it is still going.
        """
        from verbatim.orchestrator import orchestrate
        from verbatim.report import render_markdown

        events: queue.Queue = queue.Queue()
        model = answering()
        encoder_for_run = embedder()
        caller = routing()
        # This review's cost, not the server's since it started: usage
        # accumulates on a client that lives as long as the process.
        before = _usage_totals(model, caller)

        def run() -> None:
            # Its own store: this runs on a thread that has none, and a
            # connection may only be used where it was opened.
            try:
                with CorpusStore(database) as own:
                    built, ledger = orchestrate(
                        own,
                        encoder_for_run,
                        model,
                        lambda system, user: model.complete_text(system, user),
                        topic,
                        caller=caller if agents else None,
                        max_sections=sections,
                        per_section=passages,
                        ladder=None if agents else ("hybrid",),
                        on_section=lambda heading, done, total: events.put(
                            ("working", {"heading": heading, "done": done, "total": total})
                        ),
                        on_section_done=lambda section, entry: events.put(
                            ("section", _section_payload(section, entry))
                        ),
                    )
                    events.put((
                        "report",
                        {
                            "topic": built.topic,
                            "markdown": render_markdown(built, ledger=ledger),
                            "ledger_markdown": ledger.render_markdown(),
                            "escalations": ledger.escalations,
                            # What each section actually came to. The count of
                            # escalations alone cannot say this: a section can
                            # be a gap on its first rung, having escalated
                            # nowhere.
                            "outcomes": [entry.outcome for entry in ledger.sections],
                        },
                    ))
            # Deliberately broad: the sentinel below ends the stream whatever
            # happens, so an exception this did not name would close the
            # stream with a `done` event and present a half-written review as
            # a finished one.
            except Exception as error:  # noqa: BLE001 - see above
                events.put(("failed", {"message": f"{type(error).__name__}: {error}"}))
            finally:
                events.put(None)

        def stream() -> Any:
            worker = threading.Thread(target=run, name="report", daemon=True)
            worker.start()
            while True:
                item = events.get()
                if item is None:
                    break
                name, payload = item
                yield _event(name, payload)
            yield _event("done", {"cost": _cost_since(before, model, caller)})

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/papers/{arxiv_id}")
    def paper(arxiv_id: str) -> dict[str, Any]:
        found = corpus().papers_by_id([arxiv_id]).get(arxiv_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"No paper {arxiv_id} in this corpus")
        return found.to_dict()

    @app.get("/api/ask")
    def ask(
        q: str = Query(min_length=1, description="The question to answer"),
        arxiv_id: str | None = Query(None, description="Restrict to one paper version"),
        history: str | None = Query(
            None,
            description=(
                "Earlier turns as JSON [{question, answer}], oldest first. Used "
                "only to resolve what a follow-up refers to, before retrieval."
            ),
        ),
    ) -> Any:
        """Answer a question as a stream of server-sent events.

        A local model takes twenty seconds to write an answer, and a silent
        page for twenty seconds reads as a broken one. The draft streams as
        the model writes it; verification then runs on the finished reply
        exactly as it does everywhere else, and the final `answer` event
        replaces the draft with the verified text and its citations. Nothing
        is accepted differently for having been streamed.

        Retrieval happens here, before the stream begins, because SQLite
        connections belong to the thread that made them and the streaming
        generator may run on another.

        A follow-up is resolved against `history` into a question that stands
        alone, and only that resolved question reaches retrieval. The history
        never reaches the model that writes the answer: that model may state
        only what a retrieved passage supports, and prose from an earlier turn
        cannot be checked against any paper.
        """
        turns = _turns(history)
        asked = standalone_question(
            lambda system, user: answering().complete_text(system, user), turns, q
        ) if turns else q
        matches = hybrid_search(
            corpus(), embedder(), asked,
            limit=config.ASK_PASSAGES,
            arxiv_id=arxiv_id,
            max_per_paper=config.ASK_MAX_PER_PAPER if arxiv_id is None else None,
            use_graph=True,
        )
        papers = corpus().papers_by_id(sorted({m.chunk.arxiv_id for m in matches}))

        def events() -> Any:
            if asked != q:
                # Shown because the reader should be able to see, and dispute,
                # the question that was actually looked up.
                yield _event("resolved", {"question": asked})
            yield _event(
                "passages",
                {
                    "passages": [
                        {
                            "arxiv_id": m.chunk.arxiv_id,
                            "locator": render_locator(
                                m.chunk.source_kind, m.chunk.page_number, m.chunk.section_path
                            ),
                        }
                        for m in matches
                    ]
                },
            )
            if not matches:
                yield _event(
                    "answer",
                    Answer(
                        question=q,
                        text="No passages in the corpus match this question.",
                        citations=[], rejected=[], declined=True,
                    ).to_dict(),
                )
                yield _event("done", {})
                return
            collected: list[str] = []
            try:
                for fragment in answering().stream(SYSTEM_PROMPT, build_prompt(q, matches)):
                    collected.append(fragment)
                    yield _event("delta", {"text": fragment})
                payload = parse_line_protocol("".join(collected))
                if payload is None:
                    raise RuntimeError("The model did not answer in the expected format.")
                answer = answer_from_payload(q, matches, payload, papers)
                yield _event("answer", answer.to_dict())
                # After the answer, not before: the check costs a model call,
                # and the reader should not wait on a warning that usually
                # does not fire.
                subject = describe_subject(answering(), matches)
                closeness = subject_similarity(embedder(), q, subject)
                yield _event(
                    "subject",
                    {
                        "subject": subject,
                        "closeness": round(closeness, 3),
                        "warn": closeness < config.SUBJECT_WARN_BELOW,
                    },
                )
            # Named "failed" because EventSource fires its own "error" event
            # for connection trouble, and the page must tell them apart.
            except (RuntimeError, OSError) as error:
                yield _event("failed", {"message": str(error)})
            yield _event("done", {})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _fingerprint() -> str:
    from verbatim.chunking import chunker_fingerprint

    return chunker_fingerprint(config.CHUNK_SIZE, config.CHUNK_OVERLAP)


def serve(
    *,
    state: Path | None = None,
    embedding_model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the service. Bound to the loopback address by default.

    The corpus is 26 GB of other people's papers under mixed licences, so it is
    not served to the network unless someone says so explicitly.
    """
    import uvicorn

    uvicorn.run(create_app(state, embedding_model), host=host, port=port, log_level="info")


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>verbatim</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #fbfbfa; --fg: #1a1a18; --muted: #6b6b66;
    --line: #e0e0dc; --card: #ffffff; --accent: #2f5d8a;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #16171a; --fg: #e8e8e6; --muted: #9a9a95;
      --line: #2c2e33; --card: #1d1f23; --accent: #7fb0e0;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main { max-width: 54rem; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
  h1 { font-size: 1.35rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
  .sub { color: var(--muted); margin: 0 0 1.75rem; font-size: .9rem; }
  form { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1rem; }
  input[type=search] {
    flex: 1 1 22rem; padding: .6rem .75rem; font: inherit; color: var(--fg);
    background: var(--card); border: 1px solid var(--line); border-radius: 6px;
  }
  select, button {
    padding: .6rem .8rem; font: inherit; border-radius: 6px;
    border: 1px solid var(--line); background: var(--card); color: var(--fg);
  }
  button { background: var(--accent); color: #fff; border-color: transparent; cursor: pointer; }
  button[disabled] { opacity: .55; cursor: default; }
  .note { color: var(--muted); font-size: .85rem; margin: .75rem 0 1.5rem; }
  .warn { color: #b45309; font-size: .85rem; margin: .6rem 0 0; }
  .refs { margin-top: .75rem; border-top: 1px solid var(--line); padding-top: .6rem;
          font-size: .85rem; }
  .refs p { margin: .25rem 0; }
  .refs a { color: var(--accent); text-decoration: none; }
  .rejected { color: #b45309; }
  .sec { border-left: 2px solid var(--line); padding: 0 0 0 .85rem; margin: 1.1rem 0; }
  .sec h3 { font-size: .98rem; margin: 0 0 .4rem; line-height: 1.35; }
  .trace { color: var(--muted); font-size: .78rem; margin: .45rem 0 0;
           font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .pending { opacity: .55; }
  details { margin-top: 1rem; }
  summary { cursor: pointer; color: var(--muted); font-size: .85rem; }
  pre.md { white-space: pre-wrap; font-size: .8rem; background: var(--card);
           border: 1px solid var(--line); border-radius: 6px; padding: .8rem;
           overflow-x: auto; }
  article {
    background: var(--card); border: 1px solid var(--line); border-radius: 8px;
    padding: .9rem 1rem; margin-bottom: .75rem;
  }
  .head { display: flex; flex-wrap: wrap; gap: .5rem; align-items: baseline;
          font-size: .85rem; margin-bottom: .5rem; }
  .head a { color: var(--accent); text-decoration: none; font-weight: 600; }
  .head a:hover { text-decoration: underline; }
  .meta { color: var(--muted); }
  .text { margin: 0; white-space: pre-wrap; }
  mark { background: transparent; color: inherit; font-weight: 700; }
  footer { color: var(--muted); font-size: .8rem; margin-top: 2rem;
           border-top: 1px solid var(--line); padding-top: 1rem; }

  /* The conversation. Questions and answers stack, so a reader can see what
     was asked three turns ago and what it was answered with. */
  #thread { display: flex; flex-direction: column; gap: .9rem; margin: 1rem 0; }
  .turn { display: flex; }
  .turn.you { justify-content: flex-end; }
  .bubble {
    max-width: 46rem; border-radius: 10px; padding: .75rem .95rem;
    border: 1px solid var(--line); background: var(--card);
  }
  .turn.you .bubble {
    background: var(--accent); color: #fff; border-color: transparent;
    max-width: 34rem;
  }
  .who { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
         color: var(--muted); margin-bottom: .3rem; }
  .turn.you .who { color: rgba(255,255,255,.75); }
  .thinking { color: var(--muted); font-size: .85rem; }
  .resolved { color: var(--muted); font-size: .8rem; font-style: italic;
              margin: 0 0 .45rem; }
  .askbar { position: sticky; bottom: 0; background: var(--bg);
            padding: .75rem 0 .25rem; border-top: 1px solid var(--line); }
  .askbar form { margin: 0; }
  .cursor::after { content: "\u258c"; animation: blink 1s step-end infinite; }
  @keyframes blink { 50% { opacity: 0; } }
</style>
</head>
<body>
<main>
  <h1>verbatim</h1>
  <p class="sub">Citation-checked search over a local arXiv corpus.</p>

  <form id="search">
    <input type="search" id="q" placeholder="Search the corpus" autofocus
           aria-label="Search the corpus">
    <select id="mode" aria-label="Retrieval mode">
      <option value="hybrid">hybrid</option>
      <option value="keyword">keyword</option>
    </select>
    <button type="submit" id="go">Search</button>
  </form>

  <p class="note" id="status"></p>
  <div id="results"></div>

  <div id="thread"></div>

  <div class="askbar">
    <form id="askform">
      <input type="search" id="aq" autocomplete="off"
             placeholder="Ask a question, answered with verified citations"
             aria-label="Ask a question">
      <button type="submit" id="askgo">Ask</button>
      <button type="button" id="askreset" hidden>New topic</button>
    </form>
    <p class="note" id="askstatus"></p>
  </div>

  <form id="reportform">
    <input type="search" id="rt" placeholder="Or write a review on a topic"
           aria-label="Topic to review">
    <label class="meta"><input type="checkbox" id="ragents" checked> orchestrate</label>
    <button type="submit" id="reportgo">Review</button>
  </form>
  <p class="note" id="reportstatus"></p>
  <div id="reportout"></div>

  <footer id="stats"></footer>
</main>
<script>
const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => (
    {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]
  ));
}

// The API returns FTS5 snippets marked with [ ]; render those as emphasis
// rather than letting the brackets read as part of the paper.
function renderSnippet(text) {
  return escapeHtml(text).replace(/\\[([^\\]]+)\\]/g, "<mark>$1</mark>");
}

function renderResult(hit) {
  const licence = hit.license_url
    ? '<a href="' + escapeHtml(hit.license_url) + '">licence</a>'
    : "no licence reported";
  return [
    "<article>",
    '  <div class="head">',
    '    <a href="', escapeHtml(hit.source_url), '">arXiv:', escapeHtml(hit.arxiv_id), "</a>",
    '    <span class="meta">', escapeHtml(hit.locator), "</span>",
    '    <span class="meta">score ', escapeHtml(hit.score), "</span>",
    '    <span class="meta">', escapeHtml(hit.source_kind), "</span>",
    '    <span class="meta">', licence, "</span>",
    "  </div>",
    '  <p class="text">', renderSnippet(hit.snippet || hit.text), "</p>",
    "</article>",
  ].join("");
}

async function runSearch(event) {
  event.preventDefault();
  const query = $("q").value.trim();
  if (!query) return;
  $("go").disabled = true;
  $("status").textContent = "Searching...";
  $("results").innerHTML = "";
  try {
    const url = "/api/search?q=" + encodeURIComponent(query)
      + "&mode=" + encodeURIComponent($("mode").value);
    const response = await fetch(url);
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || ("HTTP " + response.status));
    }
    const data = await response.json();
    $("status").textContent = data.results.length
      ? data.results.length + " passages, " + data.mode + " retrieval"
      : "Nothing matched. Keyword search matches whole words: try field* for fields.";
    $("results").innerHTML = data.results.map(renderResult).join("");
  } catch (error) {
    $("status").textContent = "Search failed: " + error.message;
  } finally {
    $("go").disabled = false;
  }
}

async function loadStats() {
  try {
    const counts = await (await fetch("/api/stats")).json();
    $("stats").textContent =
      counts.papers.toLocaleString() + " papers, "
      + counts.chunks.toLocaleString() + " passages, "
      + counts.vectors.toLocaleString() + " vectors."
      + (counts.vectors === 0
          ? " No vectors yet, so hybrid retrieval falls back to keywords."
          : "");
  } catch (error) {
    $("stats").textContent = "Corpus statistics unavailable.";
  }
}

function locatorOf(citation) {
  if (citation.source_kind !== "html") return "p. " + citation.page_number;
  const parts = (citation.section_path || "").split(" > ");
  const heading = parts[parts.length - 1].trim();
  return heading ? "section " + heading : "section " + citation.page_number;
}

function renderReference(citation, index) {
  return '<p>[' + (index + 1) + '] <a href="' + escapeHtml(citation.source_url) + '">'
    + escapeHtml(citation.title) + "</a>"
    + '<span class="meta"> — arXiv:' + escapeHtml(citation.arxiv_id)
    + ", " + escapeHtml(locatorOf(citation)) + "</span></p>";
}

let asking = null;
// The conversation lives in the page, not on the server: it exists so a
// follow-up can be resolved before retrieval, and it is never evidence.
let thread = [];

function resetThread() {
  thread = [];
  $("thread").innerHTML = "";
  $("askreset").hidden = true;
  $("askstatus").textContent = "";
  $("aq").focus();
}

function bubble(who, label) {
  const turn = document.createElement("div");
  turn.className = "turn " + who;
  const box = document.createElement("div");
  box.className = "bubble";
  const tag = document.createElement("div");
  tag.className = "who";
  tag.textContent = label;
  box.appendChild(tag);
  turn.appendChild(box);
  $("thread").appendChild(turn);
  turn.scrollIntoView({ block: "end", behavior: "smooth" });
  return box;
}

function runAsk(event) {
  event.preventDefault();
  const question = $("aq").value.trim();
  if (!question) return;
  if (asking) asking.close();
  $("askgo").disabled = true;
  $("aq").value = "";

  // Your question stays on screen, above the answer to it.
  const yours = bubble("you", "You");
  const text = document.createElement("p");
  text.className = "text";
  text.textContent = question;
  yours.appendChild(text);

  const reply = bubble("agent", "verbatim");
  const resolved = document.createElement("p");
  resolved.className = "resolved";
  resolved.hidden = true;
  const body = document.createElement("p");
  body.className = "text thinking";
  body.textContent = "Retrieving passages\u2026";
  const subject = document.createElement("p");
  subject.className = "warn";
  subject.hidden = true;
  const refs = document.createElement("div");
  refs.className = "refs";
  refs.hidden = true;
  reply.append(resolved, body, subject, refs);

  let draft = "";
  const history = thread.length
    ? "&history=" + encodeURIComponent(JSON.stringify(thread))
    : "";
  asking = new EventSource(
    "/api/ask?q=" + encodeURIComponent(question) + history
  );

  asking.addEventListener("resolved", (e) => {
    resolved.hidden = false;
    resolved.textContent = "Read as: " + JSON.parse(e.data).question;
  });
  asking.addEventListener("passages", (e) => {
    const got = JSON.parse(e.data).passages;
    const papers = new Set(got.map((p) => p.arxiv_id)).size;
    if (got.length) {
      body.textContent =
        "Reading " + got.length + " passages from " + papers + " papers\u2026";
    }
  });
  asking.addEventListener("delta", (e) => {
    draft += JSON.parse(e.data).text;
    body.className = "text cursor";
    body.textContent =
      draft.split(/\\n\\s*CITE\\s*\\[/)[0].replace(/^[\\s\\S]*?ANSWER:\\s*/, "");
    reply.scrollIntoView({ block: "end" });
  });
  asking.addEventListener("answer", (e) => {
    const answer = JSON.parse(e.data);
    body.className = "text";
    body.textContent = answer.text;
    thread.push({ question: question, answer: answer.text });
    if (thread.length > 6) thread.shift();
    $("askreset").hidden = false;
    if (answer.declined) return;
    let markup = answer.citations.map(renderReference).join("");
    if (answer.rejected.length) {
      markup += '<p class="rejected">' + answer.rejected.length
        + " citation(s) failed verification and are marked [unverified].</p>";
    }
    if (markup) { refs.innerHTML = markup; refs.hidden = false; }
  });
  asking.addEventListener("subject", (e) => {
    const check = JSON.parse(e.data);
    if (check.warn) {
      subject.hidden = false;
      subject.textContent = "These passages are about: " + check.subject
        + " \u2014 which may not be what you asked ("
        + check.closeness.toFixed(2) + " similarity).";
    }
  });
  asking.addEventListener("failed", (e) => {
    body.className = "text warn";
    body.textContent = "Answering failed: " + JSON.parse(e.data).message;
  });
  asking.addEventListener("done", () => {
    asking.close();
    asking = null;
    $("askgo").disabled = false;
    $("aq").focus();
  });
  asking.onerror = () => {
    if (!asking) return;
    asking.close();
    asking = null;
    $("askgo").disabled = false;
    if (!draft) {
      body.className = "text warn";
      body.textContent = "Connection lost while answering.";
    }
  };
}

function renderSectionCard(payload) {
  const cited = payload.citations.map((c, i) =>
    '<p>[' + (i + 1) + '] <a href="' + escapeHtml(c.source_url) + '">'
    + escapeHtml(c.title) + '</a><span class="meta"> — arXiv:'
    + escapeHtml(c.arxiv_id) + ", " + escapeHtml(c.locator) + "</span></p>").join("");
  const trace = payload.attempts.map((a) => {
    if (a.error) return "* " + a.strategy + ": failed — " + escapeHtml(a.error);
    const route = a.route ? " via " + escapeHtml(a.route) : "";
    return "* " + escapeHtml(a.strategy) + route + ": " + a.verified + " verified, "
      + a.rejected + " rejected \u2192 " + escapeHtml(a.verdict);
  }).join("<br>");
  return [
    '<div class="sec">',
    "  <h3>", escapeHtml(payload.heading), "</h3>",
    '  <p class="text">', escapeHtml(payload.text), "</p>",
    payload.evidence.length
      ? '<p class="meta">' + escapeHtml(payload.evidence.join(" ")) + "</p>" : "",
    cited ? '<div class="refs">' + cited + "</div>" : "",
    '  <p class="trace">', escapeHtml(payload.shape), " \u2014 ",
    escapeHtml(payload.outcome), "<br>", trace, "</p>",
    "</div>",
  ].join("");
}

let reviewing = null;

function runReport(event) {
  event.preventDefault();
  const topic = $("rt").value.trim();
  if (!topic) return;
  if (reviewing) reviewing.close();
  $("reportgo").disabled = true;
  $("reportout").innerHTML = "";
  $("reportstatus").textContent = "Planning the sections\u2026";

  const url = "/api/report?topic=" + encodeURIComponent(topic)
    + "&agents=" + ($("ragents").checked ? "true" : "false");
  reviewing = new EventSource(url);

  reviewing.addEventListener("working", (e) => {
    const at = JSON.parse(e.data);
    $("reportstatus").textContent =
      "Section " + at.done + " of " + at.total + ": " + at.heading;
  });
  reviewing.addEventListener("section", (e) => {
    $("reportout").insertAdjacentHTML("beforeend", renderSectionCard(JSON.parse(e.data)));
  });
  reviewing.addEventListener("report", (e) => {
    const done = JSON.parse(e.data);
    const outcomes = done.outcomes || [];
    const accepted = outcomes.filter((o) => o.indexOf("accepted") === 0).length;
    let note = "Review complete. " + accepted + " of " + outcomes.length
      + " sections accepted";
    note += done.escalations
      ? ", " + done.escalations + " needed another strategy."
      : ", none needed a second strategy.";
    $("reportstatus").textContent = note;
    $("reportout").insertAdjacentHTML("beforeend",
      "<details><summary>The whole review as markdown</summary>"
      + '<pre class="md">' + escapeHtml(done.markdown) + "</pre></details>");
  });
  reviewing.addEventListener("failed", (e) => {
    $("reportstatus").textContent = "Review failed: " + JSON.parse(e.data).message;
  });
  reviewing.addEventListener("done", (e) => {
    const cost = JSON.parse(e.data).cost;
    if (cost) $("reportstatus").textContent += "  (" + cost + ")";
    reviewing.close();
    reviewing = null;
    $("reportgo").disabled = false;
  });
  reviewing.onerror = () => {
    if (!reviewing) return;
    reviewing.close();
    reviewing = null;
    $("reportstatus").textContent = "Connection lost while writing the review.";
    $("reportgo").disabled = false;
  };
}

$("askform").addEventListener("submit", runAsk);
$("askreset").addEventListener("click", resetThread);
$("reportform").addEventListener("submit", runReport);
loadStats();
</script>
</body>
</html>
"""
