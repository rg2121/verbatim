"""HTTP API and web page over the corpus, calling the same functions as the CLI.

The page is one self-contained HTML file that makes no external requests.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
import queue
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse

from verbatim import config
from verbatim.answers import (
    SYSTEM_PROMPT,
    Answer,
    OllamaCompletion,
    OllamaToolCaller,
    answer_from_payload,
    build_prompt,
    parse_line_protocol,
    subject_check,
)
from verbatim.chunking import chunker_fingerprint
from verbatim.citations import render_locator
from verbatim.conversation import Turn, standalone_question
from verbatim.corpus import ChunkMatch, CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.orchestrator import orchestrate, work_section
from verbatim.report import render_markdown
from verbatim.retrieval import hybrid_search
from verbatim.runs import last_run_per_command, recent_runs

MAX_LIMIT = 50
# Reviews are the costliest request, so the ceiling is low.
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
        # The full citation, so the page renders review and answer sources alike.
        "citations": [citation.to_dict() for citation in section.answer.citations],
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
    """What these clients have counted so far, or None if none counts."""
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
    """This run's cost, as the difference from a snapshot; served clients count for the whole process."""
    after = _usage_totals(*models)
    if before is None or after is None:
        return "cost not reported"
    calls, prompt, output, seconds = (a - b for a, b in zip(after, before, strict=True))
    return (
        f"{calls} model call{'s' if calls != 1 else ''}, "
        f"{prompt:,} tokens in, {output:,} out, {seconds:.1f}s in the model"
    )


def _turns(history: str | None) -> list[Turn]:
    """Read the client's conversation; anything malformed means no history."""
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
    """Render one hit with its locator, licence and source URL, so it can be checked."""
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
    """Build the application; tests inject `completion` and `encoder`.

    Models load lazily, so serving the search page never needs one.
    """

    database = state or config.STATE_PATH
    model_name = embedding_model or config.EMBEDDING_MODEL
    resources: dict[str, Any] = {}
    # Guards the lazily built models, which request threads share.
    building = threading.Lock()
    # SQLite connections are per thread, and FastAPI serves requests from a
    # thread pool, so each thread opens its own.
    stores = threading.local()

    @asynccontextmanager
    async def lifespan(_: Any) -> Any:
        yield
        # Only this thread's connection can be closed here; the rest close at exit.
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
                    resources["completion"] = OllamaCompletion(config.OLLAMA_MODEL)
            return resources["completion"]

    def routing() -> Any:
        """The tool-choosing model for the orchestrator's last rung."""
        with building:
            if "caller" not in resources:
                if tool_caller is not None:
                    resources["caller"] = tool_caller
                elif completion is not None:
                    # An injected model is the whole backend: never build an
                    # Ollama caller behind it.
                    resources["caller"] = (
                        completion if hasattr(completion, "choose_tool") else None
                    )
                else:
                    resources["caller"] = OllamaToolCaller(config.OLLAMA_MODEL)
            return resources["caller"]

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return PAGE

    @app.get("/api/stats")
    def stats() -> dict[str, int]:
        # The model is named, not loaded, so this stays fast.
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
                # The default graph leg, as in every other front end.
                use_graph=True,
            )
        return {
            "query": q,
            "mode": mode,
            "results": [match_to_dict(match) for match in matches],
        }

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Whether this corpus is usable, and what its recent runs did.

        More than a 200: an empty corpus or a failed ingest still serves pages.
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
                # Share of the model's citations that failed verification.
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
        """Write a sectioned review, streaming each section as it is written.

        Events: `working` when a section starts, `section` when it is checked,
        `report` with the markdown and ledger, then `done`. The work runs on
        its own thread and corpus connection, feeding the stream through a queue.
        """
        events: queue.Queue = queue.Queue()
        model = answering()
        encoder_for_run = embedder()
        caller = routing()
        # This review's cost, not the server's lifetime total.
        before = _usage_totals(model, caller)

        def run() -> None:
            # Its own connection: this runs on a new thread.
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
                            # Each section's outcome; escalations alone cannot show it.
                            "outcomes": [entry.outcome for entry in ledger.sections],
                        },
                    ))
            # Broad on purpose: otherwise a failure would end the stream with
            # `done` and a half-written review would look finished.
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

    def _deep_answer(asked: str, original: str) -> Any:
        """Answer by climbing the ladder, streaming each rung rather than tokens.

        A draft that a retry replaces must never be shown, so the answer is sent
        only once a rung is accepted. Runs on its own thread, as `/api/report` does.
        """
        events: queue.Queue = queue.Queue()
        model = answering()
        encoder_for_run = embedder()
        caller = routing()
        before = _usage_totals(model, caller)

        def run() -> None:
            try:
                with CorpusStore(database) as own:
                    section, ledger = work_section(
                        own, encoder_for_run, model, asked,
                        caller=caller,
                        per_section=config.ASK_PASSAGES,
                        max_per_paper=config.ASK_MAX_PER_PAPER,
                        on_attempt=lambda strategy, rung, rungs: events.put(
                            ("working", {"strategy": strategy, "rung": rung,
                                         "rungs": rungs})
                        ),
                    )
                    answer = section.answer
                    events.put(("answer", answer.to_dict()))
                    events.put((
                        "ladder",
                        {
                            "shape": ledger.shape,
                            "outcome": ledger.outcome,
                            "attempts": [
                                {
                                    "strategy": a.strategy,
                                    "verdict": a.verdict,
                                    "passages": a.passages,
                                    "verified": a.verified,
                                    "closeness": a.closeness,
                                    "error": a.error,
                                }
                                for a in ledger.attempts
                            ],
                        },
                    ))
                    events.put((
                        "subject",
                        {
                            "subject": section.subject,
                            "closeness": (
                                None if section.subject_closeness is None
                                else round(section.subject_closeness, 3)
                            ),
                            # The ladder already retried on a low subject score,
                            # so the accepted answer is not warned about.
                            "warn": False,
                        },
                    ))
            # Broad on purpose, as in `/api/report`.
            except Exception as error:  # noqa: BLE001 - see above
                events.put(("failed", {"message": str(error)}))
            finally:
                events.put(None)

        def stream() -> Any:
            if asked != original:
                yield _event("resolved", {"question": asked})
            worker = threading.Thread(target=run, name="ask-deep", daemon=True)
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
        deep: bool = Query(
            False,
            description=(
                "Escalate when an answer has no verified citation or drifts "
                "off-subject: route by question shape and climb the retrieval "
                "ladder. Streams rungs instead of tokens, because an attempt "
                "that gets retried must not have been shown as the answer."
            ),
        ),
    ) -> Any:
        """Answer a question as a stream of server-sent events.

        The draft streams as the model writes; the final `answer` event replaces
        it with the verified text and citations. A follow-up is first resolved
        against `history` into a standalone question, shown to the reader; the
        history never reaches the model that writes the answer.
        """
        if deep and arxiv_id is not None:
            # Refused rather than ignored, as in the CLI.
            raise HTTPException(
                status_code=400,
                detail="arxiv_id has no effect with deep: each rung of the "
                "ladder sets its own retrieval. Drop one of the two.",
            )
        turns = _turns(history)
        asked = standalone_question(
            lambda system, user: answering().complete_text(system, user), turns, q
        ) if turns else q
        if deep:
            return _deep_answer(asked, q)
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
                # Show the question actually looked up, so the reader can dispute it.
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
                        question=asked,
                        text="No passages in the corpus match this question.",
                        citations=[], rejected=[], declined=True,
                    ).to_dict(),
                )
                yield _event("done", {})
                return
            collected: list[str] = []
            try:
                for fragment in answering().stream(SYSTEM_PROMPT, build_prompt(asked, matches)):
                    collected.append(fragment)
                    yield _event("delta", {"text": fragment})
                payload = parse_line_protocol("".join(collected))
                if payload is None:
                    raise RuntimeError("The model did not answer in the expected format.")
                answer = answer_from_payload(asked, matches, payload, papers)
                yield _event("answer", answer.to_dict())
                # After the answer: the check costs a model call and rarely warns.
                subject, closeness = subject_check(answering(), embedder(), asked, matches)
                # Always sent, so the page stops waiting; no warning without a subject.
                yield _event(
                    "subject",
                    {
                        "subject": subject,
                        "closeness": None if closeness is None else round(closeness, 3),
                        "warn": closeness is not None
                        and closeness < config.SUBJECT_WARN_BELOW,
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
    return chunker_fingerprint(config.CHUNK_SIZE, config.CHUNK_OVERLAP)


def serve(
    *,
    state: Path | None = None,
    embedding_model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the service, bound to loopback unless `host` says otherwise."""
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
    color-scheme: dark;
    --bg: #0e0f11;
    --surface: #15171a;
    --raised: #1b1e22;
    --line: #23272c;
    --ink: #e7e9ec;
    --muted: #8b939c;
    --faint: #626a73;
    --accent: #8ab4f8;
    --warn: #e0b877;
    --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    --serif: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.6 var(--sans);
    -webkit-font-smoothing: antialiased;
  }
  main { max-width: 46rem; margin: 0 auto; padding: 4rem 1.5rem 6rem; }

  header { margin-bottom: 2.5rem; }
  h1 { font: 500 1.4rem/1 var(--sans); letter-spacing: -.01em; margin: 0 0 .5rem; }
  header p { margin: 0; color: var(--muted); font-size: .92rem; }

  .tabs { display: flex; gap: .25rem; margin-bottom: 1.5rem; }
  .tabs button {
    font: inherit; font-size: .9rem; color: var(--muted); background: none;
    border: 0; border-radius: 7px; padding: .4rem .8rem; cursor: pointer;
  }
  .tabs button:hover { color: var(--ink); }
  .tabs button[aria-selected="true"] { color: var(--ink); background: var(--surface); }

  form { display: flex; gap: .5rem; align-items: center; }
  input[type=search], input[type=text] {
    flex: 1; min-width: 0; font: inherit; color: var(--ink);
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 10px; padding: .7rem .9rem;
  }
  input::placeholder { color: var(--faint); }
  input:focus-visible, button:focus-visible, summary:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 2px;
  }
  form button {
    font: inherit; border: 1px solid var(--line); background: var(--raised);
    color: var(--ink); border-radius: 10px; padding: .7rem 1.1rem; cursor: pointer;
  }
  form button:hover { border-color: var(--accent); color: var(--accent); }
  form button[disabled] { opacity: .5; cursor: default; }
  select {
    font: inherit; color: var(--muted); background: var(--surface);
    border: 1px solid var(--line); border-radius: 10px; padding: .7rem .6rem;
  }

  .opt { display: flex; gap: 1rem; align-items: center;
         margin-top: .7rem; color: var(--faint); font-size: .85rem; }
  .opt label { display: flex; gap: .4rem; align-items: center; cursor: pointer; }
  .note { color: var(--muted); font-size: .85rem; margin: .9rem 0 0; min-height: 1.2em; }

  /* conversation */
  #thread { display: flex; flex-direction: column; gap: 3rem; margin: 0 0 1.5rem; }
  #thread:empty { margin: 0; }
  .turn .who {
    font: 500 .7rem/1 var(--sans); letter-spacing: .09em; text-transform: uppercase;
    color: var(--faint); margin: 0 0 .5rem;
  }
  .turn .asked {
    font: 500 1.15rem/1.45 var(--sans); margin: 0 0 1rem; text-wrap: balance;
  }
  /* The composer stays within reach, so a follow-up is always one line away. */
  .composer {
    position: sticky; bottom: 0; padding: 1rem 0 .8rem; background: var(--bg);
    box-shadow: 0 -1.5rem 1rem -1rem var(--bg);
  }
  #thread:not(:empty) + .composer { border-top: 1px solid var(--line); }
  .hint { color: var(--faint); font-size: .82rem; margin: .7rem 0 0; }
  .resolved { color: var(--faint); font-size: .85rem; margin: -.6rem 0 1rem; }
  .resolved button {
    font: inherit; color: var(--accent); background: none; border: 0;
    padding: 0; cursor: pointer; text-decoration: underline;
  }
  .answer { font: 1.02rem/1.75 var(--serif); }
  .answer p { margin: 0 0 .9rem; }
  .answer ul { margin: 0 0 .9rem; padding-left: 1.1rem; font: .92rem/1.6 var(--sans); }
  .answer li { margin: 0 0 .35rem; }
  .thinking { color: var(--muted); font-family: var(--sans); font-size: .95rem; }
  .cursor::after { content: "▌"; animation: blink 1.1s step-end infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  a.cite {
    font: 500 .7rem/1 var(--sans); vertical-align: super; color: var(--accent);
    text-decoration: none; padding: 0 .12rem;
  }
  a.cite:hover { text-decoration: underline; }
  .unverified {
    font: .72rem/1 var(--sans); vertical-align: super; color: var(--warn);
  }

  .sources { margin-top: 1.4rem; border-top: 1px solid var(--line); padding-top: 1rem; }
  .sources h3 {
    font: 500 .72rem/1 var(--sans); letter-spacing: .09em; text-transform: uppercase;
    color: var(--faint); margin: 0 0 .8rem;
  }
  .source { display: flex; gap: .6rem; margin-bottom: .8rem; font-size: .9rem; }
  .source .n { color: var(--faint); font-family: var(--mono); font-size: .78rem; padding-top: .15rem; }
  .source a { color: var(--ink); text-decoration: none; }
  .source a:hover { color: var(--accent); }
  .source .where { color: var(--faint); font-size: .82rem; }
  .source blockquote {
    margin: .4rem 0 0; padding-left: .7rem; border-left: 2px solid var(--line);
    color: var(--muted); font: .88rem/1.6 var(--serif);
  }
  .caution { color: var(--warn); font-size: .87rem; margin: 1rem 0 0; }

  details.how { margin-top: 1rem; }
  details.how > summary {
    cursor: pointer; color: var(--faint); font-size: .82rem; list-style: none;
  }
  details.how > summary::-webkit-details-marker { display: none; }
  details.how > summary::before { content: "› "; }
  details.how[open] > summary::before { content: "˅ "; }
  details.how .body {
    margin-top: .6rem; color: var(--muted); font-size: .85rem;
    border-left: 1px solid var(--line); padding-left: .8rem;
  }
  details.how .body p { margin: 0 0 .35rem; }

  /* search results */
  .hit { padding: 1rem 0; border-bottom: 1px solid var(--line); }
  .hit:last-child { border-bottom: 0; }
  .hit .head { display: flex; flex-wrap: wrap; gap: .6rem; align-items: baseline;
               font-size: .82rem; color: var(--faint); margin-bottom: .45rem; }
  .hit .head a { color: var(--accent); text-decoration: none; }
  .hit .text { margin: 0; color: var(--muted); font: .95rem/1.65 var(--serif); }
  .hit mark { background: none; color: var(--ink); font-weight: 600; }

  /* review */
  .section { margin: 2rem 0; }
  .section h3 { font: 500 1.05rem/1.45 var(--sans); margin: 0 0 .6rem; text-wrap: balance; }
  pre.md {
    white-space: pre-wrap; font: .82rem/1.6 var(--mono); background: var(--surface);
    border: 1px solid var(--line); border-radius: 10px; padding: 1rem; overflow-x: auto;
  }

  footer {
    margin-top: 5rem; padding-top: 1.2rem; border-top: 1px solid var(--line);
    color: var(--faint); font-size: .82rem;
  }
  @media (max-width: 560px) {
    main { padding: 2.5rem 1.1rem 4rem; }
    form { flex-wrap: wrap; }
    form button { flex: 1; }
  }
</style>
</head>
<body>
<main>
  <header>
    <h1>Verbatim</h1>
    <p>Ask a corpus of papers. Every quote is checked against the passage it comes from.</p>
  </header>

  <nav class="tabs" role="tablist">
    <button id="tabask" type="button" role="tab" aria-selected="true">Ask</button>
    <button id="tabsearch" type="button" role="tab" aria-selected="false">Search</button>
    <button id="tabreview" type="button" role="tab" aria-selected="false">Review</button>
  </nav>

  <section id="panelask">
    <div id="thread"></div>
    <div class="composer">
      <form id="askform">
        <input type="search" id="aq" autocomplete="off" autofocus
               placeholder="Ask a question" aria-label="Ask a question">
        <button type="submit" id="askgo">Ask</button>
      </form>
      <div class="opt">
        <label><input type="checkbox" id="askdeep"> Try harder if nothing checks out</label>
        <button type="button" id="askreset" hidden
                style="background:none;border:0;color:inherit;font:inherit;cursor:pointer;padding:0">
          Start over
        </button>
      </div>
      <p class="hint" id="askhint">
        Ask follow-ups in your own words - "why does it do that?" is resolved
        against what you just asked, and you are shown how it was read.
      </p>
      <p class="note" id="askstatus"></p>
    </div>
  </section>

  <section id="panelsearch" hidden>
    <form id="search">
      <input type="search" id="q" placeholder="Search passages" aria-label="Search passages">
      <select id="mode" aria-label="How to search">
        <option value="hybrid">Meaning and words</option>
        <option value="keyword">Exact words</option>
      </select>
      <button type="submit" id="go">Search</button>
    </form>
    <p class="note" id="status"></p>
    <div id="results"></div>
  </section>

  <section id="panelreview" hidden>
    <form id="reportform">
      <input type="search" id="rt" placeholder="A topic to review" aria-label="Topic to review">
      <button type="submit" id="reportgo">Write</button>
    </form>
    <div class="opt">
      <label><input type="checkbox" id="ragents" checked> Retry sections that come back thin</label>
    </div>
    <p class="note" id="reportstatus"></p>
    <div id="reportout"></div>
  </section>

  <footer id="stats"></footer>
</main>
<script>
const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => (
    {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]
  ));
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

// -- tabs -------------------------------------------------------------
const PANELS = {tabask: "panelask", tabsearch: "panelsearch", tabreview: "panelreview"};
function show(tab) {
  for (const [button, panel] of Object.entries(PANELS)) {
    const on = button === tab;
    $(button).setAttribute("aria-selected", on ? "true" : "false");
    $(panel).hidden = !on;
  }
}
for (const tab of Object.keys(PANELS)) $(tab).onclick = () => show(tab);

// -- answers ----------------------------------------------------------
// Markers point at the sources below; a failed quote says so.
function renderAnswer(text, turn) {
  const marked = escapeHtml(text)
    .replace(/\\[unverified\\]/g, '<span class="unverified" title="this quote could not be checked">unverified</span>')
    .replace(/\\[(\\d+)\\]/g, (m, n) => '<a class="cite" href="#ref-' + turn + '-' + n + '">' + n + '</a>');
  return marked
    .split(/\\n\\s*\\n/)
    .map((para) => {
      const lines = para.split(/\\n/).filter((line) => line.trim());
      // A catalogue listing (recency) is shown as rows.
      if (lines.length && lines.every((line) => /^\\s*- /.test(line))) {
        return "<ul>" + lines.map((line) => "<li>" + line.replace(/^\\s*- /, "") + "</li>").join("") + "</ul>";
      }
      return "<p>" + para.replace(/\\n/g, " ") + "</p>";
    })
    .join("");
}

function whereIn(citation) {
  if (citation.source_kind !== "html") return "page " + citation.page_number;
  const parts = (citation.section_path || "").split(" > ");
  const heading = parts[parts.length - 1].trim();
  return heading ? heading : "section " + citation.page_number;
}

// Surnames, with "et al." past three, as in the CLI's bibliography.
function formatAuthors(authors) {
  const surname = (name) => name.trim().split(/\\s+/).pop();
  if (authors.length > 3) return surname(authors[0]) + " et al.";
  return authors.map(surname).join(", ");
}

function renderSources(citations, turn) {
  if (!citations.length) return "";
  const rows = citations.map((c, i) => [
    '<div class="source" id="ref-', turn, '-', i + 1, '">',
    '  <span class="n">', i + 1, "</span>",
    "  <div>",
    '    <a href="', escapeHtml(c.source_url), '">', escapeHtml(c.title), "</a>",
    '    <div class="where">', escapeHtml(c.authors && c.authors.length ? formatAuthors(c.authors) + " · " : ""),
    escapeHtml(c.arxiv_id), " · ", escapeHtml(whereIn(c)), "</div>",
    c.quoted_text ? "<blockquote>" + escapeHtml(c.quoted_text) + "</blockquote>" : "",
    "  </div>",
    "</div>",
  ].join(""));
  return '<div class="sources"><h3>Sources</h3>' + rows.join("") + "</div>";
}

// Plain words for the retrieval strategies.
const PLAIN = {
  hybrid: "the usual search", titles: "paper titles", compare: "a comparison search",
  agent: "letting the model pick the tool",
};
const SHAPE = {
  definitional: "a question about what something is", comparison: "a comparison",
  recency: "a question about what is new", open: "an open question",
};
function plain(name) { return PLAIN[name] || name; }

function howBox(lines) {
  const box = el("details", "how");
  box.appendChild(el("summary", null, "How this answer was found"));
  const body = el("div", "body");
  for (const line of lines) body.appendChild(el("p", null, line));
  box.appendChild(body);
  return box;
}

let asking = null;
let thread = [];
let turns = 0;
// Set when a follow-up's reading is edited: that question is sent as written.
let exact = false;
const HINT = $("askhint").textContent;

function resetThread() {
  // Close an answer still arriving, or it would land in the new conversation.
  if (asking) {
    asking.close();
    asking = null;
    $("askgo").disabled = false;
  }
  exact = false;
  $("askhint").textContent = HINT;
  thread = [];
  $("thread").replaceChildren();
  $("askreset").hidden = true;
  $("askstatus").textContent = "";
  $("aq").placeholder = "Ask a question";
  $("aq").focus();
}

function runAsk(event) {
  event.preventDefault();
  const question = $("aq").value.trim();
  if (!question) return;
  if (asking) asking.close();
  $("askgo").disabled = true;
  $("aq").value = "";
  const turn = ++turns;

  const block = el("article", "turn");
  block.appendChild(el("p", "who", thread.length ? "You, following up" : "You"));
  block.appendChild(el("p", "asked", question));
  const resolved = el("p", "resolved");
  resolved.hidden = true;
  const body = el("div", "answer thinking");
  body.textContent = "Looking through the corpus…";
  const caution = el("p", "caution");
  caution.hidden = true;
  const sources = el("div");
  block.append(resolved, body, caution, sources);
  $("thread").appendChild(block);
  block.scrollIntoView({block: "start", behavior: "smooth"});

  const how = [];
  let draft = "";
  const history = thread.length && !exact
    ? "&history=" + encodeURIComponent(JSON.stringify(thread))
    : "";
  // "Asked exactly as written" applied to one question only.
  exact = false;
  $("askhint").textContent = HINT;
  const deep = $("askdeep").checked;
  asking = new EventSource(
    "/api/ask?q=" + encodeURIComponent(question) + history + (deep ? "&deep=true" : "")
  );

  asking.addEventListener("resolved", (e) => {
    // The rewrite is shown and can be corrected, since a follow-up depends on it.
    const readAs = JSON.parse(e.data).question;
    resolved.hidden = false;
    resolved.replaceChildren(document.createTextNode("Read as: " + readAs + " · "));
    const fix = el("button", null, "ask it differently");
    fix.type = "button";
    fix.onclick = () => {
      $("aq").value = readAs;
      exact = true;
      $("askhint").textContent =
        "This one is asked exactly as written - the conversation is left out of it.";
      $("aq").focus();
    };
    resolved.appendChild(fix);
  });
  asking.addEventListener("passages", (e) => {
    const got = JSON.parse(e.data).passages;
    const papers = new Set(got.map((p) => p.arxiv_id)).size;
    if (got.length) {
      body.textContent = "Reading " + got.length + " passages from " + papers + " papers…";
      how.push("Read " + got.length + " passages from " + papers + " papers.");
    }
  });
  asking.addEventListener("working", (e) => {
    const step = JSON.parse(e.data);
    body.className = "answer thinking";
    body.textContent = "Trying " + plain(step.strategy) + "…";
  });
  asking.addEventListener("delta", (e) => {
    draft += JSON.parse(e.data).text;
    body.className = "answer cursor";
    body.textContent =
      draft.split(/\\n\\s*CITE\\s*\\[/)[0].replace(/^[\\s\\S]*?ANSWER:\\s*/, "");
    block.scrollIntoView({block: "end"});
  });
  asking.addEventListener("answer", (e) => {
    const answer = JSON.parse(e.data);
    body.className = "answer";
    body.innerHTML = renderAnswer(answer.text, turn);
    thread.push({question: question, answer: answer.text});
    if (thread.length > 6) thread.shift();
    $("askreset").hidden = false;
    $("aq").placeholder = "Ask a follow-up";
    sources.innerHTML = renderSources(answer.citations, turn);
    if (answer.rejected.length) {
      how.push(answer.rejected.length + " quote(s) did not match their passage and were dropped.");
    }
  });
  asking.addEventListener("ladder", (e) => {
    const climb = JSON.parse(e.data);
    how.push("Read as " + (SHAPE[climb.shape] || climb.shape) + ".");
    how.push("Tried " + climb.attempts.map((a) => plain(a.strategy)).join(", then ") + ".");
  });
  asking.addEventListener("subject", (e) => {
    const check = JSON.parse(e.data);
    if (check.subject) how.push("Those passages are about: " + check.subject + ".");
    if (check.warn) {
      caution.hidden = false;
      caution.textContent =
        "These passages are mostly about " + check.subject + ", which may not be what you asked.";
    }
  });
  asking.addEventListener("failed", (e) => {
    body.className = "answer";
    body.textContent = "That did not work: " + JSON.parse(e.data).message;
  });
  asking.addEventListener("done", () => {
    if (how.length) block.appendChild(howBox(how));
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
      body.className = "answer";
      body.textContent = "The connection dropped while answering.";
    }
  };
}

// -- search -----------------------------------------------------------
function renderSnippet(text) {
  return escapeHtml(text).replace(/\\[([^\\]]+)\\]/g, "<mark>$1</mark>");
}

function renderResult(hit) {
  return [
    '<article class="hit">',
    '  <div class="head">',
    '    <a href="', escapeHtml(hit.source_url), '">', escapeHtml(hit.arxiv_id), "</a>",
    "    <span>", escapeHtml(hit.locator), "</span>",
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
  $("status").textContent = "Searching…";
  $("results").replaceChildren();
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
      ? data.results.length + " passages"
      : "Nothing matched. Exact-word search matches whole words: try field* for fields.";
    $("results").innerHTML = data.results.map(renderResult).join("");
  } catch (error) {
    $("status").textContent = "Search failed: " + error.message;
  } finally {
    $("go").disabled = false;
  }
}

// -- review -----------------------------------------------------------
function renderSectionCard(payload) {
  const card = el("section", "section");
  card.appendChild(el("h3", null, payload.heading));
  const text = el("div", "answer");
  text.innerHTML = renderAnswer(payload.text, "r" + (++turns));
  card.appendChild(text);
  if (payload.citations.length) {
    const sources = el("div");
    sources.innerHTML = renderSources(payload.citations, "r" + turns);
    card.appendChild(sources);
  }
  const how = ["Read as " + (SHAPE[payload.shape] || payload.shape) + "."];
  how.push("Tried " + payload.attempts.map((a) => plain(a.strategy)).join(", then ") + ".");
  if (payload.rejected) how.push(payload.rejected + " quote(s) were dropped as unverifiable.");
  card.appendChild(howBox(how));
  return card;
}

let reviewing = null;

function runReport(event) {
  event.preventDefault();
  const topic = $("rt").value.trim();
  if (!topic) return;
  if (reviewing) reviewing.close();
  $("reportgo").disabled = true;
  $("reportout").replaceChildren();
  $("reportstatus").textContent = "Planning the sections…";

  reviewing = new EventSource("/api/report?topic=" + encodeURIComponent(topic)
    + "&agents=" + ($("ragents").checked ? "true" : "false"));

  reviewing.addEventListener("working", (e) => {
    const at = JSON.parse(e.data);
    $("reportstatus").textContent = "Section " + at.done + " of " + at.total + ": " + at.heading;
  });
  reviewing.addEventListener("section", (e) => {
    $("reportout").appendChild(renderSectionCard(JSON.parse(e.data)));
  });
  reviewing.addEventListener("report", (e) => {
    const done = JSON.parse(e.data);
    const outcomes = done.outcomes || [];
    const accepted = outcomes.filter((o) => o.indexOf("accepted") === 0).length;
    $("reportstatus").textContent =
      "Done. " + accepted + " of " + outcomes.length + " sections stood up to checking.";
    const box = el("details", "how");
    box.appendChild(el("summary", null, "The whole review as markdown"));
    const pre = el("pre", "md", done.markdown);
    box.appendChild(pre);
    $("reportout").appendChild(box);
  });
  reviewing.addEventListener("failed", (e) => {
    $("reportstatus").textContent = "That did not work: " + JSON.parse(e.data).message;
  });
  reviewing.addEventListener("done", () => {
    reviewing.close();
    reviewing = null;
    $("reportgo").disabled = false;
  });
  reviewing.onerror = () => {
    if (!reviewing) return;
    reviewing.close();
    reviewing = null;
    $("reportstatus").textContent = "The connection dropped while writing the review.";
    $("reportgo").disabled = false;
  };
}

async function loadStats() {
  try {
    const counts = await (await fetch("/api/stats")).json();
    $("stats").textContent =
      counts.papers.toLocaleString() + " papers · "
      + counts.chunks.toLocaleString() + " passages"
      + (counts.vectors === 0 ? " · no vectors yet, so search falls back to words" : "");
  } catch (error) {
    $("stats").textContent = "";
  }
}

$("askform").addEventListener("submit", runAsk);
$("askreset").addEventListener("click", resetThread);
$("search").addEventListener("submit", runSearch);
$("reportform").addEventListener("submit", runReport);
loadStats();
</script>
</body>
</html>
"""
