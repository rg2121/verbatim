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
from pathlib import Path
from typing import Any

from verbatim import config
from verbatim.citations import render_locator
from verbatim.corpus import ChunkMatch, CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.retrieval import hybrid_search

MAX_LIMIT = 50


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
    state: Path | None = None, embedding_model: str | None = None
) -> Any:
    """Build the application. Kept a factory so tests can point it at a fixture."""
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse

    database = state or config.STATE_PATH
    model_name = embedding_model or config.EMBEDDING_MODEL
    resources: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: Any) -> Any:
        yield
        if "corpus" in resources:
            resources.pop("corpus").close()

    app = FastAPI(
        title="verbatim",
        description="Citation-checked search over a local arXiv corpus.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def corpus() -> CorpusStore:
        if "corpus" not in resources:
            resources["corpus"] = CorpusStore(database)
        return resources["corpus"]

    def embedder() -> FastEmbedEmbedder:
        if "embedder" not in resources:
            resources["embedder"] = FastEmbedEmbedder(model_name=model_name)
        return resources["embedder"]

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

    @app.get("/api/papers/{arxiv_id}")
    def paper(arxiv_id: str) -> dict[str, Any]:
        found = corpus().papers_by_id([arxiv_id]).get(arxiv_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"No paper {arxiv_id} in this corpus")
        return found.to_dict()

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

$("search").addEventListener("submit", runSearch);
loadStats();
</script>
</body>
</html>
"""
