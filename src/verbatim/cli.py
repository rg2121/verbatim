"""Command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from verbatim import config
from verbatim.agent import answer_question
from verbatim.answers import (
    OllamaCompletion,
    OllamaToolCaller,
    answer_from_matches,
    subject_check,
)
from verbatim.arxiv import (
    MAX_IDS_PER_REQUEST,
    fetch_license,
    fetch_metadata,
    fetch_papers_by_id,
    license_class,
)
from verbatim.chunking import (
    chunker_fingerprint,
    extract_pdf_chunks,
    extract_plain_pages_from_pdf,
    normalize_page_text,
)
from verbatim.citations import render_bibliography, render_locator
from verbatim.corpus import ChunkMatch, CorpusStore, PaperFilter
from verbatim.downloads import DownloadedPaper, DownloadReport, download_papers
from verbatim.embedding import FastEmbedEmbedder, embed_pending, embed_pending_papers
from verbatim.evaluation import (
    describe_case_mix,
    evaluate,
    evaluate_compound,
    render_compound_report,
    render_report,
)
from verbatim.evidence import describe_evidence
from verbatim.html_source import (
    MIN_COMPLETENESS,
    HtmlSource,
    completeness,
    extract_html_chunks,
    fetch_html,
    html_url_for,
    parse_html_sections,
)
from verbatim.logs import configure_logging
from verbatim.pipeline import MAX_ROUNDS, render_rounds, run_pipeline
from verbatim.references import references_from_file
from verbatim.reranking import CrossEncoderReranker, rerank
from verbatim.research import research
from verbatim.retrieval import hybrid_search, vector_search
from verbatim.runs import record_run, recent_runs, render_runs
from verbatim.sampling import export_sample
from verbatim.storage import (
    load_download_manifest,
    load_papers,
    save_chunks,
    save_download_manifest,
    save_papers,
)
from verbatim.sync import format_sync_report, sync_category
from verbatim.vectors import SqlitePaperIndex, SqliteVecIndex


def main() -> None:
    """Entry point: report expected failures as messages, not tracebacks."""
    configure_logging()
    try:
        run()
    except (RuntimeError, ValueError, OSError) as error:
        raise SystemExit(f"error: {error}") from error
    except sqlite3.DatabaseError as error:
        # The base class is a damaged file, OperationalError one that cannot be
        # opened or written. The other kinds are our bugs and keep their traceback.
        if type(error) in (sqlite3.DatabaseError, sqlite3.OperationalError):
            raise SystemExit(f"error: database: {error}") from error
        raise
    except KeyboardInterrupt:
        raise SystemExit("interrupted") from None


AUTHORED_HEADER = """\
# Held-out cases written by `verbatim author-cases` from each paper's title and
# abstract only; no retrieval ran against any question before this file was
# committed. Papers: current, 25-90 body passages, none named by another key,
# taken at a fixed stride per primary category. Questions sharing a 5-word run
# with their abstract were rejected. There are no unanswerable cases: those need
# the corpus inspected, which blind authoring forbids.
"""


# Every committed key; `author-cases` never writes about a paper these name.
SCORED_KEYS = (
    Path("evals/retrieval.jsonl"),
    Path("evals/heldout.jsonl"),
    Path("evals/heldout-generated.jsonl"),
    Path("evals/gate.jsonl"),
    Path("evals/breadth/questions.jsonl"),
    Path("evals/conversational.jsonl"),
)


# Help text shared by `search` and `ask`.
NO_GRAPH_HELP = (
    "Skip the citation-graph leg. On by default; it only fires on "
    "definition-shaped questions such as \"what is the Adam optimizer\""
)
USE_COMPARE_HELP = (
    "On comparison-shaped questions, also retrieve from papers whose titles "
    "match. Off by default: its effect on single-paper questions is unmeasured"
)


CONVERSATION_HEADER = """\
# Two-turn cases written by `verbatim author-cases --conversational` from each
# paper's title and abstract only; no retrieval ran against them before this
# file was committed. Each row is an opening question and a follow-up that
# refers back without naming the subject; the same paper answers both.
#
# NOT YET SCORED. Commit this file before running any arm against it.
"""


BREADTH_HEADER = """\
# Breadth questions written under evals/breadth/PROTOCOL.md, each from the
# titles and abstracts of two papers joined by a citation, named as
# require_papers. No retrieval ran against them before this file was committed.
#
# NOT YET RUN. Commit this file before running any arm against it.
"""


def run() -> None:
    parser = argparse.ArgumentParser(
        description="A citation-checked literature assistant over a local arXiv corpus."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subparsers.add_parser("fetch", help="Fetch paper metadata from one arXiv category")
    fetch_parser.add_argument("--category", required=True, help="arXiv category, e.g. physics.gen-ph")
    fetch_parser.add_argument("--max-results", type=int, default=10, help="Number of papers to fetch")
    fetch_parser.add_argument("--output", type=Path, default=config.PAPERS_SNAPSHOT_PATH)

    sync_parser = subparsers.add_parser("sync", help="Fetch metadata and upsert it into the cumulative corpus catalogue")
    sync_parser.add_argument("--category", required=True, nargs="+", help="One or more arXiv categories")
    sync_parser.add_argument("--limit", type=int, default=config.SYNC_LIMIT, help="Papers to ingest per category this run")
    sync_parser.add_argument("--page-size", type=int, default=config.SYNC_PAGE_SIZE, help="Papers per API request")
    sync_parser.add_argument("--delay", type=float, default=config.SYNC_DELAY, help="Seconds between requests")
    sync_parser.add_argument("--full", action="store_true", help="Ignore the watermark and re-walk the category")
    sync_parser.add_argument("--state", type=Path, default=config.STATE_PATH)

    download_parser = subparsers.add_parser("download", help="Download a bounded set of PDFs from a metadata snapshot")
    download_parser.add_argument("--input", type=Path, help="Legacy one-off metadata snapshot; omit to use --state")
    download_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    download_parser.add_argument("--limit", type=int, default=3, help="Maximum number of PDFs to download")
    download_parser.add_argument("--output-dir", type=Path, default=config.PDF_DIR)
    download_parser.add_argument("--delay", type=float, default=config.DOWNLOAD_DELAY,
                                 help="Seconds between PDF requests")
    download_parser.add_argument(
        "--require-license",
        action="store_true",
        help="Download only papers that report a licence at all (weaker than --license)",
    )
    download_parser.add_argument(
        "--license",
        dest="license_classes",
        nargs="+",
        choices=("open", "noncommercial", "no-derivatives", "arxiv-nonexclusive", "unknown"),
        help="Download only papers whose licence permits this. 'open' is the only "
             "class that permits redistribution or training",
    )

    stats_parser = subparsers.add_parser("stats", help="Show corpus size and work remaining")
    stats_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    stats_parser.add_argument("--model", default=config.EMBEDDING_MODEL, help="Embedding model")
    stats_parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    stats_parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)

    eval_parser = subparsers.add_parser("eval", help="Score every retrieval mode on an answer key")
    eval_parser.add_argument(
        "--cases", type=Path, nargs="+", default=[config.EVAL_PATH],
        help="One or more key files, scored as one set. Name both held-out "
             "files to score the whole frozen key",
    )
    eval_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    eval_parser.add_argument(
        "--rerank", action="store_true",
        help="Re-score with a cross-encoder (measured worse than without)",
    )
    eval_parser.add_argument("--model", default=config.EMBEDDING_MODEL, help="Embedding model")
    eval_parser.add_argument("--limit", type=int, default=10, help="Results per query")
    eval_parser.add_argument("--max-per-paper", type=int, help="Diversity cap for hybrid")
    eval_parser.add_argument(
        "--llm", default=None, help="Ollama model used for query rewriting"
    )
    eval_parser.add_argument(
        "--rewrites", type=int, default=0,
        help="Also score retrieval over this many model-written rephrasings",
    )
    eval_parser.add_argument(
        "--decompose", action="store_true",
        help="Also score retrieval after splitting each question into parts",
    )
    eval_parser.add_argument(
        "--hyde", action="store_true",
        help="Also score retrieval with a hypothetical answer added to the query",
    )
    eval_parser.add_argument(
        "--min-mrr", type=float,
        help="Exit non-zero if --gate-mode scores below this (used by CI)",
    )
    eval_parser.add_argument(
        "--gate-mode", default="hybrid+graph",
        help="Which mode --min-mrr applies to (default: the shipped default)",
    )

    answers_parser = subparsers.add_parser(
        "eval-answers",
        help="Score the answers themselves, not whether retrieval named the "
             "expected paper",
    )
    answers_parser.add_argument(
        "--cases", type=Path, nargs="+", default=[config.EVAL_PATH],
        help="One or more key files, scored as one set",
    )
    answers_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    answers_parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    answers_parser.add_argument("--llm", help="Generation model")
    answers_parser.add_argument(
        "--limit", type=int, default=20, help="Questions to answer; each costs a generation"
    )
    answers_parser.add_argument(
        "--sample", type=int,
        help="Draw this many questions evenly across authoring batches instead "
             "of the first --limit",
    )
    answers_parser.add_argument("--seed", type=int, default=0)
    answers_parser.add_argument(
        "--runs", type=int, default=1,
        help="Repeat the sample this many times, to show run-to-run spread",
    )
    answers_parser.add_argument("--passages", type=int, default=config.ASK_PASSAGES)
    answers_parser.add_argument(
        "--judge", action="store_true",
        help="Also ask a model whether each quote supports its claim, scored "
             "against mismatched control pairs",
    )

    distill_parser = subparsers.add_parser(
        "distill",
        help="Generate training examples from licence-eligible papers",
    )
    distill_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    distill_parser.add_argument("--out", type=Path, default=config.DISTILL_PATH)
    distill_parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    distill_parser.add_argument("--llm", help="Generation model")
    distill_parser.add_argument(
        "--examples", type=int, default=50,
        help="Examples to keep; each costs at least two generations",
    )
    distill_parser.add_argument(
        "--questions-per-passage", type=int, default=2,
        help="Questions asked of each seed passage",
    )
    distill_parser.add_argument("--passages", type=int, default=config.ASK_PASSAGES)
    distill_parser.add_argument(
        "--seed", type=int, default=0,
        help="Which seed passages are drawn; the same seed draws the same ones",
    )

    author_parser = subparsers.add_parser(
        "author-cases",
        help="Write held-out evaluation cases from titles and abstracts only",
    )
    author_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    author_parser.add_argument(
        "--out", type=Path, default=Path("evals/heldout-generated.jsonl")
    )
    author_parser.add_argument("--cases", type=int, default=100)
    author_parser.add_argument(
        "--breadth", action="store_true",
        help="Write two-paper questions for the breadth evaluation, under "
             "evals/breadth/PROTOCOL.md",
    )
    author_parser.add_argument(
        "--per-kind", type=int, default=8,
        help="With --breadth: questions of each kind (comparison, limitation, synthesis)",
    )
    author_parser.add_argument(
        "--conversational", action="store_true",
        help="Write two-turn cases (a question and a follow-up) for evaluating "
             "conversations",
    )
    author_parser.add_argument("--llm", help="Generation model")
    author_parser.add_argument(
        "--exclude", type=Path, nargs="*", default=list(SCORED_KEYS),
        help="Existing keys whose papers must not be reused",
    )

    gaps_parser = subparsers.add_parser(
        "gaps",
        help="Propose what a topic's literature is missing, then try to disprove it",
    )
    gaps_parser.add_argument("topic", help="The area to look for gaps in")
    gaps_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    gaps_parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    gaps_parser.add_argument("--llm", help="Generation model")
    gaps_parser.add_argument("--limit", type=int, default=4,
                             help="Gaps to propose; each costs a falsification")
    gaps_parser.add_argument("--queries", type=int, default=4,
                             help="Searches the falsifier may run per gap")
    gaps_parser.add_argument(
        "--hide", nargs="*", default=[], metavar="ARXIV_ID",
        help="Hide these papers from the falsifier, to plant a gap whose "
             "answer is known. Nothing is deleted",
    )

    team_parser = subparsers.add_parser(
        "team",
        help="Answer through a shared evidence board that decides what runs next",
    )
    team_parser.add_argument("question")
    team_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    team_parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    team_parser.add_argument("--llm", help="Generation model")
    team_parser.add_argument("--tasks", type=int, default=3,
                             help="Most tasks the coordinator may create")
    team_parser.add_argument("--needs", type=int, default=2,
                             help="Passages a task wants before it is satisfied")
    team_parser.add_argument("--max-steps", type=int, default=12)

    index_parser = subparsers.add_parser("index", help="Reconcile the vector index")
    index_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    index_parser.add_argument("--model", default=config.EMBEDDING_MODEL, help="Embedding model")
    index_parser.add_argument(
        "--rebuild", action="store_true", help="Discard and reload, rather than syncing changes"
    )

    licenses_parser = subparsers.add_parser(
        "licenses", help="Look up licences via OAI-PMH, which the query API omits"
    )
    licenses_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    licenses_parser.add_argument("--limit", type=int, default=50, help="Papers to look up this run")
    licenses_parser.add_argument(
        "--delay", type=float, default=config.LICENSE_LOOKUP_DELAY, help="Seconds between requests; arXiv asks for a pause"
    )

    html_parser = subparsers.add_parser(
        "html", help="Fetch arXiv HTML renderings, which keep maths as LaTeX"
    )
    html_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    html_parser.add_argument("--output-dir", type=Path, default=config.HTML_DIR)
    html_parser.add_argument("--limit", type=int, default=200, help="Papers to fetch this run")
    html_parser.add_argument("--delay", type=float, default=config.DOWNLOAD_DELAY)

    references_parser = subparsers.add_parser(
        "references",
        help="Read bibliographies from HTML renderings and rank what the corpus lacks",
    )
    references_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    references_parser.add_argument(
        "--limit", type=int, default=2000, help="Papers to read this run"
    )
    references_parser.add_argument(
        "--reading-list", type=int, default=25,
        help="How many uncited-but-absent papers to name at the end",
    )

    runs_parser = subparsers.add_parser(
        "runs", help="What each command did, and whether it finished"
    )
    runs_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    runs_parser.add_argument(
        "--command", dest="command_filter", help="Only runs of this command"
    )
    runs_parser.add_argument("--limit", type=int, default=20)
    runs_parser.add_argument(
        "--json", action="store_true",
        help="Emit the ledger as JSON",
    )

    graph_parser = subparsers.add_parser(
        "graph",
        help="Walk the internal citation graph: what cites a paper, what it builds on",
    )
    graph_parser.add_argument(
        "identifier", nargs="?",
        help="arXiv ID or approximate title; omit for corpus-wide graph stats",
    )
    graph_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    graph_parser.add_argument(
        "--top", type=int, default=10,
        help="Without an identifier: how many most-cited held papers to list",
    )
    graph_parser.add_argument(
        "--limit", type=int, default=25, help="Citing or cited papers to list per direction"
    )

    grow_parser = subparsers.add_parser(
        "grow",
        help="Add the papers this corpus cites most and does not hold",
    )
    grow_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    grow_parser.add_argument(
        "--limit", type=int, default=100, help="How many papers to add"
    )
    grow_parser.add_argument(
        "--min-citations", type=int, default=2,
        help="Only add papers cited by at least this many held papers",
    )
    grow_parser.add_argument("--delay", type=float, default=config.DOWNLOAD_DELAY)
    grow_parser.add_argument(
        "--dry-run", action="store_true", help="List what would be added and stop"
    )

    serve_parser = subparsers.add_parser(
        "serve", help="Serve the corpus over HTTP, with a search page"
    )
    serve_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    serve_parser.add_argument("--model", default=config.EMBEDDING_MODEL, help="Embedding model")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument(
        "--host", default="127.0.0.1",
        help="Loopback by default: the corpus is other people's papers under mixed licences",
    )

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run every ingest stage in rounds until nothing is pending",
    )
    pipeline_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    pipeline_parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    pipeline_parser.add_argument(
        "--max-rounds", type=int, default=MAX_ROUNDS,
        help="Stop after this many rounds even if work remains",
    )

    sample_parser = subparsers.add_parser(
        "sample",
        help="Export a small self-contained corpus that works without an ingest",
    )
    sample_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    sample_parser.add_argument("--out", type=Path, default=Path("data/sample-export.db"))
    sample_parser.add_argument("--papers", type=int, default=40)
    sample_parser.add_argument("--model", default=config.EMBEDDING_MODEL)

    mcp_parser = subparsers.add_parser(
        "mcp", help="Serve the corpus over the Model Context Protocol, on stdio"
    )
    mcp_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    mcp_parser.add_argument("--model", default=config.EMBEDDING_MODEL, help="Embedding model")

    embed_parser = subparsers.add_parser("embed", help="Embed chunks that have no vector yet")
    embed_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    embed_parser.add_argument(
        "--with-context",
        action="store_true",
        default=config.EMBED_WITH_CONTEXT,
        help="Embed each passage behind its paper title and section heading. "
             "Stored under a separate model id, so both can be measured.",
    )
    embed_parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    embed_parser.add_argument(
        "--only-current", action="store_true",
        help="Skip passages from an older chunker, which the next chunk run replaces",
    )
    embed_parser.add_argument("--batch-size", type=int, default=64)
    embed_parser.add_argument("--limit", type=int, help="Embed at most this many chunks this run")

    search_parser = subparsers.add_parser(
        "search", help="Search the corpus: hybrid by default, or keyword or vector only"
    )
    search_parser.add_argument("query", help="Search terms; 'term*' matches by prefix")
    search_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    search_parser.add_argument(
        "--rerank", action="store_true",
        help="Re-score with a cross-encoder (measured worse than without)",
    )
    search_parser.add_argument("--limit", type=int, default=config.SEARCH_LIMIT, help="Maximum number of passages")
    search_parser.add_argument(
        "--use-titles", action="store_true",
        help="Also retrieve from papers whose title matches (measured worse overall)",
    )
    search_parser.add_argument(
        "--no-graph", dest="use_graph", action="store_false",
        help=NO_GRAPH_HELP,
    )
    search_parser.add_argument(
        "--use-compare", action="store_true",
        help=USE_COMPARE_HELP,
    )
    search_parser.add_argument(
        "--since", help="Only papers published on or after this date (YYYY-MM-DD)"
    )
    search_parser.add_argument(
        "--category", help="Only papers in this arXiv category, e.g. cs.CL"
    )
    search_parser.add_argument("--arxiv-id", help="Restrict the search to one paper version")
    search_parser.add_argument("--full", action="store_true", help="Print whole passages, not snippets")
    search_parser.add_argument(
        "--mode",
        choices=("hybrid", "keyword", "vector"),
        default="hybrid",
        help="hybrid: both, fused by rank (default). keyword: exact terms. vector: meaning",
    )
    search_parser.add_argument(
        "--match",
        choices=("any", "all"),
        default="any",
        help="keyword mode only: rank by matched terms (default), or require every term",
    )
    search_parser.add_argument("--model", default=config.EMBEDDING_MODEL, help="Embedding model for --mode vector")
    search_parser.add_argument(
        "--include-references",
        action="store_true",
        help="Also search bibliographies, which are excluded by default",
    )

    ask_parser = subparsers.add_parser("ask", help="Answer a question from the corpus, with citations")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    ask_parser.add_argument("--passages", type=int, default=config.ASK_PASSAGES, help="Passages to retrieve")
    ask_parser.add_argument("--arxiv-id", help="Restrict the answer to one paper version")
    ask_parser.add_argument("--model", default=config.EMBEDDING_MODEL, help="Embedding model")
    ask_parser.add_argument(
        "--plan", action="store_true",
        help="Split the question into parts and retrieve for each, for "
             "compound questions one retrieval cannot reach all of",
    )
    ask_parser.add_argument(
        "--no-graph", dest="use_graph", action="store_false",
        help=NO_GRAPH_HELP,
    )
    ask_parser.add_argument(
        "--use-compare", action="store_true",
        help=USE_COMPARE_HELP,
    )
    ask_parser.add_argument(
        "--llm",
        help="Ollama model (default: $OLLAMA_MODEL or the first local model)",
    )
    ask_parser.add_argument(
        "--supervisor", default="builtin", choices=["builtin", "langgraph"],
        help="With --deep, which implementation climbs the ladder",
    )
    ask_parser.add_argument(
        "--deep", action="store_true",
        help="Retry with the next retrieval strategy until a citation verifies and "
             "the passages are on-subject; no extra cost when the first attempt works",
    )
    ask_parser.add_argument(
        "--tools",
        action="store_true",
        help="Let the model route the question: recency and single-paper questions "
             "go to metadata rather than passage search",
    )
    ask_parser.add_argument(
        "--include-references",
        action="store_true",
        help="Also search bibliographies, which are excluded by default",
    )
    ask_parser.add_argument(
        "--no-subject-check",
        action="store_true",
        help="Skip naming what the passages are about (saves one short model call)",
    )
    ask_parser.add_argument(
        "--max-per-paper",
        type=int,
        default=config.ASK_MAX_PER_PAPER,
        help="Cap passages taken from any one paper, so one source cannot dominate",
    )

    report_parser = subparsers.add_parser(
        "report", help="Write a sectioned literature review with verified citations"
    )
    report_parser.add_argument("topic", help="What the review is about; a phrase, not a question")
    report_parser.add_argument(
        "--supervisor", default="builtin", choices=["builtin", "langgraph"],
        help="Which implementation runs each section's retries (langgraph needs "
             "the optional extra)",
    )
    report_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    report_parser.add_argument("--model", default=config.EMBEDDING_MODEL, help="Embedding model")
    report_parser.add_argument(
        "--llm", help="Ollama model (default: $OLLAMA_MODEL or the first local model)"
    )
    report_parser.add_argument(
        "--sections", type=int, default=4, help="Maximum sections to plan"
    )
    report_parser.add_argument(
        "--passages", type=int, default=6, help="Passages retrieved per section"
    )
    report_parser.add_argument(
        "--max-per-paper", type=int, default=2,
        help="Cap passages taken from any one paper within a section",
    )
    report_parser.add_argument(
        "--use-compare", action="store_true",
        help="Enable the comparison-gated title leg for comparison-shaped sections",
    )
    report_parser.add_argument(
        "--no-subject-check", action="store_true",
        help="Skip naming what each section's passages are about (saves one model call per section)",
    )
    report_parser.add_argument(
        "--out", type=Path, help="Write the review to this markdown file instead of printing it"
    )
    report_parser.add_argument(
        "--agents", action="store_true",
        help="Retry a section that fails verification with the next strategy, "
             "and append a ledger of what was tried",
    )
    report_parser.add_argument(
        "--workers", type=int, default=1,
        help="With --agents: sections worked in parallel by this many threads",
    )
    report_parser.add_argument(
        "--critic", action="store_true",
        help="With --agents: flag uncited claims and uncovered subjects; only "
             "uncovered subjects trigger a retry",
    )
    report_parser.add_argument(
        "--max-tokens", type=int,
        help="Stop the run once this many tokens have been spent",
    )
    report_parser.add_argument(
        "--max-seconds", type=float, help="Stop the run after this long"
    )
    report_parser.add_argument(
        "--max-attempts", type=int,
        help="Stop after this many retrieval attempts across all sections",
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Extract passages from HTML renderings, or from the PDF where there is none"
    )
    chunk_parser.add_argument("--manifest", type=Path, help="Legacy one-off PDF manifest; omit to use --state")
    chunk_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    chunk_parser.add_argument("--output", type=Path, default=config.CHUNKS_PATH)
    chunk_parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    chunk_parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    chunk_parser.add_argument("--limit", type=int, help="Chunk at most this many PDFs this run")
    chunk_parser.add_argument(
        "--only-checked", action="store_true",
        help="Skip papers whose HTML has not been looked for yet",
    )
    chunk_parser.add_argument("--batch-size", type=int, default=config.CHUNK_BATCH_SIZE,
                              help="PDFs held in memory at once; bounds peak memory")
    args = parser.parse_args()

    if args.command == "fetch":
        papers = fetch_metadata(args.category, args.max_results)
        save_papers(papers, args.output)
        print(f"Saved {len(papers)} papers to {args.output}")

    if args.command == "sync":
        with CorpusStore(args.state) as corpus, record_run(corpus, "sync", limit=args.limit, categories=args.category, full=args.full) as run:
            for position, category in enumerate(args.category):
                if position:
                    time.sleep(args.delay)
                report = sync_category(
                    corpus,
                    category,
                    limit=args.limit,
                    page_size=args.page_size,
                    delay=args.delay,
                    full=args.full,
                    on_page=lambda done, total: print(
                        f"  {category}: {done}/{total}", end="\r", flush=True
                    ),
                )
                print(format_sync_report(report))

    if args.command == "download":
        manifest_path = args.output_dir / "manifest.json"
        if args.input is not None:
            papers = load_papers(args.input)
            existing_downloads = load_download_manifest(manifest_path) if manifest_path.exists() else []
            report = download_papers(
                papers,
                args.output_dir,
                args.limit,
                require_license=args.require_license,
                existing_downloads=existing_downloads,
            )
            downloads = _merge_downloads(existing_downloads, report.downloaded)
        else:
            with CorpusStore(args.state) as corpus, record_run(corpus, "download", limit=args.limit, license_classes=args.license_classes) as run:
                papers = corpus.pending_downloads(
                    args.limit,
                    require_license=args.require_license,
                    license_classes=args.license_classes,
                )
                report = (
                    download_papers(
                        papers,
                        args.output_dir,
                        len(papers),
                        require_license=args.require_license,
                        existing_downloads=corpus.all_downloads(),
                        delay=args.delay,
                        # Persist each success as it lands: a run this long
                        # must not lose hours of work to one interruption.
                        on_download=lambda record: corpus.record_downloads([record]),
                    )
                    if papers
                    else DownloadReport(downloaded=[], failed=[])
                )
                # Recorded before reporting failures, so a partial run still
                # leaves every fetched PDF accounted for.
                corpus.record_downloads(report.downloaded)
                # A 404 for a versioned PDF is final; record it so it is not retried.
                settled = [
                    (f.arxiv_id, f.pdf_url, f.reason)
                    for f in report.failed if f.permanent
                ]
                if settled:
                    corpus.record_permanent_download_failures(settled)
                downloads = corpus.all_downloads()
                run.record(downloaded=len(report.downloaded),
                           failed=len(report.failed), held=len(downloads))
        save_download_manifest(downloads, manifest_path)
        print(
            f"Downloaded {len(report.downloaded)} new PDFs; "
            f"corpus manifest has {len(downloads)} PDFs: {manifest_path}"
        )
        for failure in report.failed:
            print(f"  ! {failure.arxiv_id}: {failure.reason}")
        if report.failed:
            print(f"  {len(report.failed)} paper(s) could not be downloaded.")

    if args.command == "stats":
        # Count under the id `embed` writes, which adds "+context" when used.
        model_id = FastEmbedEmbedder(model_name=args.model).model_id
        with CorpusStore(args.state) as corpus:
            counts = corpus.pipeline_stats(
                chunker_fingerprint(args.chunk_size, args.overlap), model_id
            )
            synced = corpus.sync_summary()
        print(format_stats(counts))
        if synced:
            print("\n  synced categories")
            for category, watermark, seen in synced:
                mark = (watermark or "never")[:19]
                print(f"    {category:<18} {seen:>7,} ingested   up to {mark}")

    if args.command == "eval":
        from verbatim.evaluation import load_key

        cases = load_key(args.cases)
        embedder = FastEmbedEmbedder(model_name=args.model)
        named = ",".join(str(path) for path in args.cases)
        with CorpusStore(args.state) as corpus, record_run(corpus, "eval", cases=named, model=args.model, limit=args.limit) as run:
            expanding = args.rewrites or args.hyde or args.decompose
            options = dict(
                limit=args.limit, max_per_paper=args.max_per_paper,
                reranker=CrossEncoderReranker() if args.rerank else None,
                complete=_line_completion(args.llm) if expanding else None,
                rewrites=args.rewrites, hyde=args.hyde,
                decompose=args.decompose,
            )
            scores = evaluate(corpus, embedder, cases, **options)
            compound_scores = (
                evaluate_compound(corpus, embedder, cases, **options)
                if any(case.is_compound for case in cases)
                else []
            )
            best = max(scores, key=lambda score: score.mrr)
            run.record(cases=len(cases), scored=best.cases,
                       best_mode=best.mode, best_mrr=round(best.mrr, 4))
        print(named)
        print(describe_case_mix(cases))
        print()
        print(render_report(scores))
        if compound_scores:
            print()
            print(render_compound_report(compound_scores))
        if args.min_mrr is not None:
            gated = next((s for s in scores if s.mode == args.gate_mode), None)
            if gated is None:
                raise SystemExit(
                    f"error: --gate-mode {args.gate_mode!r} is not one of "
                    f"{', '.join(sorted(s.mode for s in scores))}"
                )
            print(
                f"\n  gate: {gated.mode} scored mrr {gated.mrr:.3f} "
                f"against a floor of {args.min_mrr:.3f}"
            )
            if gated.mrr < args.min_mrr:
                raise SystemExit(
                    f"error: {gated.mode} regressed to mrr {gated.mrr:.3f}, "
                    f"below the floor of {args.min_mrr:.3f}. Either a change made "
                    f"retrieval worse, or the floor needs re-measuring on purpose."
                )

    if args.command == "author-cases" and args.conversational:
        from verbatim.authoring import (
            author_conversations,
            excluded_papers,
            render_rejections,
        )

        out = (
            Path("evals/conversational.jsonl")
            if args.out == Path("evals/heldout-generated.jsonl") else args.out
        )
        if out.exists() and out.read_text(encoding="utf-8").strip():
            # A frozen key is written once; re-authoring takes a deliberate deletion.
            raise SystemExit(
                f"error: {out} already exists. A frozen key is written once; "
                "delete it deliberately if you mean to re-author it."
            )
        completion = OllamaCompletion(args.llm or config.OLLAMA_MODEL)
        exclude = excluded_papers(str(path) for path in args.exclude)
        rejected: dict[str, int] = {}
        with CorpusStore(args.state) as corpus, record_run(
            corpus, "author-cases", conversational=True, cases=args.cases, llm=args.llm
        ) as run:
            try:
                written = list(author_conversations(
                    corpus,
                    lambda system, user: completion.complete_text(system, user),
                    cases=args.cases, exclude=exclude, rejected=rejected,
                ))
            except (RuntimeError, OSError) as error:
                raise SystemExit(
                    f"error: authoring stopped ({error}). Nothing was written: a "
                    "half-written key must never sit on disk looking frozen."
                ) from error
            run.record(written=len(written), rejected=sum(rejected.values()))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            CONVERSATION_HEADER + "".join(case.to_row() + "\n" for case in written),
            encoding="utf-8",
        )
        print(f"{out}\n")
        print(f"  wrote {len(written)}/{args.cases} two-turn case(s)")
        print(render_rejections(rejected))
        print("\n  NOT YET SCORED. Commit this file before running any arm against it.")
        if len(written) < args.cases:
            raise SystemExit(1)
        return

    if args.command == "author-cases" and args.breadth:
        from verbatim.authoring import (
            PAIR_KINDS,
            author_pairs,
            excluded_papers,
            render_rejections,
        )

        out = (
            Path("evals/breadth/questions.jsonl")
            if args.out == Path("evals/heldout-generated.jsonl") else args.out
        )
        if out.exists() and out.read_text(encoding="utf-8").strip():
            # A frozen key is written once; re-authoring takes a deliberate deletion.
            raise SystemExit(
                f"error: {out} already exists. A frozen key is written once; "
                "delete it deliberately if you mean to re-author it."
            )
        completion = OllamaCompletion(args.llm or config.OLLAMA_MODEL)
        exclude = excluded_papers(str(path) for path in args.exclude)
        rejected: dict[str, int] = {}
        with CorpusStore(args.state) as corpus, record_run(
            corpus, "author-cases", breadth=True, per_kind=args.per_kind, llm=args.llm
        ) as run:
            try:
                cases = list(author_pairs(
                    corpus,
                    lambda system, user: completion.complete_text(system, user),
                    per_kind=args.per_kind, exclude=exclude, rejected=rejected,
                ))
            except (RuntimeError, OSError) as error:
                raise SystemExit(
                    f"error: authoring stopped ({error}). Nothing was written: a "
                    "half-written key must never sit on disk looking frozen."
                ) from error
            run.record(written=len(cases), rejected=sum(rejected.values()))
        out.parent.mkdir(parents=True, exist_ok=True)
        # Written only once every case exists, for the same reason.
        out.write_text(
            BREADTH_HEADER + "".join(case.to_row() + "\n" for case in cases),
            encoding="utf-8",
        )
        print(f"{out}\n")
        short = []
        for kind in PAIR_KINDS:
            got = sum(1 for case in cases if case.kind == kind)
            print(f"  {kind:<11} {got}/{args.per_kind}")
            if got < args.per_kind:
                short.append(kind)
        print(render_rejections(rejected))
        if short:
            print(f"\n  SHORT on {', '.join(short)}: the protocol asks for "
                  f"{args.per_kind} of each, and candidate pairs ran out.")
        print("\n  NOT YET RUN. Commit this file before running any arm against it.")
        if short:
            raise SystemExit(1)
        return

    if args.command == "author-cases":
        from verbatim.authoring import author, excluded_papers, render_rejections

        completion = OllamaCompletion(args.llm or config.OLLAMA_MODEL)
        exclude = excluded_papers(str(path) for path in args.exclude)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        existing = [
            line for line in
            (args.out.read_text().splitlines() if args.out.exists() else [])
            if line.strip() and not line.startswith("#")
        ]
        if existing:
            for line in existing:
                for arxiv_id in json.loads(line).get("expect_papers") or []:
                    exclude.add(str(arxiv_id))
            print(f"resuming: {len(existing)} case(s) already in {args.out}")
        rejected: dict[str, int] = {}
        header = args.out.exists() and bool(existing)
        stopped: str | None = None
        written = 0
        with CorpusStore(args.state) as corpus, record_run(
            corpus, "author-cases", cases=args.cases, llm=args.llm
        ) as run, args.out.open("a") as sink:
            if not header:
                sink.write(AUTHORED_HEADER)
            try:
                for case in author(
                    corpus,
                    lambda system, user: completion.complete_text(system, user),
                    cases=args.cases,
                    exclude=exclude,
                    rejected=rejected,
                ):
                    sink.write(case.to_row() + "\n")
                    sink.flush()
                    written += 1
                    print(f"  wrote {written}/{args.cases}", end="\r", flush=True)
            except (RuntimeError, OSError) as error:
                stopped = f"{type(error).__name__}: {error}"
            run.record(written=written, rejected=sum(rejected.values()))
        print(f"{args.out}\n")
        print(f"  wrote {written} case(s)")
        print(render_rejections(rejected))
        print(
            "\n  NOT YET SCORED. Commit this file before running any evaluation"
            " against it:\n  a question that has been retrieved for is no longer"
            " held out."
        )
        if stopped is not None:
            print(f"\n  STOPPED: {stopped}\n  Re-run to resume.")
            raise SystemExit(1)
        return

    if args.command == "team":
        from verbatim.team import run_team

        embedder = FastEmbedEmbedder(model_name=args.model)
        completion = OllamaCompletion(args.llm or config.OLLAMA_MODEL)
        with CorpusStore(args.state) as corpus, record_run(
            corpus, "team", question=args.question, tasks=args.tasks, llm=args.llm
        ) as run:
            report = run_team(
                corpus, embedder, completion,
                lambda system, user: completion.complete_text(system, user),
                args.question, max_tasks=args.tasks, needs=args.needs,
                max_steps=args.max_steps,
            )
            run.record(
                tasks=len(report.state.tasks),
                evidence=len(report.state.evidence),
                decisions=len(report.state.decisions),
                redirections=report.redirections,
            )
        print(report.render())
        print(f"\n  {_usage_footer(completion)}")
        return

    if args.command == "gaps":
        from verbatim.gaps import REFUTED, STANDS, find_gaps, render

        embedder = FastEmbedEmbedder(model_name=args.model)
        completion = OllamaCompletion(args.llm or config.OLLAMA_MODEL)
        with CorpusStore(args.state) as corpus, record_run(
            corpus, "gaps", topic=args.topic, limit=args.limit, llm=args.llm
        ) as run:
            findings, rejected = find_gaps(
                corpus, embedder, completion,
                lambda system, user: completion.complete_text(system, user),
                args.topic, limit=args.limit, hidden=args.hide,
                queries=args.queries,
            )
            run.record(
                proposed=len(findings),
                refuted=sum(1 for f in findings if f.verdict == REFUTED),
                stood=sum(1 for f in findings if f.verdict == STANDS),
                unfalsifiable=len(rejected),
            )
        print(f"{args.topic}\n")
        print(render(findings, rejected))
        if args.hide:
            print(f"\n  {len(args.hide)} paper(s) were hidden from the falsifier."
                  "\n  A gap that stands here is only absent from what it could reach.")
        print(f"\n  {_usage_footer(completion)}")
        return

    if args.command == "distill":
        from verbatim.distill import (
            Tally,
            attributions,
            generate,
            source_papers,
        )

        completion = OllamaCompletion(args.llm or config.OLLAMA_MODEL)
        embedder = FastEmbedEmbedder(model_name=args.model)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Resumed by example id rather than by count: the same seed draws the
        # same passages, so a second run would otherwise regenerate them.
        existing = [
            json.loads(line)
            for line in (args.out.read_text().splitlines() if args.out.exists() else [])
            if line.strip()
        ]
        if existing:
            print(f"resuming: {len(existing)} example(s) already in {args.out}")
        tally = Tally()
        kept = list(existing)
        stopped: str | None = None
        with CorpusStore(args.state) as corpus, record_run(
            corpus, "distill", examples=args.examples, llm=args.llm, seed=args.seed
        ) as run, args.out.open("a") as sink:
            examples_made = generate(
                corpus, embedder, completion,
                complete_text=completion.complete_text,
                generator_model=completion.model,
                examples=args.examples,
                questions_per_passage=args.questions_per_passage,
                seed=args.seed,
                passages=args.passages,
                tally=tally,
                skip=[row["example_id"] for row in existing],
            )
            # Every kept example is already on disk, so a failure stops the run and
            # the same command resumes it.
            try:
                for example in examples_made:
                    record = example.to_dict()
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
                    kept.append(record)
                    print(f"  kept {tally.kept}/{args.examples}", end="\r", flush=True)
            except (RuntimeError, OSError) as error:
                stopped = f"{type(error).__name__}: {error}"
            run.record(
                kept=tally.kept,
                considered=tally.considered,
                faults=sum(tally.faults.values()),
            )
        print(f"{args.out}\n")
        print(tally.render())
        if stopped is not None:
            print(f"\n  STOPPED: {stopped}")
            print("  Re-run the same command with the same --seed to resume.")
        if tally.faults:
            print("\n  A fault means an ineligible paper reached an answer's"
                  " passages.\n  The licence gate is not holding; do not train"
                  " on this dataset.")
        # Rewritten whole from the file, so a resumed run's ATTRIBUTIONS covers
        # every example in the dataset and not only this run's.
        papers = source_papers(
            paper for record in kept for paper in record["papers"]
        )
        credits = args.out.with_name("ATTRIBUTIONS")
        credits.write_text(attributions(papers))
        print(f"\n  {len(papers)} source paper(s) credited in {credits}")
        if stopped is not None:
            raise SystemExit(1)
        return

    if args.command == "eval-answers":
        from verbatim.answer_eval import (
            AnsweredCase,
            judge_with_control,
            score_answers,
        )
        from verbatim.evaluation import load_key, source_of, stratified_sample

        answerable = [c for c in load_key(args.cases) if not c.expect_none]
        cases = (
            stratified_sample(answerable, size=args.sample, seed=args.seed)
            if args.sample else answerable[: args.limit]
        )
        source_by_question = {c.question: source_of(c) for c in cases}
        embedder = FastEmbedEmbedder(model_name=args.model)
        completion = OllamaCompletion(args.llm or config.OLLAMA_MODEL)
        answered: list = []
        per_run: list[list] = []
        stopped: str | None = None
        total = len(cases) * args.runs
        with CorpusStore(args.state) as corpus, record_run(
            corpus, "eval-answers", cases=len(cases), runs=args.runs, llm=args.llm
        ) as run:
            done = 0
            for _repeat in range(args.runs):
                this_run: list = []
                for case in cases:
                    try:
                        matches = hybrid_search(
                            corpus, embedder, case.question, limit=args.passages,
                            max_per_paper=config.ASK_MAX_PER_PAPER, use_graph=True,
                        )
                        answer = answer_from_matches(
                            corpus, completion, case.question, matches
                        )
                    except (RuntimeError, OSError) as error:
                        # Score what was answered before the failure.
                        stopped = f"{type(error).__name__}: {error}"
                        break
                    this_run.append(AnsweredCase(
                        question=case.question, answer=answer,
                        expected_papers=case.expect_papers,
                    ))
                    done += 1
                    print(f"  answered {done}/{total}", end="\r", flush=True)
                if this_run:
                    per_run.append(this_run)
                    answered.extend(this_run)
                if stopped is not None:
                    break
            score = score_answers(answered)
            run.record(
                cases=score.cases, declined=score.declined,
                verified=score.with_verified_citation, fully_cited=score.fully_cited,
            )
        print(",".join(str(path) for path in args.cases) + "\n")
        print(score.render())
        if len(per_run) > 1:
            # One run cannot show its own spread.
            print("\n  per run, answers with a verified citation:")
            for number, batch in enumerate(per_run, start=1):
                got = score_answers(batch)
                print(f"    run {number}: {got.with_verified_citation}/{got.cases}"
                      f" ({got.with_verified_citation / max(got.cases, 1):.0%})")
        batches: dict[str, list] = {}
        for case in answered:
            batches.setdefault(source_by_question.get(case.question, "unknown"),
                               []).append(case)
        if len(batches) > 1:
            # How a batch was authored moves its score more than the system does.
            print("\n  by authoring batch:")
            for source, group in sorted(batches.items()):
                got = score_answers(group)
                print(f"    {source:<20} {got.cases:>4} answers, "
                      f"verified {_percent(got.with_verified_citation, got.cases)}, "
                      f"marked {_percent(got.fully_cited, got.cases)}, "
                      f"expected paper {_percent(got.cited_expected_paper, got.cases)}")
        if stopped is not None:
            print(f"\n  STOPPED after {len(answered)} answer(s): {stopped}")
        if args.judge:
            print("\n  Does each cited quote support its claim? Real pairs are scored")
            print("  beside mismatched control pairs.\n")
            report = judge_with_control(
                answered, lambda system, user: completion.complete_text(system, user)
            )
            print(report.render())
        print(f"\n  {_usage_footer(completion)}")

    if args.command == "index":
        embedder = FastEmbedEmbedder(model_name=args.model)
        with CorpusStore(args.state) as corpus, record_run(corpus, "index", model=args.model, rebuild=args.rebuild) as run:
            model = embedder.model
            index = SqliteVecIndex(corpus.connection, model.dimensions)
            papers = SqlitePaperIndex(corpus.connection, model.dimensions)
            # Record the generation built from, so the next search finds it current.
            generation = corpus.chunk_generation()
            if args.rebuild:
                rebuilt = index.rebuild(model.model_id, generation)
                rebuilt_papers = papers.rebuild(model.model_id, generation)
                run.record(rebuilt=rebuilt, rebuilt_papers=rebuilt_papers,
                           model=model.model_id)
                print(
                    f"Rebuilt the index with {rebuilt} passage "
                    f"vectors and {rebuilt_papers} paper vectors."
                )
            else:
                change = index.sync(model.model_id, generation)
                paper_change = papers.sync(model.model_id, generation)
                run.record(added=change.added, removed=change.removed,
                           vectors=index.count(), model=model.model_id,
                           paper_vectors_added=paper_change.added,
                           paper_vectors_removed=paper_change.removed)
                print(f"Index synced: {change.added} added, {change.removed} removed; "
                      f"{index.count()} passage vectors, {papers.count()} paper vectors.")

    if args.command == "licenses":
        # Recorded in the run ledger like every other stage.
        with CorpusStore(args.state) as corpus, record_run(corpus, "licenses", limit=args.limit) as run:
            pending = corpus.base_ids_missing_license(args.limit)
            found = failed = 0
            for position, base_id in enumerate(pending):
                if position:
                    time.sleep(args.delay)  # arXiv asks callers to pace themselves
                try:
                    licence = fetch_license(base_id)
                    corpus.record_license(base_id, licence)
                    found += licence is not None
                except (OSError, ValueError) as error:
                    failed += 1
                    print(f"  ! {base_id}: {error}")
            run.record(looked_up=len(pending), found=found, failed=failed)
            summary = corpus.license_summary()
        print(f"Looked up {len(pending)} papers. Licences across the corpus:")
        for license_url, count in summary.items():
            url = None if license_url == "(none reported)" else license_url
            print(f"  {count:4}  [{license_class(url)}]  {license_url}")
        print(
            "  Only 'open' permits redistribution or training; "
            "'arxiv-nonexclusive' grants arXiv distribution rights, not you."
        )

    if args.command == "html":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        fetched = missing = truncated = deferred = 0
        with CorpusStore(args.state) as corpus, record_run(corpus, "html", limit=args.limit) as run:
            pending = corpus.papers_without_html(args.limit)
            for position, arxiv_id in enumerate(pending):
                if position:
                    time.sleep(args.delay)
                try:
                    payload = fetch_html(arxiv_id)
                except (OSError, RuntimeError, ValueError) as error:
                    print(f"  ! {arxiv_id}: {error}")
                    continue
                # A write can lose a race with a long chunking transaction: defer
                # the paper to the next run rather than abort this one.
                try:
                    if payload is None:
                        # No TeX source, so no rendering; record that so it is not re-asked.
                        corpus.record_html_absent(arxiv_id, html_url_for(arxiv_id))
                        missing += 1
                        continue
                    path = args.output_dir / f"{arxiv_id.replace('/', '-')}.html"
                    path.write_bytes(payload)
                    source = HtmlSource(
                        arxiv_id=arxiv_id, url=html_url_for(arxiv_id), local_path=str(path),
                        sha256=hashlib.sha256(payload).hexdigest(), byte_count=len(payload),
                    )
                    # A rendering is only better input if it is the whole paper.
                    covered = html_completeness(corpus, source)
                    if covered < MIN_COMPLETENESS:
                        corpus.record_html_source(
                            source, usable=False,
                            note=f"rendering covers {covered:.0%} of the PDF text",
                        )
                        truncated += 1
                    else:
                        corpus.record_html_source(source)
                        fetched += 1
                except sqlite3.OperationalError as error:
                    deferred += 1
                    print(f"  ! {arxiv_id} deferred: {error}")
                    continue
                print(f"  fetched {fetched}/{len(pending)}", end="\r", flush=True)
        print(
            f"Fetched {fetched} usable HTML renderings; {missing} papers have none, "
            f"{truncated} were truncated by arXiv's conversion (both use the PDF)."
            + (f" {deferred} deferred to a later run because the database was busy."
               if deferred else "")
        )

    if args.command == "references":
        with CorpusStore(args.state) as corpus, record_run(corpus, "references", limit=args.limit) as run:
            pending = corpus.papers_without_references(args.limit)
            read = entries = 0
            for arxiv_id in pending:
                row = corpus.connection.execute(
                    "SELECT local_path, sha256 FROM html_sources WHERE arxiv_id = ?",
                    (arxiv_id,),
                ).fetchone()
                if row is None or row["local_path"] is None:
                    continue
                try:
                    found = references_from_file(Path(row["local_path"]), arxiv_id)
                except OSError as error:
                    print(f"  ! {arxiv_id}: {error}")
                    continue
                corpus.record_references(found, row["sha256"], for_papers=[arxiv_id])
                read += 1
                entries += len(found)
                print(f"  read {read}/{len(pending)}", end="\r", flush=True)
            total = corpus.reference_count()
            reading_list = corpus.most_cited_absent_papers(limit=args.reading_list)
        print(
            f"Read {entries} references from {read} papers; corpus holds {total}."
        )
        if reading_list:
            print("\nMost cited by this corpus and absent from it:")
            for cited, citations in reading_list:
                print(f"  {citations:>4} citing papers   arXiv:{cited}")

    if args.command == "serve":
        from verbatim.web import serve as serve_http

        serve_http(
            state=args.state, embedding_model=args.model,
            host=args.host, port=args.port,
        )

    if args.command == "sample":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        model_id = FastEmbedEmbedder(model_name=args.model).model_id
        report = export_sample(
            args.state, args.out, papers=args.papers, model_id=model_id
        )
        print(f"Wrote {args.out}: {report.render()}")
        print(f"Try it with:  verbatim search 'your question' --state {args.out}")

    if args.command == "pipeline":
        fingerprint = chunker_fingerprint(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        model_id = FastEmbedEmbedder(model_name=args.model).model_id

        def pending() -> dict[str, int]:
            with CorpusStore(args.state) as corpus:
                return corpus.pipeline_stats(fingerprint, model_id)

        rounds = run_pipeline(
            pending,
            state=str(args.state),
            max_rounds=args.max_rounds,
            # The stages that make and index vectors use the model counted above.
            stage_arguments={
                "embed": ("--model", args.model),
                "index": ("--model", args.model),
            },
            on_stage=lambda stage: print(
                f"  {'ok ' if stage.ok else 'FAILED'} {stage.name:<12}"
                f" {stage.pending_before:>7,} -> {stage.pending_after:>7,}",
                flush=True,
            ),
        )
        print()
        print(render_rounds(rounds))

    if args.command == "mcp":
        from verbatim.mcp_server import serve

        serve(state=args.state, embedding_model=args.model)

    if args.command == "runs":
        with CorpusStore(args.state) as corpus:
            listed = recent_runs(corpus, command=args.command_filter, limit=args.limit)
        if args.json:
            print(json.dumps(listed, indent=1))
        else:
            print(render_runs(listed))
        # A driver can stop on this rather than parsing prose.
        if any(run["ok"] is False for run in listed):
            raise SystemExit("error: the ledger contains a failed run")

    if args.command == "graph":
        from verbatim.tools import get_citations

        with CorpusStore(args.state) as corpus:
            if args.identifier is None:
                stats = corpus.citation_graph_stats()
                top = corpus.most_cited_held_papers(limit=args.top)
                degree = 2 * stats["edges"] / stats["papers"] if stats["papers"] else 0.0
                print(
                    f"  {stats['edges']:,} edges between held papers; "
                    f"{stats['connected_papers']:,} of {stats['papers']:,} papers "
                    f"({stats['connected_papers'] / (stats['papers'] or 1):.0%}) touch at "
                    f"least one; average degree {degree:.2f}"
                )
                if top:
                    print("\n  Most cited by this corpus, of the papers it holds:")
                    for cited_paper, citations in top:
                        print(
                            f"  {citations:>4} citing papers   "
                            f"arXiv:{cited_paper.arxiv_id}  {cited_paper.title}"
                        )
                return
            citers = get_citations(corpus, args.identifier, direction="cited_by", limit=args.limit)
            if citers is None:
                raise SystemExit(f"error: no paper in the corpus matches {args.identifier!r}")
            builds_on = get_citations(corpus, args.identifier, direction="references", limit=args.limit)
        print(f"arXiv:{citers['arxiv_id']}  {citers['title']}")
        # The identifier just resolved; the fallback guards a concurrent rewrite.
        cited = builds_on["papers"] if builds_on else []
        for label, related in (
            ("builds on (held here)", cited),
            ("cited by (held here)", citers["papers"]),
        ):
            print(f"\n  {label}: {len(related)}")
            for entry in related:
                print(f"    arXiv:{entry['arxiv_id']}  {entry['title']}")
        print(
            "\n  Only citations resolving into the corpus are shown; the paper's"
            "\n  full bibliography is longer."
        )

    if args.command == "grow":
        with CorpusStore(args.state) as corpus, record_run(corpus, "grow", limit=args.limit, min_citations=args.min_citations) as run:
            wanted = [
                (cited, citations)
                for cited, citations in corpus.most_cited_absent_papers(limit=args.limit)
                if citations >= args.min_citations
            ]
            if not wanted:
                print(
                    "Nothing to add. Run 'verbatim references' first, or lower "
                    "--min-citations."
                )
                return
            if args.dry_run:
                print(f"Would add {len(wanted)} papers:")
                for cited, citations in wanted:
                    print(f"  {citations:>4} citing papers   arXiv:{cited}")
                return
            added = missing = 0
            for start in range(0, len(wanted), MAX_IDS_PER_REQUEST):
                batch = [cited for cited, _ in wanted[start : start + MAX_IDS_PER_REQUEST]]
                if start:
                    time.sleep(args.delay)
                papers = fetch_papers_by_id(batch)
                # arXiv silently omits identifiers it does not recognise, and a
                # reference can name a withdrawn or mistyped one.
                missing += len(batch) - len(papers)
                summary = corpus.upsert_papers(papers)
                added += summary.inserted
                print(f"  added {added}/{len(wanted)}", end="\r", flush=True)
        print(
            f"Added {added} papers the corpus was citing and did not hold"
            + (f"; {missing} identifiers arXiv did not recognise." if missing else ".")
        )
        print("Their PDFs and HTML are not fetched yet: run download, html, chunk, embed.")

    if args.command == "embed":
        embedder = FastEmbedEmbedder(
            model_name=args.model, with_context=args.with_context
        )
        with CorpusStore(args.state) as corpus, record_run(corpus, "embed", model=args.model, batch_size=args.batch_size, limit=args.limit) as run:
            embedded = embed_pending(
                corpus,
                embedder,
                batch_size=args.batch_size,
                limit=args.limit,
                chunker_version=(
                    chunker_fingerprint(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                    if args.only_current else None
                ),
                on_progress=lambda done, total: print(
                    f"  embedded {done}/{total}", end="\r", flush=True
                ),
            )
            # Abstracts are embedded in the same pass, so the two indexes never drift.
            papers = embed_pending_papers(
                corpus,
                embedder,
                batch_size=args.batch_size,
                on_progress=lambda done, total: print(
                    f"  embedded {done}/{total} abstracts", end="\r", flush=True
                ),
            )
            model = embedder.model
            stored = corpus.embedding_count(model.model_id)
            stored_papers = corpus.paper_embedding_count(model.model_id)
            run.record(chunks_embedded=embedded, abstracts_embedded=papers,
                       vectors=stored, model=model.model_id)
        print(
            f"Embedded {embedded} new chunks and {papers} new abstracts with "
            f"{model.model_id} ({model.dimensions}d); corpus holds {stored} "
            f"passage vectors and {stored_papers} paper vectors."
        )

    if args.command == "search":
        papers = PaperFilter(since=args.since, category=args.category)
        with CorpusStore(args.state) as corpus:
            if args.mode == "keyword":
                matches = corpus.search_chunks(
                    args.query,
                    limit=args.limit,
                    arxiv_id=args.arxiv_id,
                    match=args.match,
                    include_references=args.include_references,
                    papers=papers,
                )
            else:
                search = vector_search if args.mode == "vector" else hybrid_search
                extra = (
                    {
                        "use_titles": args.use_titles,
                        "use_graph": args.use_graph,
                        "use_compare": args.use_compare,
                    }
                    if args.mode == "hybrid"
                    else {}
                )
                matches = search(
                    corpus,
                    FastEmbedEmbedder(model_name=args.model),
                    args.query,
                    limit=args.limit,
                    arxiv_id=args.arxiv_id,
                    include_references=args.include_references,
                    papers=papers,
                    **extra,
                )
        if args.rerank and matches:
            matches = rerank(
                CrossEncoderReranker(), args.query, matches, limit=args.limit
            )
        if not matches:
            print(f"No passages matched {args.query!r}.")
        for rank, match in enumerate(matches, start=1):
            print(format_match(rank, match, full=args.full))
    if args.command == "ask":
        # Refuse flags that would be ignored rather than silently drop them.
        if args.supervisor != "builtin" and not args.deep:
            raise SystemExit(
                "error: --supervisor only applies with --deep; without it "
                "`ask` climbs no ladder. Add --deep or drop --supervisor."
            )
        if args.deep and args.tools:
            raise SystemExit(
                "error: --tools has no effect with --deep; the agent is the "
                "last rung of the ladder. Drop one of the two."
            )
        if args.deep and args.use_compare:
            raise SystemExit(
                "error: --use-compare has no effect with --deep; the shape "
                "router chooses the strategy. Drop one of the two."
            )
        if args.deep and args.plan:
            raise SystemExit(
                "error: --plan has no effect with --deep; the ladder answers "
                "one question rather than splitting it. Drop one of the two."
            )
        if args.deep:
            _refuse_ignored("--deep", "each rung of the ladder sets its own retrieval", {
                "--arxiv-id": args.arxiv_id is not None,
                "--include-references": args.include_references,
                "--no-graph": not args.use_graph,
            })
        if args.tools:
            _refuse_ignored("--tools", "the model chooses what to search and how", {
                "--arxiv-id": args.arxiv_id is not None,
                "--include-references": args.include_references,
                "--no-graph": not args.use_graph,
                "--use-compare": args.use_compare,
                "--plan": args.plan,
                "--passages": args.passages != config.ASK_PASSAGES,
                "--max-per-paper": args.max_per_paper != config.ASK_MAX_PER_PAPER,
                "--no-subject-check": args.no_subject_check,
            })
        with CorpusStore(args.state) as corpus, record_run(corpus, "ask", model=args.model, llm=args.llm, passages=args.passages) as run:
            if args.deep:
                from verbatim.orchestrator import _supervisor

                embedder = FastEmbedEmbedder(model_name=args.model)
                completion = OllamaCompletion(args.llm or config.OLLAMA_MODEL)
                # The ladder's last rung is the tool-choosing agent, which needs a caller.
                tool_caller = OllamaToolCaller(args.llm or config.OLLAMA_MODEL)
                section, ledger = _supervisor(args.supervisor)(
                    corpus, embedder, completion, args.question,
                    caller=tool_caller,
                    per_section=args.passages,
                    max_per_paper=args.max_per_paper,
                    check_subject=not args.no_subject_check,
                )
                answer = section.answer
                run.record(citations=len(answer.citations),
                           rejected=len(answer.rejected),
                           declined=answer.declined,
                           attempts=len(ledger.attempts),
                           outcome=ledger.outcome)
                print(answer.text)
                if answer.citations:
                    print("\nReferences")
                    print(render_bibliography(answer.citations))
                # Always show what was tried, not just what worked.
                print(f"\n  {ledger.shape} question, {len(ledger.attempts)} attempt(s):")
                for attempt in ledger.attempts:
                    detail = attempt.error or attempt.verdict
                    print(f"    {attempt.strategy:<8} {detail}")
                for item in answer.rejected:
                    print(f"\n! rejected citation ({item.reason}): {item.quoted_text[:80]!r}")
                print(f"\n  {_usage_footer(completion, tool_caller)}")
                return
            if args.tools:
                embedder = FastEmbedEmbedder(model_name=args.model)
                completion = OllamaCompletion(args.llm or config.OLLAMA_MODEL)
                tool_caller = OllamaToolCaller(args.llm or config.OLLAMA_MODEL)
                answer, trace = answer_question(
                    corpus, embedder, tool_caller, completion, args.question,
                )
                print(f"[route: {trace.describe()}]\n")
                print(answer.text)
                if answer.citations:
                    print("\nReferences")
                    print(render_bibliography(answer.citations))
                for item in answer.rejected:
                    print(f"\n! rejected citation ({item.reason}): {item.quoted_text[:80]!r}")
                # Routing turns are model calls too, so both clients are billed.
                print(f"\n  {_usage_footer(completion, tool_caller)}")
                return
            embedder = FastEmbedEmbedder(model_name=args.model)
            completion = OllamaCompletion(args.llm or config.OLLAMA_MODEL)
            plan: list[str] = []
            if args.plan:
                answer, plan, matches = research(
                    corpus, embedder, completion,
                    lambda system, user: completion.complete_text(system, user),
                    args.question,
                    passages=args.passages,
                    # Every flag plain `ask` honours applies to each part.
                    max_per_paper=args.max_per_paper,
                    arxiv_id=args.arxiv_id,
                    include_references=args.include_references,
                    use_graph=args.use_graph,
                    use_compare=args.use_compare,
                )
            else:
                matches = hybrid_search(
                    corpus,
                    embedder,
                    args.question,
                    limit=args.passages,
                    arxiv_id=args.arxiv_id,
                    max_per_paper=args.max_per_paper,
                    include_references=args.include_references,
                    use_graph=args.use_graph,
                    use_compare=args.use_compare,
                )
                answer = answer_from_matches(corpus, completion, args.question, matches)
            if len(plan) > 1:
                # Show the split, so a reader can disagree with it.
                print("Answered in parts:")
                for part in plan:
                    print(f"  - {part}")
                print()
            run.record(passages=len(matches),
                       papers=len({m.chunk.arxiv_id for m in matches}),
                       citations=len(answer.citations),
                       rejected=len(answer.rejected),
                       declined=answer.declined)
            subject = None
            closeness = None
            if matches and not args.no_subject_check:
                # Ask what the passages discuss, not whether they are relevant.
                subject, closeness = subject_check(
                    completion, embedder, args.question, matches
                )
            # Computed while the corpus is open; printed with the answer.
            evidence = describe_evidence(
                corpus, args.question, matches, subject=subject, closeness=closeness
            )
        print(answer.text)
        if answer.citations:
            print("\nReferences")
            print(render_bibliography(answer.citations))
        # Always say what the answer rests on: some failures are invisible to
        # every detector tried, so the reader judges.
        for line in evidence.lines():
            print(f"  {line}")
        if subject and closeness is not None and closeness < config.SUBJECT_WARN_BELOW:
            # A warning, never a suppression: the similarity ranges overlap.
            print(f"\n  ! these passages are about: {subject}")
            print(f"    which may not be what you asked ({closeness:.2f} similarity)")
        for item in answer.rejected:
            # A rejected citation is what this tool exists to catch, so it is shown.
            print(f"\n! rejected citation ({item.reason}): {item.quoted_text[:80]!r}")
        print(f"\n  {_usage_footer(completion)}")

    if args.command == "report":
        from verbatim.report import build_report, render_markdown

        if args.supervisor != "builtin" and not args.agents:
            raise SystemExit(
                "error: --supervisor only applies with --agents; the plain "
                "report runs a single rung and climbs no ladder. Add --agents "
                "or drop --supervisor."
            )
        if args.agents and args.use_compare:
            # Refused rather than ignored: with --agents the router picks each strategy.
            raise SystemExit(
                "error: --use-compare has no effect with --agents; the shape "
                "router chooses each section's strategy. Drop one of the two."
            )
        embedder = FastEmbedEmbedder(model_name=args.model)
        completion = OllamaCompletion(args.llm or config.OLLAMA_MODEL)
        def progress(heading: str, done: int, total: int) -> None:
            print(f"  [{done}/{total}] {heading}", flush=True)
        ledger = None
        with CorpusStore(args.state) as corpus, record_run(corpus, "report", model=args.model, llm=args.llm, sections=args.sections, agents=args.agents) as run:
            if args.agents:
                from verbatim.contracts import Budget
                from verbatim.orchestrator import orchestrate

                caller = OllamaToolCaller(args.llm or config.OLLAMA_MODEL)
                ceiling = Budget(
                    max_tokens=args.max_tokens, max_seconds=args.max_seconds,
                    max_attempts=args.max_attempts,
                )
                built, ledger = orchestrate(
                    corpus,
                    embedder,
                    completion,
                    lambda system, user: completion.complete_text(system, user),
                    args.topic,
                    caller=caller,
                    max_sections=args.sections,
                    per_section=args.passages,
                    max_per_paper=args.max_per_paper,
                    workers=args.workers,
                    check_subject=not args.no_subject_check,
                    budget=ceiling if any(
                        (args.max_tokens, args.max_seconds, args.max_attempts)
                    ) else None,
                    critic=args.critic,
                    on_section=progress,
                    supervisor=args.supervisor,
                )
            else:
                built = build_report(
                    corpus,
                    embedder,
                    completion,
                    lambda system, user: completion.complete_text(system, user),
                    args.topic,
                    max_sections=args.sections,
                    per_section=args.passages,
                    max_per_paper=args.max_per_paper,
                    use_compare=args.use_compare,
                    check_subject=not args.no_subject_check,
                    on_section=progress,
                )
            run.record(
                sections=len(built.sections),
                citations=sum(len(s.answer.citations) for s in built.sections),
                rejected=sum(len(s.answer.rejected) for s in built.sections),
                declined=sum(1 for s in built.sections if s.answer.declined),
                escalations=ledger.escalations if ledger else 0,
            )
        # The agent rung's routing calls are billed with the rest.
        billed = (completion, caller) if args.agents else (completion,)
        markdown = render_markdown(built, ledger=ledger)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(markdown + "\n", encoding="utf-8")
            print(f"Wrote {args.out}")
        else:
            print()
            print(markdown)
        print(f"\n  {_usage_footer(*billed)}")

    if args.command == "chunk":
        if args.manifest is not None:
            chunks = extract_pdf_chunks(
                load_download_manifest(args.manifest),
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
            save_chunks(chunks, args.output)
            print(f"Saved {len(chunks)} page-aware chunks to {args.output}")
        else:
            fingerprint = chunker_fingerprint(args.chunk_size, args.overlap)
            with CorpusStore(args.state) as corpus, record_run(corpus, "chunk", limit=args.limit, batch_size=args.batch_size) as run:
                # HTML first: a paper with a rendering is never chunked from its PDF.
                html_pending = corpus.html_sources_pending_chunking(
                    fingerprint, limit=args.limit
                )
                html_written = 0
                for start in range(0, len(html_pending), args.batch_size):
                    batch = html_pending[start : start + args.batch_size]
                    produced = extract_html_chunks(
                        batch, chunk_size=args.chunk_size, overlap=args.overlap
                    )
                    corpus.record_chunks(
                        produced, for_papers=[source.arxiv_id for source in batch]
                    )
                    corpus.mark_html_chunked(batch, fingerprint)
                    html_written += len(produced)
                    print(f"  chunked {min(start + len(batch), len(html_pending))}"
                          f"/{len(html_pending)} HTML", end="\r", flush=True)
                remaining = None if args.limit is None else max(args.limit - len(html_pending), 0)
                pending = (
                    [] if remaining == 0
                    else corpus.pending_chunking(
                        fingerprint,
                        limit=remaining,
                        require_html_checked=args.only_checked,
                    )
                )
                # Batches bound peak memory (about 11 MB per PDF) and commit as they
                # go, so an interrupted run resumes.
                written = 0
                for start in range(0, len(pending), args.batch_size):
                    batch = pending[start : start + args.batch_size]
                    produced = extract_pdf_chunks(
                        batch, chunk_size=args.chunk_size, overlap=args.overlap
                    )
                    corpus.record_chunks(
                        produced, for_papers=[download.arxiv_id for download in batch]
                    )
                    corpus.mark_chunked(batch, fingerprint)
                    written += len(produced)
                    print(
                        f"  chunked {min(start + len(batch), len(pending))}/{len(pending)} PDFs",
                        end="\r",
                        flush=True,
                    )
                exported = save_chunks(corpus.iter_chunks(), args.output)
            print(
                f"Chunked {len(html_pending)} HTML renderings and {len(pending)} PDFs "
                f"into {html_written + written} new passages; corpus holds "
                f"{exported}. Exported to {args.output}"
            )


def _refuse_ignored(mode: str, reason: str, given: dict[str, bool]) -> None:
    """Refuse flags a routed `ask` would parse and then silently ignore."""
    named = [flag for flag, present in given.items() if present]
    if named:
        raise SystemExit(
            f"error: {', '.join(named)} has no effect with {mode}; {reason}. "
            f"Drop {'it' if len(named) == 1 else 'them'}, or drop {mode}."
        )


def _percent(part: int, whole: int) -> str:
    return f"{part / max(whole, 1):.0%}"


def _usage_footer(*clients: object) -> str:
    """What the run cost across every client used; clients without usage are skipped."""
    counted = [getattr(client, "usage", None) for client in clients]
    present = [usage for usage in counted if usage is not None]
    if not present:
        return "cost not reported"
    calls = sum(u.calls for u in present)
    return (
        f"{calls} model call{'s' if calls != 1 else ''}, "
        f"{sum(u.prompt_tokens for u in present):,} tokens in, "
        f"{sum(u.output_tokens for u in present):,} out, "
        f"{sum(u.seconds for u in present):.1f}s in the model"
    )


def _line_completion(model: str | None) -> Callable[[str, str], str]:
    """A plain-text completion, for query rewriting."""
    completion = OllamaCompletion(model or config.OLLAMA_MODEL)
    return lambda system, user: completion.complete_text(system, user)


def format_match(rank: int, match: ChunkMatch, *, full: bool = False) -> str:
    """Render one hit with the paper version, location and URL it came from."""
    chunk = match.chunk
    where_in_paper = render_locator(
        chunk.source_kind, chunk.page_number, chunk.section_path
    )
    header = (
        f"{rank}. arXiv:{chunk.arxiv_id}  {where_in_paper}  "
        f"[chars {chunk.char_start}-{chunk.char_end}]  score {match.score:.2f}"
    )
    # The deepest heading only; the outermost is the paper title.
    section = chunk.section_path.split(" > ")[-1] if chunk.section_path else ""
    where = f"   {chunk.source_url}  ({chunk.license_url or 'no licence reported'})"
    if section and section not in where_in_paper:
        where += f"\n   section: {section}"
    body = chunk.text if full else match.snippet
    return f"{header}\n{where}\n   {body}\n"


def _merge_downloads(
    existing_downloads: list[DownloadedPaper], new_downloads: list[DownloadedPaper]
) -> list[DownloadedPaper]:
    """Merge snapshot downloads, keeping the newest record per paper."""
    by_id = {download.arxiv_id: download for download in existing_downloads}
    by_id.update({download.arxiv_id: download for download in new_downloads})
    return list(by_id.values())


def html_completeness(corpus: CorpusStore, source: HtmlSource) -> float:
    """How much of its PDF's text a rendering covers, by length.

    Plain extraction suffices: only a length is needed, and layout extraction is slow.
    """
    download = corpus.download_for(source.arxiv_id)
    if download is None:
        return 1.0  # no PDF to compare against
    try:
        pdf_text = normalize_page_text(
            " ".join(extract_plain_pages_from_pdf(Path(download.local_path)))
        )
        html_text = " ".join(
            section.text
            for section in parse_html_sections(
                Path(source.local_path).read_text(encoding="utf-8", errors="replace")
            )
        )
    except (OSError, RuntimeError, ValueError):
        return 1.0  # unreadable PDF is not evidence against the rendering
    return completeness(len(html_text), len(pdf_text))


def format_stats(counts: dict[str, int]) -> str:
    """Render pipeline counts as a table of done-versus-remaining per stage."""
    gigabytes = counts["bytes_on_disk"] / 1e9
    rows = [
        ("papers", f"{counts['papers']:,}", f"{counts['papers_current']:,} current", ""),
        ("licences", f"{counts['licences_known']:,} known",
         f"{counts['licences_pending']:,} to look up", "verbatim licenses"),
        ("pdfs",
         f"{counts['downloaded']:,} ({gigabytes:.1f} GB)"
         + (f", {counts['downloads_unavailable']:,} arXiv has none for"
            if counts["downloads_unavailable"] else ""),
         f"{counts['downloads_pending']:,} to fetch", "verbatim download"),
        ("html", f"{counts['html_available']:,} ({counts['html_absent']:,} have none)",
         f"{counts['html_pending']:,} to check", "verbatim html"),
        ("references", f"{counts['references']:,}",
         f"{counts['references_pending']:,} papers to read", "verbatim references"),
        ("chunks", f"{counts['chunks']:,} "
                   f"({counts['chunks_body']:,} body / {counts.get('chunks_tables', 0):,} tables"
                   f" / {counts['chunks_references']:,} refs)",
         f"{counts['chunking_pending']:,} papers to chunk", "verbatim chunk"),
        ("vectors", f"{counts['vectors']:,}",
         f"{counts['embedding_pending']:,} to embed", "verbatim embed"),
    ]
    width = max(len(row[1]) for row in rows)
    lines = [f"  {'stage':<10} {'done':<{width}}  {'remaining':<22} next"]
    for stage, done, remaining, command in rows:
        lines.append(f"  {stage:<10} {done:<{width}}  {remaining:<22} {command}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - module entry point
    # Needed for `python -m verbatim.cli`, which the pipeline uses to run stages.
    main()
