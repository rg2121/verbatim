"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from verbatim.agent import answer_question
from verbatim.answers import (
    HostedCompletion,
    describe_subject,
    subject_similarity,
    OllamaCompletion,
    OllamaToolCaller,
    answer_from_matches,
)
from verbatim.arxiv import fetch_license, fetch_metadata, license_class
from verbatim.sync import format_sync_report, sync_category
from verbatim.citations import render_bibliography
from verbatim.chunking import chunker_fingerprint, extract_pdf_chunks
from verbatim.corpus import ChunkMatch, CorpusStore
from verbatim import config
import hashlib

from verbatim.embedding import FastEmbedEmbedder, embed_pending
from verbatim.html_source import (
    HtmlSource,
    extract_html_chunks,
    fetch_html,
    html_url_for,
)
from verbatim.evaluation import evaluate, load_cases, render_report
from verbatim.reranking import CrossEncoderReranker, rerank
from verbatim.retrieval import hybrid_search, vector_search
from verbatim.vectors import SqliteVecIndex
from verbatim.downloads import DownloadedPaper, DownloadReport, download_papers
from verbatim.storage import (
    load_download_manifest,
    load_papers,
    save_chunks,
    save_download_manifest,
    save_papers,
)


def main() -> None:
    """Entry point: report expected failures as messages, not tracebacks."""
    try:
        run()
    except (RuntimeError, ValueError, OSError) as error:
        raise SystemExit(f"error: {error}") from error
    except KeyboardInterrupt:
        raise SystemExit("interrupted") from None


def run() -> None:
    parser = argparse.ArgumentParser(description="Ingest a small, traceable arXiv metadata snapshot.")
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
    eval_parser.add_argument("--cases", type=Path, default=config.EVAL_PATH)
    eval_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    eval_parser.add_argument(
        "--rerank", action="store_true",
        help="Re-score with a cross-encoder. Measured worse: mrr 0.940 -> 0.752",
    )
    eval_parser.add_argument("--model", default=config.EMBEDDING_MODEL, help="Embedding model")
    eval_parser.add_argument("--limit", type=int, default=10, help="Results per query")
    eval_parser.add_argument("--max-per-paper", type=int, help="Diversity cap for hybrid")

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

    embed_parser = subparsers.add_parser("embed", help="Embed chunks that have no vector yet")
    embed_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    embed_parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    embed_parser.add_argument("--batch-size", type=int, default=64)
    embed_parser.add_argument("--limit", type=int, help="Embed at most this many chunks this run")

    search_parser = subparsers.add_parser("search", help="Keyword-search the chunked corpus")
    search_parser.add_argument("query", help="Search terms; 'term*' matches by prefix")
    search_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    search_parser.add_argument(
        "--rerank", action="store_true",
        help="Re-score with a cross-encoder. Measured worse: mrr 0.940 -> 0.752",
    )
    search_parser.add_argument("--limit", type=int, default=config.SEARCH_LIMIT, help="Maximum number of passages")
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
        help="keyword mode only: rank by matched terms (default), or require every term. "
             "Requiring every term scored mrr 0.214 against 0.845 on the eval set",
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
        "--backend",
        choices=("ollama", "hosted"),
        default=config.ANSWER_BACKEND,
        help="Which model answers: a local Ollama model (default) or a hosted API",
    )
    ask_parser.add_argument(
        "--llm",
        help="Generation model. Ollama: defaults to $OLLAMA_MODEL or the first pulled model",
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

    chunk_parser = subparsers.add_parser("chunk", help="Extract page-aware text chunks from downloaded PDFs")
    chunk_parser.add_argument("--manifest", type=Path, help="Legacy one-off PDF manifest; omit to use --state")
    chunk_parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    chunk_parser.add_argument("--output", type=Path, default=config.CHUNKS_PATH)
    chunk_parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    chunk_parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    chunk_parser.add_argument("--limit", type=int, help="Chunk at most this many PDFs this run")
    chunk_parser.add_argument("--batch-size", type=int, default=config.CHUNK_BATCH_SIZE,
                              help="PDFs held in memory at once; bounds peak memory")
    args = parser.parse_args()

    if args.command == "fetch":
        papers = fetch_metadata(args.category, args.max_results)
        save_papers(papers, args.output)
        print(f"Saved {len(papers)} papers to {args.output}")

    if args.command == "sync":
        with CorpusStore(args.state) as corpus:
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
            with CorpusStore(args.state) as corpus:
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
                downloads = corpus.all_downloads()
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
        with CorpusStore(args.state) as corpus:
            counts = corpus.pipeline_stats(
                chunker_fingerprint(args.chunk_size, args.overlap), args.model
            )
            synced = corpus.sync_summary()
        print(format_stats(counts))
        if synced:
            print("\n  synced categories")
            for category, watermark, seen in synced:
                mark = (watermark or "never")[:19]
                print(f"    {category:<18} {seen:>7,} ingested   up to {mark}")

    if args.command == "eval":
        cases = load_cases(args.cases)
        embedder = FastEmbedEmbedder(model_name=args.model)
        with CorpusStore(args.state) as corpus:
            scores = evaluate(
                corpus, embedder, cases,
                limit=args.limit, max_per_paper=args.max_per_paper,
                reranker=CrossEncoderReranker() if args.rerank else None,
            )
        print(f"{len(cases)} cases from {args.cases}\n")
        print(render_report(scores))

    if args.command == "index":
        embedder = FastEmbedEmbedder(model_name=args.model)
        with CorpusStore(args.state) as corpus:
            model = embedder.model
            index = SqliteVecIndex(corpus.connection, model.dimensions)
            if args.rebuild:
                print(f"Rebuilt the index with {index.rebuild(model.model_id)} vectors.")
            else:
                change = index.sync(model.model_id)
                print(f"Index synced: {change.added} added, {change.removed} removed; "
                      f"{index.count()} vectors total.")

    if args.command == "licenses":
        with CorpusStore(args.state) as corpus:
            pending = corpus.base_ids_missing_license(args.limit)
            for position, base_id in enumerate(pending):
                if position:
                    time.sleep(args.delay)  # arXiv asks callers to pace themselves
                try:
                    corpus.record_license(base_id, fetch_license(base_id))
                except (OSError, ValueError) as error:
                    print(f"  ! {base_id}: {error}")
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
        fetched = missing = 0
        with CorpusStore(args.state) as corpus:
            pending = corpus.papers_without_html(args.limit)
            for position, arxiv_id in enumerate(pending):
                if position:
                    time.sleep(args.delay)
                try:
                    payload = fetch_html(arxiv_id)
                except (OSError, RuntimeError, ValueError) as error:
                    print(f"  ! {arxiv_id}: {error}")
                    continue
                if payload is None:
                    # No TeX source, so no rendering. The PDF remains the
                    # source, and the absence is recorded so this paper is
                    # not asked for again on every later run.
                    corpus.record_html_absent(arxiv_id, html_url_for(arxiv_id))
                    missing += 1
                    continue
                path = args.output_dir / f"{arxiv_id.replace('/', '-')}.html"
                path.write_bytes(payload)
                corpus.record_html_source(
                    HtmlSource(
                        arxiv_id=arxiv_id, url=html_url_for(arxiv_id), local_path=str(path),
                        sha256=hashlib.sha256(payload).hexdigest(), byte_count=len(payload),
                    )
                )
                fetched += 1
                print(f"  fetched {fetched}/{len(pending)}", end="\r", flush=True)
        print(f"Fetched {fetched} HTML renderings; {missing} papers have none (PDF-only).")

    if args.command == "embed":
        embedder = FastEmbedEmbedder(model_name=args.model)
        with CorpusStore(args.state) as corpus:
            embedded = embed_pending(
                corpus,
                embedder,
                batch_size=args.batch_size,
                limit=args.limit,
                on_progress=lambda done, total: print(
                    f"  embedded {done}/{total}", end="\r", flush=True
                ),
            )
            model = embedder.model
            stored = corpus.embedding_count(model.model_id)
        print(
            f"Embedded {embedded} new chunks with {model.model_id} "
            f"({model.dimensions}d); corpus holds {stored} vectors."
        )

    if args.command == "search":
        with CorpusStore(args.state) as corpus:
            if args.mode == "keyword":
                matches = corpus.search_chunks(
                    args.query,
                    limit=args.limit,
                    arxiv_id=args.arxiv_id,
                    match=args.match,
                    include_references=args.include_references,
                )
            else:
                search = vector_search if args.mode == "vector" else hybrid_search
                matches = search(
                    corpus,
                    FastEmbedEmbedder(model_name=args.model),
                    args.query,
                    limit=args.limit,
                    arxiv_id=args.arxiv_id,
                    include_references=args.include_references,
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
        with CorpusStore(args.state) as corpus:
            if args.tools:
                embedder = FastEmbedEmbedder(model_name=args.model)
                completion = (
                    OllamaCompletion(args.llm or config.OLLAMA_MODEL)
                    if args.backend == "ollama"
                    else HostedCompletion(args.llm or config.HOSTED_MODEL)
                )
                answer, trace = answer_question(
                    corpus, embedder, OllamaToolCaller(args.llm or config.OLLAMA_MODEL),
                    completion, args.question,
                )
                print(f"[route: {trace.describe()}]\n")
                print(answer.text)
                if answer.citations:
                    print("\nReferences")
                    print(render_bibliography(answer.citations))
                for item in answer.rejected:
                    print(f"\n! rejected citation ({item.reason}): {item.quoted_text[:80]!r}")
                return
            embedder = FastEmbedEmbedder(model_name=args.model)
            matches = hybrid_search(
                corpus,
                embedder,
                args.question,
                limit=args.passages,
                arxiv_id=args.arxiv_id,
                max_per_paper=args.max_per_paper,
                include_references=args.include_references,
            )
            completion = (
                OllamaCompletion(args.llm or config.OLLAMA_MODEL)
                if args.backend == "ollama"
                else HostedCompletion(args.llm or config.HOSTED_MODEL)
            )
            answer = answer_from_matches(corpus, completion, args.question, matches)
            subject = None
            if matches and not args.no_subject_check:
                # Asked what the passages discuss rather than whether they are
                # relevant: every model tested says yes to the latter.
                subject = describe_subject(completion, matches)
                closeness = subject_similarity(embedder, args.question, subject)
        print(answer.text)
        if answer.citations:
            print("\nReferences")
            print(render_bibliography(answer.citations))
        if subject is not None and closeness < config.SUBJECT_WARN_BELOW:
            # A warning, never a suppression: the ranges for answerable and
            # unanswerable questions overlap, so this cannot decide, only show.
            print(f"\n  ! these passages are about: {subject}")
            print(f"    which may not be what you asked ({closeness:.2f} similarity)")
        for item in answer.rejected:
            # Loud on purpose: an unverifiable citation is the failure this
            # tool exists to catch.
            print(f"\n! rejected citation ({item.reason}): {item.quoted_text[:80]!r}")

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
            with CorpusStore(args.state) as corpus:
                # HTML first: it keeps maths as LaTeX and carries the author's
                # own sectioning, so a paper with a rendering is chunked from
                # that and never from its PDF.
                html_pending = corpus.html_sources_pending_chunking(
                    fingerprint, limit=args.limit
                )
                for start in range(0, len(html_pending), args.batch_size):
                    batch = html_pending[start : start + args.batch_size]
                    produced = extract_html_chunks(
                        batch, chunk_size=args.chunk_size, overlap=args.overlap
                    )
                    corpus.record_chunks(produced)
                    corpus.mark_html_chunked(batch, fingerprint)
                    print(f"  chunked {min(start + len(batch), len(html_pending))}"
                          f"/{len(html_pending)} HTML", end="\r", flush=True)
                remaining = None if args.limit is None else max(args.limit - len(html_pending), 0)
                pending = (
                    [] if remaining == 0
                    else corpus.pending_chunking(fingerprint, limit=remaining)
                )
                # Extraction holds roughly 11MB per PDF and never plateaus, so
                # chunking five thousand at once would need tens of gigabytes.
                # Batching bounds peak memory and commits as it goes, which also
                # makes an interrupted run resume rather than restart.
                written = 0
                for start in range(0, len(pending), args.batch_size):
                    batch = pending[start : start + args.batch_size]
                    produced = extract_pdf_chunks(
                        batch, chunk_size=args.chunk_size, overlap=args.overlap
                    )
                    corpus.record_chunks(produced)
                    corpus.mark_chunked(batch, fingerprint)
                    written += len(produced)
                    print(
                        f"  chunked {min(start + len(batch), len(pending))}/{len(pending)} PDFs",
                        end="\r",
                        flush=True,
                    )
                new_chunks = range(written + sum(1 for _ in html_pending))
                exported = save_chunks(corpus.iter_chunks(), args.output)
            print(
                f"Chunked {len(pending)} PDFs into {len(new_chunks)} new passages; "
                f"corpus holds {exported}. Exported to {args.output}"
            )


def format_match(rank: int, match: ChunkMatch, *, full: bool = False) -> str:
    """Render one hit so the passage can be traced back to a paper and page.

    The header is the citation: paper version, page, and the abs URL that
    resolves to exactly the version the text came from.
    """
    chunk = match.chunk
    header = (
        f"{rank}. arXiv:{chunk.arxiv_id}  p.{chunk.page_number}  "
        f"[chars {chunk.char_start}-{chunk.char_end}]  score {match.score:.2f}"
    )
    # The deepest heading, not the whole ancestry: the outermost element is the
    # paper title, which every passage shares and so tells the reader nothing.
    section = chunk.section_path.split(" > ")[-1] if chunk.section_path else ""
    where = f"   {chunk.source_url}  ({chunk.license_url or 'no licence reported'})"
    if section:
        where += f"\n   section: {section}"
    body = chunk.text if full else match.snippet
    return f"{header}\n{where}\n   {body}\n"


def _merge_downloads(
    existing_downloads: list[DownloadedPaper], new_downloads: list[DownloadedPaper]
) -> list[DownloadedPaper]:
    """Keep legacy snapshot downloads cumulative while callers migrate to SQLite."""
    by_id = {download.arxiv_id: download for download in existing_downloads}
    by_id.update({download.arxiv_id: download for download in new_downloads})
    return list(by_id.values())


def format_stats(counts: dict[str, int]) -> str:
    """Render pipeline counts as a table of done-versus-remaining per stage."""
    gigabytes = counts["bytes_on_disk"] / 1e9
    rows = [
        ("papers", f"{counts['papers']:,}", f"{counts['papers_current']:,} current", ""),
        ("licences", f"{counts['licences_known']:,} known",
         f"{counts['licences_pending']:,} to look up", "verbatim licenses"),
        ("pdfs", f"{counts['downloaded']:,} ({gigabytes:.1f} GB)",
         f"{counts['downloads_pending']:,} to fetch", "verbatim download"),
        ("chunks", f"{counts['chunks']:,} "
                   f"({counts['chunks_body']:,} body / {counts['chunks_references']:,} refs)",
         f"{counts['chunking_pending']:,} pdfs to chunk", "verbatim chunk"),
        ("vectors", f"{counts['vectors']:,}",
         f"{counts['embedding_pending']:,} to embed", "verbatim embed"),
    ]
    width = max(len(row[1]) for row in rows)
    lines = [f"  {'stage':<10} {'done':<{width}}  {'remaining':<22} next"]
    for stage, done, remaining, command in rows:
        lines.append(f"  {stage:<10} {done:<{width}}  {remaining:<22} {command}")
    return "\n".join(lines)
