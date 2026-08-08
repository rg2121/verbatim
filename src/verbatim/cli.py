"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from verbatim.answers import MODEL as HOSTED_MODEL
from verbatim.answers import HostedCompletion, OllamaCompletion, answer_from_matches
from verbatim.arxiv import fetch_license, fetch_metadata, license_class
from verbatim.citations import render_bibliography
from verbatim.chunking import chunker_fingerprint, extract_pdf_chunks
from verbatim.corpus import ChunkMatch, CorpusStore
from verbatim.embedding import DEFAULT_MODEL_NAME, FastEmbedEmbedder, embed_pending
from verbatim.retrieval import hybrid_search, vector_search
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
    fetch_parser.add_argument("--output", type=Path, default=Path("data/papers.json"))

    sync_parser = subparsers.add_parser("sync", help="Fetch metadata and upsert it into the cumulative corpus catalogue")
    sync_parser.add_argument("--category", required=True, help="arXiv category, e.g. physics.gen-ph")
    sync_parser.add_argument("--max-results", type=int, default=10, help="Number of recently updated papers to fetch")
    sync_parser.add_argument("--state", type=Path, default=Path("data/corpus.db"))
    sync_parser.add_argument("--output", type=Path, default=Path("data/papers.json"))

    download_parser = subparsers.add_parser("download", help="Download a bounded set of PDFs from a metadata snapshot")
    download_parser.add_argument("--input", type=Path, help="Legacy one-off metadata snapshot; omit to use --state")
    download_parser.add_argument("--state", type=Path, default=Path("data/corpus.db"))
    download_parser.add_argument("--limit", type=int, default=3, help="Maximum number of PDFs to download")
    download_parser.add_argument("--output-dir", type=Path, default=Path("data/pdfs"))
    download_parser.add_argument(
        "--require-license",
        action="store_true",
        help="Download only papers whose metadata explicitly reports a licence",
    )

    licenses_parser = subparsers.add_parser(
        "licenses", help="Look up licences via OAI-PMH, which the query API omits"
    )
    licenses_parser.add_argument("--state", type=Path, default=Path("data/corpus.db"))
    licenses_parser.add_argument("--limit", type=int, default=50, help="Papers to look up this run")
    licenses_parser.add_argument(
        "--delay", type=float, default=3.0, help="Seconds between requests; arXiv asks for a pause"
    )

    embed_parser = subparsers.add_parser("embed", help="Embed chunks that have no vector yet")
    embed_parser.add_argument("--state", type=Path, default=Path("data/corpus.db"))
    embed_parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    embed_parser.add_argument("--batch-size", type=int, default=64)
    embed_parser.add_argument("--limit", type=int, help="Embed at most this many chunks this run")

    search_parser = subparsers.add_parser("search", help="Keyword-search the chunked corpus")
    search_parser.add_argument("query", help="Search terms; 'term*' matches by prefix")
    search_parser.add_argument("--state", type=Path, default=Path("data/corpus.db"))
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum number of passages")
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
        choices=("all", "any"),
        default="all",
        help="keyword mode only: require every term (default) or any term",
    )
    search_parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Embedding model for --mode vector")
    search_parser.add_argument(
        "--include-references",
        action="store_true",
        help="Also search bibliographies, which are excluded by default",
    )

    ask_parser = subparsers.add_parser("ask", help="Answer a question from the corpus, with citations")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--state", type=Path, default=Path("data/corpus.db"))
    ask_parser.add_argument("--passages", type=int, default=8, help="Passages to retrieve")
    ask_parser.add_argument("--arxiv-id", help="Restrict the answer to one paper version")
    ask_parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Embedding model")
    ask_parser.add_argument(
        "--backend",
        choices=("ollama", "hosted"),
        default="ollama",
        help="Which model answers: a local Ollama model (default) or a hosted API",
    )
    ask_parser.add_argument(
        "--llm",
        help="Generation model. Ollama: defaults to $OLLAMA_MODEL or the first pulled model",
    )
    ask_parser.add_argument(
        "--include-references",
        action="store_true",
        help="Also search bibliographies, which are excluded by default",
    )
    ask_parser.add_argument(
        "--max-per-paper",
        type=int,
        default=3,
        help="Cap passages taken from any one paper, so one source cannot dominate",
    )

    chunk_parser = subparsers.add_parser("chunk", help="Extract page-aware text chunks from downloaded PDFs")
    chunk_parser.add_argument("--manifest", type=Path, help="Legacy one-off PDF manifest; omit to use --state")
    chunk_parser.add_argument("--state", type=Path, default=Path("data/corpus.db"))
    chunk_parser.add_argument("--output", type=Path, default=Path("data/chunks.jsonl"))
    chunk_parser.add_argument("--chunk-size", type=int, default=1_200)
    chunk_parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    if args.command == "fetch":
        papers = fetch_metadata(args.category, args.max_results)
        save_papers(papers, args.output)
        print(f"Saved {len(papers)} papers to {args.output}")

    if args.command == "sync":
        papers = fetch_metadata(args.category, args.max_results)
        save_papers(papers, args.output)
        with CorpusStore(args.state) as corpus:
            summary = corpus.upsert_papers(papers)
        print(f"Synced {len(papers)} papers ({summary.inserted} new, {summary.updated} refreshed) to {args.state}")

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
                papers = corpus.pending_downloads(args.limit, require_license=args.require_license)
                report = (
                    download_papers(
                        papers,
                        args.output_dir,
                        len(papers),
                        require_license=args.require_license,
                        existing_downloads=corpus.all_downloads(),
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
        if not matches:
            print(f"No passages matched {args.query!r}.")
        for rank, match in enumerate(matches, start=1):
            print(format_match(rank, match, full=args.full))
    if args.command == "ask":
        with CorpusStore(args.state) as corpus:
            matches = hybrid_search(
                corpus,
                FastEmbedEmbedder(model_name=args.model),
                args.question,
                limit=args.passages,
                arxiv_id=args.arxiv_id,
                max_per_paper=args.max_per_paper,
                include_references=args.include_references,
            )
            completion = (
                OllamaCompletion(args.llm)
                if args.backend == "ollama"
                else HostedCompletion(args.llm or HOSTED_MODEL)
            )
            answer = answer_from_matches(corpus, completion, args.question, matches)
        print(answer.text)
        if answer.citations:
            print("\nReferences")
            print(render_bibliography(answer.citations))
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
                pending = corpus.pending_chunking(fingerprint)
                new_chunks = (
                    extract_pdf_chunks(
                        pending,
                        chunk_size=args.chunk_size,
                        overlap=args.overlap,
                    )
                    if pending
                    else []
                )
                corpus.record_chunks(new_chunks)
                corpus.mark_chunked(pending, fingerprint)
                chunks = corpus.all_chunks()
            save_chunks(chunks, args.output)
            print(
                f"Chunked {len(pending)} PDFs into {len(new_chunks)} new passages; "
                f"corpus holds {len(chunks)}. Exported to {args.output}"
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
    licence = chunk.license_url or "no licence reported"
    body = chunk.text if full else match.snippet
    return f"{header}\n   {chunk.source_url}  ({licence})\n   {body}\n"


def _merge_downloads(
    existing_downloads: list[DownloadedPaper], new_downloads: list[DownloadedPaper]
) -> list[DownloadedPaper]:
    """Keep legacy snapshot downloads cumulative while callers migrate to SQLite."""
    by_id = {download.arxiv_id: download for download in existing_downloads}
    by_id.update({download.arxiv_id: download for download in new_downloads})
    return list(by_id.values())
