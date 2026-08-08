"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from verbatim.arxiv import fetch_metadata
from verbatim.chunking import chunker_fingerprint, extract_pdf_chunks
from verbatim.corpus import CorpusStore
from verbatim.downloads import DownloadedPaper, download_papers
from verbatim.storage import (
    load_download_manifest,
    load_papers,
    save_chunks,
    save_download_manifest,
    save_papers,
)


def main() -> None:
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
            new_downloads = download_papers(
                papers,
                args.output_dir,
                args.limit,
                require_license=args.require_license,
                existing_downloads=existing_downloads,
            )
            downloads = _merge_downloads(existing_downloads, new_downloads)
        else:
            with CorpusStore(args.state) as corpus:
                papers = corpus.pending_downloads(args.limit, require_license=args.require_license)
                existing_downloads = corpus.all_downloads()
                new_downloads = (
                    download_papers(
                        papers,
                        args.output_dir,
                        len(papers),
                        require_license=args.require_license,
                        existing_downloads=existing_downloads,
                    )
                    if papers
                    else []
                )
                corpus.record_downloads(new_downloads)
                downloads = corpus.all_downloads()
        save_download_manifest(downloads, manifest_path)
        print(f"Downloaded {len(new_downloads)} new PDFs; corpus manifest has {len(downloads)} PDFs: {manifest_path}")

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


def _merge_downloads(
    existing_downloads: list[DownloadedPaper], new_downloads: list[DownloadedPaper]
) -> list[DownloadedPaper]:
    """Keep legacy snapshot downloads cumulative while callers migrate to SQLite."""
    by_id = {download.arxiv_id: download for download in existing_downloads}
    by_id.update({download.arxiv_id: download for download in new_downloads})
    return list(by_id.values())
