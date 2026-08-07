"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from verbatim.arxiv import fetch_metadata
from verbatim.chunking import extract_pdf_chunks
from verbatim.downloads import download_papers
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

    download_parser = subparsers.add_parser("download", help="Download a bounded set of PDFs from a metadata snapshot")
    download_parser.add_argument("--input", type=Path, default=Path("data/papers.json"))
    download_parser.add_argument("--limit", type=int, default=3, help="Maximum number of PDFs to download")
    download_parser.add_argument("--output-dir", type=Path, default=Path("data/pdfs"))
    download_parser.add_argument(
        "--require-license",
        action="store_true",
        help="Download only papers whose metadata explicitly reports a licence",
    )

    chunk_parser = subparsers.add_parser("chunk", help="Extract page-aware text chunks from downloaded PDFs")
    chunk_parser.add_argument("--manifest", type=Path, default=Path("data/pdfs/manifest.json"))
    chunk_parser.add_argument("--output", type=Path, default=Path("data/chunks.jsonl"))
    chunk_parser.add_argument("--chunk-size", type=int, default=1_200)
    chunk_parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    if args.command == "fetch":
        papers = fetch_metadata(args.category, args.max_results)
        save_papers(papers, args.output)
        print(f"Saved {len(papers)} papers to {args.output}")

    if args.command == "download":
        papers = load_papers(args.input)
        manifest_path = args.output_dir / "manifest.json"
        existing_downloads = load_download_manifest(manifest_path) if manifest_path.exists() else []
        downloads = download_papers(
            papers,
            args.output_dir,
            args.limit,
            require_license=args.require_license,
            existing_downloads=existing_downloads,
        )
        save_download_manifest(downloads, manifest_path)
        print(f"Downloaded or reused {len(downloads)} PDFs; manifest: {manifest_path}")

    if args.command == "chunk":
        downloads = load_download_manifest(args.manifest)
        chunks = extract_pdf_chunks(
            downloads,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        save_chunks(chunks, args.output)
        print(f"Saved {len(chunks)} page-aware chunks to {args.output}")
