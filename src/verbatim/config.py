"""Every tunable in one place, read from the environment or a local .env file.

Real environment variables always win over the .env file, so a one-off run can
override the project defaults without editing anything:

    OLLAMA_MODEL=llama3.1 uv run verbatim ask "..."
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# override=False keeps an exported variable authoritative over the file.
load_dotenv(find_dotenv(usecwd=True), override=False)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_HOSTED_MODEL = "hosted-model"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def _text(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _path(name: str, default: str) -> Path:
    return Path(_text(name, default)).expanduser()


def _integer(name: str, default: int) -> int:
    raw = _text(name, str(default))
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error


# Where the corpus lives. The database is the source of truth; the rest is
# derived and can be rebuilt from it.
STATE_PATH = _path("VERBATIM_STATE_PATH", "data/corpus.db")
PDF_DIR = _path("VERBATIM_PDF_DIR", "data/pdfs")
CHUNKS_PATH = _path("VERBATIM_CHUNKS_PATH", "data/chunks.jsonl")
PAPERS_SNAPSHOT_PATH = _path("VERBATIM_PAPERS_SNAPSHOT_PATH", "data/papers.json")

# Chunking. Changing either forces affected PDFs to be re-chunked, because both
# are part of the chunker fingerprint.
CHUNK_SIZE = _integer("VERBATIM_CHUNK_SIZE", 1_200)
CHUNK_OVERLAP = _integer("VERBATIM_CHUNK_OVERLAP", 200)

# Retrieval.
EMBEDDING_MODEL = _text("VERBATIM_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
SEARCH_LIMIT = _integer("VERBATIM_SEARCH_LIMIT", 10)
ASK_PASSAGES = _integer("VERBATIM_ASK_PASSAGES", 8)
ASK_MAX_PER_PAPER = _integer("VERBATIM_ASK_MAX_PER_PAPER", 3)

# Answer generation. HOSTED_API_KEY is read by the hosted SDK itself and
# is deliberately not mirrored here, so the secret is never held in a module.
ANSWER_BACKEND = _text("VERBATIM_ANSWER_BACKEND", "ollama")
HOSTED_MODEL = _text("VERBATIM_HOSTED_MODEL", DEFAULT_HOSTED_MODEL)
OLLAMA_HOST = _text("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
OLLAMA_MODEL = _text("OLLAMA_MODEL", "") or None  # None means "first pulled model"
OLLAMA_TIMEOUT = float(_text("OLLAMA_TIMEOUT", "600"))

# arXiv's API is variable: identical queries have taken 2s and 61s, and it
# returns 503 under load, so a long ingest needs patience and retries.
HTTP_TIMEOUT = float(_text("VERBATIM_HTTP_TIMEOUT", "90"))
HTTP_RETRIES = _integer("VERBATIM_HTTP_RETRIES", 4)
HTTP_BACKOFF = float(_text("VERBATIM_HTTP_BACKOFF", "5"))
# PDFs are larger than metadata, and one pathological file should not be able
# to exhaust memory during a bulk ingest.
PDF_TIMEOUT = float(_text("VERBATIM_PDF_TIMEOUT", "120"))
PDF_MAX_BYTES = _integer("VERBATIM_PDF_MAX_BYTES", 200_000_000)
# Seconds between PDF requests. Thousands of downloads without a gap is
# exactly the hammering arXiv asks callers to avoid.
DOWNLOAD_DELAY = float(_text("VERBATIM_DOWNLOAD_DELAY", "3"))

# Sync paging. arXiv caps a single request, so a real corpus has to be paged.
SYNC_LIMIT = _integer("VERBATIM_SYNC_LIMIT", 1_000)
SYNC_PAGE_SIZE = _integer("VERBATIM_SYNC_PAGE_SIZE", 100)
SYNC_DELAY = float(_text("VERBATIM_SYNC_DELAY", "3"))

# arXiv politeness. The OAI-PMH interface asks callers to pace themselves.
LICENSE_LOOKUP_DELAY = float(_text("VERBATIM_LICENSE_LOOKUP_DELAY", "3"))
