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
# arXiv HTML renderings, preferred over PDFs where they exist.
HTML_DIR = _path("VERBATIM_HTML_DIR", "data/html")
CHUNKS_PATH = _path("VERBATIM_CHUNKS_PATH", "data/chunks.jsonl")
PAPERS_SNAPSHOT_PATH = _path("VERBATIM_PAPERS_SNAPSHOT_PATH", "data/papers.json")
# The retrieval answer key. Committed, unlike the corpus it scores.
EVAL_PATH = _path("VERBATIM_EVAL_PATH", "evals/retrieval.jsonl")
# Under data/, which git ignores: TRAINING-POLICY.md keeps the dataset private.
DISTILL_PATH = _path("VERBATIM_DISTILL_PATH", "data/distill/dataset.jsonl")

# Chunking. Changing either forces affected PDFs to be re-chunked, because both
# are part of the chunker fingerprint.
CHUNK_SIZE = _integer("VERBATIM_CHUNK_SIZE", 1_200)
CHUNK_OVERLAP = _integer("VERBATIM_CHUNK_OVERLAP", 200)
# PDFs held in memory at once, at about 11 MB each.
CHUNK_BATCH_SIZE = _integer("VERBATIM_CHUNK_BATCH_SIZE", 100)
# Seconds before layout extraction of one PDF is abandoned and the paper skipped.
EXTRACTION_TIMEOUT = float(_text("VERBATIM_EXTRACTION_TIMEOUT", "180"))

# Retrieval.
EMBEDDING_MODEL = _text("VERBATIM_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
SEARCH_LIMIT = _integer("VERBATIM_SEARCH_LIMIT", 10)
ASK_PASSAGES = _integer("VERBATIM_ASK_PASSAGES", 8)
ASK_MAX_PER_PAPER = _integer("VERBATIM_ASK_MAX_PER_PAPER", 3)
# Warn when the passages' subject is less similar to the question than this.
# A warning only: answerable and unanswerable questions overlap.
SUBJECT_WARN_BELOW = float(_text("VERBATIM_SUBJECT_WARN_BELOW", "0.65"))

# Embed each passage behind its paper title and section heading. Off by
# default; measured, the difference was inconclusive.
EMBED_WITH_CONTEXT = _text("VERBATIM_EMBED_WITH_CONTEXT", "0") not in {"0", "false", "False", ""}

# How the answering model is asked for structure: "lines" (ANSWER/CITE lines)
# or "tools" (Ollama tool calling, measured no better).
ANSWER_PROTOCOL = _text("VERBATIM_ANSWER_PROTOCOL", "lines")

# Stage events go to stderr so stdout stays parseable. "json" for a shipper,
# "text" for a person.
LOG_FORMAT = _text("VERBATIM_LOG_FORMAT", "text")
LOG_LEVEL = _text("VERBATIM_LOG_LEVEL", "info")

# Generation goes through Ollama only. A `-cloud` model sends passages to
# Ollama's servers, so it is used only when named.
OLLAMA_HOST = _text("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
OLLAMA_MODEL = _text("OLLAMA_MODEL", "") or None  # None means "first pulled local model"
OLLAMA_TIMEOUT = float(_text("OLLAMA_TIMEOUT", "600"))

# arXiv's API is slow at times and returns 503 under load, so requests retry.
HTTP_TIMEOUT = float(_text("VERBATIM_HTTP_TIMEOUT", "90"))
HTTP_RETRIES = _integer("VERBATIM_HTTP_RETRIES", 4)
HTTP_BACKOFF = float(_text("VERBATIM_HTTP_BACKOFF", "5"))
# Limits for a single PDF download.
PDF_TIMEOUT = float(_text("VERBATIM_PDF_TIMEOUT", "120"))
PDF_MAX_BYTES = _integer("VERBATIM_PDF_MAX_BYTES", 200_000_000)
# Seconds between PDF requests, as arXiv asks.
DOWNLOAD_DELAY = float(_text("VERBATIM_DOWNLOAD_DELAY", "3"))

# Sync paging. arXiv caps a single request, so a real corpus has to be paged.
SYNC_LIMIT = _integer("VERBATIM_SYNC_LIMIT", 1_000)
SYNC_PAGE_SIZE = _integer("VERBATIM_SYNC_PAGE_SIZE", 100)
SYNC_DELAY = float(_text("VERBATIM_SYNC_DELAY", "3"))

# arXiv politeness. The OAI-PMH interface asks callers to pace themselves.
LICENSE_LOOKUP_DELAY = float(_text("VERBATIM_LICENSE_LOOKUP_DELAY", "3"))
