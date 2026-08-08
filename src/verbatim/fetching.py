"""Fetching URLs from a service that fails transiently.

arXiv is slow and intermittently unavailable under load: timing the query API,
identical requests returned in 2s, 18s, 36s and 61s, and one came back 503. A
single attempt is therefore not a reasonable way to fetch several thousand
documents, so both the metadata client and the PDF downloader share this.
"""

from __future__ import annotations

from collections.abc import Callable
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from verbatim import config

USER_AGENT = "verbatim/0.1 (personal research project)"


def fetch(
    url: str,
    *,
    timeout: float | None = None,
    retries: int | None = None,
    backoff: float | None = None,
    max_bytes: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Fetch one URL, retrying only the failures that retrying can fix.

    Server errors, timeouts and connection failures are transient and worth
    another attempt. A 4xx is not: a withdrawn paper returns 404 however many
    times it is asked for, and retrying only slows the run down.

    `max_bytes` bounds the response so one pathological document cannot
    exhaust memory during a bulk ingest.
    """
    attempts = config.HTTP_RETRIES if retries is None else retries
    pause = config.HTTP_BACKOFF if backoff is None else backoff
    deadline = config.HTTP_TIMEOUT if timeout is None else timeout
    request = Request(url, headers={"User-Agent": USER_AGENT})

    last_error: Exception | None = None
    for attempt in range(attempts + 1):
        if attempt:
            sleep(pause * (2 ** (attempt - 1)))
        try:
            with urlopen(request, timeout=deadline) as response:  # noqa: S310
                return _read_bounded(response, max_bytes)
        except HTTPError as error:
            if error.code < 500:
                raise
            last_error = error
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
    raise RuntimeError(f"{url} did not respond after {attempts + 1} attempts: {last_error}")


def _read_bounded(response: object, max_bytes: int | None) -> bytes:
    if max_bytes is None:
        return response.read()  # type: ignore[attr-defined]
    payload = response.read(max_bytes + 1)  # type: ignore[attr-defined]
    if len(payload) > max_bytes:
        raise ValueError(f"Response exceeds the {max_bytes:,}-byte limit")
    return payload
