"""HTTP fetching with retries, shared by the arXiv metadata and PDF clients."""

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
    """Fetch one URL, retrying server errors, timeouts and dropped connections.

    A 4xx is raised at once. `max_bytes` bounds the response size.
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
