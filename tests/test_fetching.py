from urllib.error import HTTPError, URLError

import pytest

import verbatim.fetching as fetching
from verbatim.fetching import fetch


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self, size: int | None = None) -> bytes:
        return self.payload if size is None else self.payload[:size]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def install(monkeypatch: pytest.MonkeyPatch, responses: list[object]) -> dict[str, int]:
    """Serve each entry in turn; an exception instance is raised instead."""
    calls = {"n": 0}

    def fake_urlopen(request, timeout):  # noqa: ARG001
        index = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        outcome = responses[index]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(fetching, "urlopen", fake_urlopen)
    return calls


def test_a_server_error_is_retried_until_it_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """arXiv returns 503 under load; a multi-hour ingest cannot abort on that."""
    error = HTTPError("https://x", 503, "Service Unavailable", {}, None)
    calls = install(monkeypatch, [error, error, FakeResponse(b"%PDF ok")])

    payload = fetch("https://x", retries=4, backoff=0, sleep=lambda _s: None)

    assert payload == b"%PDF ok"
    assert calls["n"] == 3


def test_a_timeout_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install(monkeypatch, [TimeoutError("read timed out"), FakeResponse(b"ok")])

    assert fetch("https://x", retries=2, backoff=0, sleep=lambda _s: None) == b"ok"
    assert calls["n"] == 2


def test_a_client_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A withdrawn paper returns 404 however many times it is asked for."""
    calls = install(monkeypatch, [HTTPError("https://x", 404, "Not Found", {}, None)])

    with pytest.raises(HTTPError):
        fetch("https://x", retries=4, backoff=0, sleep=lambda _s: None)

    assert calls["n"] == 1


def test_exhausted_retries_report_the_last_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install(monkeypatch, [URLError("connection refused")])

    with pytest.raises(RuntimeError, match="did not respond after 3 attempts"):
        fetch("https://x", retries=2, backoff=0, sleep=lambda _s: None)

    assert calls["n"] == 3


def test_backoff_grows_between_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, [HTTPError("https://x", 503, "busy", {}, None)])
    pauses: list[float] = []

    with pytest.raises(RuntimeError):
        fetch("https://x", retries=3, backoff=5, sleep=pauses.append)

    assert pauses == [5, 10, 20]  # doubling, and none before the first attempt


def test_an_oversized_response_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """One pathological document must not exhaust memory during a bulk ingest."""
    install(monkeypatch, [FakeResponse(b"x" * 5_000)])

    with pytest.raises(ValueError, match="exceeds the 1,000-byte limit"):
        fetch("https://x", max_bytes=1_000, retries=0, sleep=lambda _s: None)


def test_a_response_within_the_limit_is_returned_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, [FakeResponse(b"x" * 900)])

    assert len(fetch("https://x", max_bytes=1_000, retries=0, sleep=lambda _s: None)) == 900
