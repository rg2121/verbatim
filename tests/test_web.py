"""The HTTP front end.

Retrieval itself is tested elsewhere; what matters here is that a result
carries everything needed to check it, and that the page and the API are
usable without a network.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from helpers import chunked_corpus
from verbatim.web import PAGE, create_app, match_to_dict


class StreamingScript:
    """A scripted answering model with the streaming surface Ollama has."""

    def __init__(self, fragments: list[str]) -> None:
        self.fragments = fragments

    def stream(self, system, user):
        yield from self.fragments

    def judge(self, system, user):
        return "superconducting qubit decoherence"


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


class StubEncoder:
    """Enough embedder for the ask flow, with no ONNX weights to load."""

    def __init__(self) -> None:
        from verbatim.corpus import EmbeddingModel

        self.model = EmbeddingModel(
            model_id="stub", dimensions=3, normalized=True, revision="v0"
        )

    def embed_passages(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def ask_client(tmp_path: Path, fragments: list[str]) -> tuple[TestClient, str]:
    with chunked_corpus(
        tmp_path, {"2608.11111v1": ["Superconducting qubits decohere quickly."]}
    ) as store:
        chunk_id = store.all_chunks()[0].chunk_id
    app = create_app(
        tmp_path / "corpus.db",
        completion=StreamingScript(fragments),
        encoder=StubEncoder(),
    )
    return TestClient(app), chunk_id


def client(tmp_path: Path) -> TestClient:
    with chunked_corpus(
        tmp_path, {"2608.11111v1": ["Superconducting qubits decohere quickly."]}
    ):
        pass
    return TestClient(create_app(tmp_path / "corpus.db"))


def test_ask_streams_a_draft_and_finishes_with_a_verified_answer(tmp_path: Path) -> None:
    fragments = ["ANSWER: They decohere quickly [1].\n", ""]
    client_app, chunk_id = ask_client(tmp_path, fragments)
    # The script cites the passage that really was retrieved, quoting exactly.
    fragments[1] = f"CITE [1] {chunk_id} :: Superconducting qubits decohere quickly."

    events = sse_events(client_app.get("/api/ask", params={"q": "qubits"}).text)

    names = [name for name, _ in events]
    assert names[0] == "passages" and names.count("delta") == 2 and names[-1] == "done"
    answer = next(payload for name, payload in events if name == "answer")
    assert answer["text"] == "They decohere quickly [1]."
    assert [c["arxiv_id"] for c in answer["citations"]] == ["2608.11111v1"]
    assert answer["rejected"] == []
    subject = next(payload for name, payload in events if name == "subject")
    assert subject["subject"] == "superconducting qubit decoherence"
    assert "warn" in subject and "closeness" in subject


def test_ask_rejects_a_quote_the_passage_does_not_contain(tmp_path: Path) -> None:
    fragments = ["ANSWER: Made up [1].\n", ""]
    client_app, chunk_id = ask_client(tmp_path, fragments)
    fragments[1] = f"CITE [1] {chunk_id} :: Qubits never decohere at all."

    events = sse_events(client_app.get("/api/ask", params={"q": "qubits"}).text)

    answer = next(payload for name, payload in events if name == "answer")
    assert answer["citations"] == []
    assert len(answer["rejected"]) == 1
    assert "[unverified]" in answer["text"]


def test_ask_declines_when_nothing_is_retrieved(tmp_path: Path) -> None:
    client_app, _ = ask_client(tmp_path, ["ANSWER: should never be asked\n"])

    events = sse_events(
        client_app.get("/api/ask", params={"q": "photosynthesis chlorophyll"}).text
    )

    names = [name for name, _ in events]
    assert "delta" not in names  # no model call for an empty retrieval
    answer = next(payload for name, payload in events if name == "answer")
    assert answer["declined"] is True


def test_ask_reports_a_model_that_ignored_the_format(tmp_path: Path) -> None:
    client_app, _ = ask_client(tmp_path, ["I refuse to follow formats."])

    events = sse_events(client_app.get("/api/ask", params={"q": "qubits"}).text)

    failed = next(payload for name, payload in events if name == "failed")
    assert "format" in failed["message"]
    assert [name for name, _ in events][-1] == "done"


def test_a_search_result_can_be_traced_back_to_its_paper(tmp_path: Path) -> None:
    response = client(tmp_path).get("/api/search", params={"q": "qubits", "mode": "keyword"})

    assert response.status_code == 200
    (hit,) = response.json()["results"]
    assert hit["arxiv_id"] == "2608.11111v1"
    assert hit["locator"] == "p. 1"
    assert hit["source_url"].endswith("2608.11111v1")
    assert "license_url" in hit and "chunk_id" in hit


def test_an_unknown_paper_is_a_404_rather_than_an_empty_answer(tmp_path: Path) -> None:
    response = client(tmp_path).get("/api/papers/9999.99999v1")

    assert response.status_code == 404
    assert "9999.99999v1" in response.json()["detail"]


def test_a_known_paper_is_returned_whole(tmp_path: Path) -> None:
    response = client(tmp_path).get("/api/papers/2608.11111v1")

    assert response.status_code == 200
    assert response.json()["arxiv_id"] == "2608.11111v1"


def test_an_empty_query_is_refused_rather_than_scanning_everything(tmp_path: Path) -> None:
    assert client(tmp_path).get("/api/search", params={"q": ""}).status_code == 422


def test_the_limit_is_capped(tmp_path: Path) -> None:
    assert client(tmp_path).get(
        "/api/search", params={"q": "qubits", "limit": 5000}
    ).status_code == 422


def test_an_unknown_mode_is_refused(tmp_path: Path) -> None:
    assert client(tmp_path).get(
        "/api/search", params={"q": "qubits", "mode": "magic"}
    ).status_code == 422


def test_stats_are_served_for_the_page_footer(tmp_path: Path) -> None:
    counts = client(tmp_path).get("/api/stats").json()

    assert counts["papers"] == 1 and counts["chunks"] >= 1


def test_the_page_is_served_and_reaches_no_external_host(tmp_path: Path) -> None:
    """The pipeline runs offline once the corpus is built; the page must too."""
    response = client(tmp_path).get("/")

    assert response.status_code == 200
    assert "verbatim" in response.text
    for scheme in ("https://", "http://"):
        assert scheme not in PAGE.split("<script>")[0].replace(
            'xmlns="http://www.w3.org/1999/xhtml"', ""
        )
