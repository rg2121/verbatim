"""The HTTP front end.

Retrieval itself is tested elsewhere; what matters here is that a result
carries everything needed to check it, and that the page and the API are
usable without a network.
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from helpers import chunked_corpus
from verbatim.web import PAGE, create_app, match_to_dict


def client(tmp_path: Path) -> TestClient:
    with chunked_corpus(
        tmp_path, {"2608.11111v1": ["Superconducting qubits decohere quickly."]}
    ):
        pass
    return TestClient(create_app(tmp_path / "corpus.db"))


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
