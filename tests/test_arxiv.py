from pathlib import Path

import pytest

from verbatim.arxiv import fetch_metadata, parse_feed


FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_feed.xml"


def test_parse_feed_preserves_provenance() -> None:
    papers = parse_feed(FIXTURE.read_bytes())

    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "2401.12345v2"
    assert paper.title == "A Test Paper"
    assert paper.summary == "A test abstract."
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.categories == ["physics.gen-ph", "quant-ph"]
    assert paper.pdf_url == "http://arxiv.org/pdf/2401.12345v2"
    assert paper.license_url == "http://creativecommons.org/licenses/by/4.0/"


def test_fetch_metadata_builds_category_query() -> None:
    requested_urls: list[str] = []

    def fake_get(url: str) -> bytes:
        requested_urls.append(url)
        return FIXTURE.read_bytes()

    papers = fetch_metadata("physics.gen-ph", 3, http_get=fake_get)

    assert len(papers) == 1
    assert "search_query=cat%3Aphysics.gen-ph" in requested_urls[0]
    assert "max_results=3" in requested_urls[0]


def test_fetch_metadata_rejects_zero_results() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        fetch_metadata("physics.gen-ph", 0, http_get=lambda _: b"")

