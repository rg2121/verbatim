from pathlib import Path

import pytest

from verbatim.arxiv import ArxivAPIError, fetch_metadata, parse_feed


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
    assert paper.pdf_url == "https://arxiv.org/pdf/2401.12345v2"
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


def test_parse_feed_preserves_legacy_arxiv_archive_name() -> None:
    xml = FIXTURE.read_text().replace("2401.12345v2", "math/0309136v1")

    paper = parse_feed(xml.encode())[0]

    assert paper.arxiv_id == "math/0309136v1"
    assert paper.source_url == "https://arxiv.org/abs/math/0309136v1"


def test_parse_feed_rejects_arxiv_api_error_entry() -> None:
    xml = FIXTURE.read_text().replace(
        "https://arxiv.org/abs/2401.12345v2",
        "http://arxiv.org/api/errors#incorrect_id_format_for_bogus",
    )

    with pytest.raises(ArxivAPIError, match="incorrect_id_format_for_bogus"):
        parse_feed(xml.encode())
