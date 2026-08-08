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
    assert "sortBy=lastUpdatedDate" in requested_urls[0]


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


OAI_RECORD = b"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <GetRecord><record><metadata>
    <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
      <id>2608.05440</id>
      <license>http://arxiv.org/licenses/nonexclusive-distrib/1.0/</license>
    </arXiv>
  </metadata></record></GetRecord>
</OAI-PMH>"""

OAI_NO_LICENSE = b"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <GetRecord><record><metadata>
    <arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>math/0309136</id></arXiv>
  </metadata></record></GetRecord>
</OAI-PMH>"""

OAI_ERROR = b"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <error code="idDoesNotExist">No matching identifier</error>
</OAI-PMH>"""


def test_license_is_read_from_the_oai_record() -> None:
    from verbatim.arxiv import parse_license

    assert parse_license(OAI_RECORD) == "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"


def test_a_record_without_a_license_returns_none_rather_than_raising() -> None:
    from verbatim.arxiv import parse_license

    assert parse_license(OAI_NO_LICENSE) is None


def test_an_oai_error_is_raised_rather_than_read_as_no_license() -> None:
    from verbatim.arxiv import parse_license

    with pytest.raises(ArxivAPIError, match="idDoesNotExist"):
        parse_license(OAI_ERROR)


def test_license_lookup_uses_the_base_identifier() -> None:
    from verbatim.arxiv import fetch_license

    requested: list[str] = []

    def fake_get(url: str) -> bytes:
        requested.append(url)
        return OAI_RECORD

    fetch_license("math/0309136", http_get=fake_get)

    assert "identifier=oai%3AarXiv.org%3Amath%2F0309136" in requested[0]
    assert "metadataPrefix=arXiv" in requested[0]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://creativecommons.org/licenses/by/4.0/", "open"),
        ("http://creativecommons.org/licenses/by-sa/4.0/", "open"),
        ("http://creativecommons.org/publicdomain/zero/1.0/", "open"),
        ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/", "arxiv-nonexclusive"),
        (None, "unknown"),
        ("", "unknown"),
        ("https://example.com/custom", "other"),
    ],
)
def test_license_class_separates_reusable_from_merely_present(url, expected) -> None:
    from verbatim.arxiv import license_class

    assert license_class(url) == expected
