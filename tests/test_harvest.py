"""Harvesting whole arXiv subjects since a start date."""

from __future__ import annotations

from pathlib import Path

import pytest

from verbatim.arxiv import ArxivAPIError, Paper
from verbatim.corpus import CorpusStore
from verbatim.harvest import (
    SUBJECTS,
    OaiRecord,
    harvest_subject,
    parse_oai_page,
    submission_month,
)

OPEN = "http://creativecommons.org/licenses/by/4.0/"
DEFAULT = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"


def oai_xml(records: list[tuple[str, str, str | None]], *, token: str = "",
            deleted: tuple[str, ...] = ()) -> bytes:
    """A ListRecords response shaped like arXiv's, from (id, datestamp, licence)."""
    entries = []
    for arxiv_id, datestamp, licence in records:
        licence_line = f"<license>{licence}</license>" if licence else ""
        entries.append(
            "<record><header>"
            f"<identifier>oai:arXiv.org:{arxiv_id}</identifier>"
            f"<datestamp>{datestamp}</datestamp><setSpec>cs:cs:LG</setSpec>"
            "</header><metadata>"
            '<arXiv xmlns="http://arxiv.org/OAI/arXiv/">'
            f"<id>{arxiv_id}</id><created>{datestamp}</created>"
            f"<title>Paper {arxiv_id}</title><categories>cs.LG</categories>{licence_line}"
            "<abstract>An abstract.</abstract></arXiv></metadata></record>"
        )
    for arxiv_id in deleted:
        entries.append(
            '<record><header status="deleted">'
            f"<identifier>oai:arXiv.org:{arxiv_id}</identifier>"
            "<datestamp>2026-11-02</datestamp></header></record>"
        )
    resumption = (
        f"<resumptionToken expirationDate='2026-11-07T00:00:00Z'>{token}</resumptionToken>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
        "<responseDate>2026-11-06T16:08:25Z</responseDate>"
        '<request verb="ListRecords">http://oaipmh.arxiv.org/oai</request>'
        f"<ListRecords>{''.join(entries)}{resumption}</ListRecords></OAI-PMH>"
    ).encode()


def oai_error(code: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
        f'<responseDate>2026-11-06T16:08:25Z</responseDate><error code="{code}">'
        "The combination of the values of the from, until, set and metadataPrefix "
        "arguments results in an empty list.</error></OAI-PMH>"
    ).encode()


def api_paper(base_id: str, published: str, *, version: int = 1) -> Paper:
    return Paper(
        arxiv_id=f"{base_id}v{version}",
        title=f"Paper {base_id}",
        summary="An abstract.",
        authors=["Ada Lovelace"],
        categories=["cs.LG"],
        published=f"{published}T10:00:00Z",
        updated=f"{published}T10:00:00Z",
        source_url=f"https://arxiv.org/abs/{base_id}v{version}",
        pdf_url=f"https://arxiv.org/pdf/{base_id}v{version}",
        license_url=None,
    )


class FakeArxiv:
    """OAI pages served in order, and the query API answering by identifier."""

    def __init__(self, pages: list[bytes], papers: dict[str, Paper]) -> None:
        self.pages = list(pages)
        self.papers = papers
        self.oai_calls: list[tuple[str, str | None, str | None]] = []
        self.looked_up: list[str] = []

    def fetch_page(self, subject: str, *, since: str | None = None, token: str | None = None):
        self.oai_calls.append((subject, since, token))
        return parse_oai_page(self.pages.pop(0))

    def fetch_by_id(self, base_ids: list[str]) -> list[Paper]:
        self.looked_up.extend(base_ids)
        # arXiv leaves out identifiers it does not know.
        return [self.papers[i] for i in base_ids if i in self.papers]


def harvest(corpus, fake, subject="cs", **options):
    return harvest_subject(
        corpus, subject, fetch_page=fake.fetch_page, fetch_by_id=fake.fetch_by_id,
        sleep=lambda _: None, **options,
    )


def test_a_page_yields_its_records_licences_and_token() -> None:
    page = parse_oai_page(oai_xml(
        [("2611.00001", "2026-11-04", OPEN), ("2611.00002", "2026-11-04", None)],
        token="verb%3DListRecords%26skip%3D1300", deleted=("2611.00003",),
    ))

    assert page.records == [
        OaiRecord("2611.00001", "2026-11-04", OPEN),
        OaiRecord("2611.00002", "2026-11-04", None),
    ]
    assert page.resumption_token == "verb%3DListRecords%26skip%3D1300"


def test_the_last_page_has_no_token() -> None:
    assert parse_oai_page(oai_xml([("2611.00001", "2026-11-04", OPEN)])).resumption_token is None


def test_an_empty_window_is_not_an_error() -> None:
    page = parse_oai_page(oai_error("noRecordsMatch"))

    assert page.records == [] and page.resumption_token is None


def test_any_other_oai_error_is_raised() -> None:
    with pytest.raises(ArxivAPIError, match="badResumptionToken"):
        parse_oai_page(oai_error("badResumptionToken"))


def test_an_identifier_says_the_month_a_paper_was_first_submitted() -> None:
    assert submission_month("2611.08325") == (2026, 11)
    assert submission_month("0704.0001") == (2007, 4)
    assert submission_month("hep-th/9901001") == (1999, 1)
    assert submission_month("math/0601001") == (2006, 1)


def test_the_first_harvest_of_a_subject_needs_a_start_date(tmp_path: Path) -> None:
    fake = FakeArxiv([], {})
    with CorpusStore(tmp_path / "corpus.db") as corpus, pytest.raises(ValueError, match="--since"):
        harvest(corpus, fake)


def test_only_known_subjects_are_harvested(tmp_path: Path) -> None:
    assert {"cs", "math", "physics"} <= set(SUBJECTS)
    fake = FakeArxiv([], {})
    with CorpusStore(tmp_path / "corpus.db") as corpus, pytest.raises(ValueError, match="subject"):
        harvest(corpus, fake, subject="computers", since="2026-10-23")


def test_a_start_date_must_be_a_date(tmp_path: Path) -> None:
    fake = FakeArxiv([], {})
    with CorpusStore(tmp_path / "corpus.db") as corpus, pytest.raises(ValueError, match="YYYY-MM-DD"):
        harvest(corpus, fake, since="23/10/2026")


def test_papers_first_submitted_since_the_start_are_kept_and_older_ones_left_out(
    tmp_path: Path,
) -> None:
    """A revision lists a 2022 paper again; it is not new, so it stays out."""
    fake = FakeArxiv(
        [oai_xml([
            ("2611.00001", "2026-11-04", OPEN),       # new this month
            ("2610.00050", "2026-11-04", DEFAULT),    # October, but before the start
            ("2202.10527", "2026-11-04", DEFAULT),    # a 2022 paper, revised
        ])],
        {
            "2611.00001": api_paper("2611.00001", "2026-11-03"),
            "2610.00050": api_paper("2610.00050", "2026-10-02", version=2),
            "2202.10527": api_paper("2202.10527", "2022-02-21", version=3),
        },
    )
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        report = harvest(corpus, fake, since="2026-10-23")
        held = sorted(corpus.papers_by_id(["2611.00001v1", "2610.00050v2", "2202.10527v3"]))

    assert held == ["2611.00001v1"]
    assert report.kept == 1 and report.records == 3
    # The id alone shows the 2022 paper is old; it is not even looked up.
    assert "2202.10527" not in fake.looked_up


def test_harvested_papers_arrive_with_their_licences(tmp_path: Path) -> None:
    fake = FakeArxiv(
        [oai_xml([("2611.00001", "2026-11-04", OPEN), ("2611.00002", "2026-11-04", None)])],
        {"2611.00001": api_paper("2611.00001", "2026-11-03"),
         "2611.00002": api_paper("2611.00002", "2026-11-03")},
    )
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        report = harvest(corpus, fake, since="2026-10-23")
        stats = corpus.pipeline_stats("fingerprint", "model")
        licence = corpus.connection.execute(
            "SELECT license_url, license_class FROM papers WHERE arxiv_id = '2611.00001v1'"
        ).fetchone()

    # Nothing is left for the one-by-one licence lookup.
    assert stats["licences_pending"] == 0
    assert tuple(licence) == (OPEN, "open")
    assert report.licences == 1


def test_a_new_version_of_a_held_paper_is_kept_however_old_the_paper(tmp_path: Path) -> None:
    from helpers import paper

    fake = FakeArxiv(
        [oai_xml([("2202.10527", "2026-11-04", DEFAULT)])],
        {"2202.10527": api_paper("2202.10527", "2022-02-21", version=3)},
    )
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        corpus.upsert_papers([paper("2202.10527v2", updated="2022-06-01T10:00:00Z")])
        corpus.connection.execute(
            "UPDATE papers SET last_seen_at = '2026-08-08T10:00:00Z'"
        )
        report = harvest(corpus, fake, since="2026-10-23")
        latest = corpus.connection.execute(
            "SELECT arxiv_id FROM papers WHERE base_arxiv_id = '2202.10527' AND is_latest = 1"
        ).fetchone()[0]

    assert latest == "2202.10527v3"
    assert report.kept == 1


def test_a_held_paper_seen_after_its_record_changed_is_not_looked_up_again(
    tmp_path: Path,
) -> None:
    """An interrupted harvest re-reads its window; what it already stored costs nothing."""
    from helpers import paper

    fake = FakeArxiv([oai_xml([("2611.00001", "2026-11-04", OPEN)])], {})
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        corpus.upsert_papers([paper("2611.00001v1")])
        corpus.connection.execute("UPDATE papers SET last_seen_at = '2026-11-05T10:00:00Z'")
        harvest(corpus, fake, since="2026-10-23")

    assert fake.looked_up == []


def test_a_harvest_follows_resumption_tokens_and_paces_itself(tmp_path: Path) -> None:
    pauses: list[float] = []
    fake = FakeArxiv(
        [
            oai_xml([("2611.00001", "2026-11-03", OPEN)], token="page-two"),
            oai_xml([("2611.00002", "2026-11-04", OPEN)]),
        ],
        {"2611.00001": api_paper("2611.00001", "2026-11-02"),
         "2611.00002": api_paper("2611.00002", "2026-11-03")},
    )
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        report = harvest_subject(
            corpus, "cs", since="2026-10-23", fetch_page=fake.fetch_page,
            fetch_by_id=fake.fetch_by_id, delay=3.0, sleep=pauses.append,
        )

    assert [call[2] for call in fake.oai_calls] == [None, "page-two"]
    assert report.kept == 2
    # Two OAI pages and two lookups: a pause before every request but the first.
    assert pauses == [3.0, 3.0, 3.0]


def test_the_next_harvest_starts_where_the_last_one_finished(tmp_path: Path) -> None:
    first = FakeArxiv(
        [oai_xml([("2611.00001", "2026-11-04", OPEN)])],
        {"2611.00001": api_paper("2611.00001", "2026-11-03")},
    )
    second = FakeArxiv(
        [oai_xml([("2611.00009", "2026-11-06", OPEN), ("2610.00050", "2026-11-06", OPEN)])],
        {"2611.00009": api_paper("2611.00009", "2026-11-05"),
         "2610.00050": api_paper("2610.00050", "2026-10-02", version=2)},
    )
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        harvest(corpus, first, since="2026-10-23")
        report = harvest(corpus, second)
        state = corpus.subject_harvest("cs")

    # Re-read from the last day harvested, which may have gained records since.
    assert second.oai_calls[0][1] == "2026-11-04"
    # The first start date still decides what counts as new.
    assert report.kept == 1
    assert state.since == "2026-10-23" and state.harvested_through == "2026-11-06"


def test_a_failed_harvest_keeps_what_it_stored_and_does_not_move_on(tmp_path: Path) -> None:
    pages = [oai_xml([("2611.00001", "2026-11-04", OPEN)], token="page-two")]
    fake = FakeArxiv(pages, {"2611.00001": api_paper("2611.00001", "2026-11-03")})

    def flaky(subject, *, since=None, token=None):
        if token:
            raise RuntimeError("oaipmh.arxiv.org did not respond after 5 attempts")
        return fake.fetch_page(subject, since=since, token=token)

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        report = harvest_subject(
            corpus, "cs", since="2026-10-23", fetch_page=flaky,
            fetch_by_id=fake.fetch_by_id, sleep=lambda _: None,
        )
        state = corpus.subject_harvest("cs")
        held = corpus.papers_by_id(["2611.00001v1"])

    assert report.failed and "did not respond" in report.stopped_because
    assert list(held) == ["2611.00001v1"]
    # Nothing was finished, so the next harvest reads the whole window again.
    assert state is not None and state.harvested_through is None


def test_identifiers_arxiv_does_not_return_are_counted(tmp_path: Path) -> None:
    fake = FakeArxiv([oai_xml([("2611.00001", "2026-11-04", OPEN)])], {})
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        report = harvest(corpus, fake, since="2026-10-23")

    assert report.missing == 1 and report.kept == 0


def test_lookups_go_in_batches_arxiv_accepts(tmp_path: Path) -> None:
    from verbatim.arxiv import MAX_IDS_PER_REQUEST

    records = [(f"2611.{n:05d}", "2026-11-04", OPEN) for n in range(1, 251)]
    fake = FakeArxiv([oai_xml(records)], {})
    sizes: list[int] = []

    def by_id(base_ids):
        sizes.append(len(base_ids))
        return fake.fetch_by_id(base_ids)

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        harvest_subject(
            corpus, "cs", since="2026-10-23", fetch_page=fake.fetch_page,
            fetch_by_id=by_id, sleep=lambda _: None,
        )

    assert max(sizes) <= MAX_IDS_PER_REQUEST and sum(sizes) == 250


def test_opening_the_sample_still_changes_nothing_after_a_harvest_check(tmp_path: Path) -> None:
    import shutil

    from verbatim.downloads import file_sha256

    copy = tmp_path / "sample.db"
    shutil.copy(Path(__file__).parent.parent / "sample" / "corpus.db", copy)
    before = file_sha256(copy)
    with CorpusStore(copy) as corpus:
        assert corpus.subject_harvests() == []
        assert corpus.subject_harvest("cs") is None

    assert file_sha256(copy) == before
