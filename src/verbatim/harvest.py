"""Harvesting whole arXiv subjects (computer science, mathematics, physics) since a date.

arXiv's OAI-PMH interface lists every record that changed in a subject, day by
day, with its licence: about 1,300 records a request. A record does not say
when the paper was first submitted (revising a 2022 paper lists it again), so
metadata comes from the query API by identifier, whose `published` is the first
submission. A paper is kept when it was first submitted on or after the
subject's start date, or when it is a new version of a paper already held. An
identifier carries the month it was assigned, so most old papers are left out
without a lookup.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
import time
from urllib.parse import urlencode
from xml.etree import ElementTree

from verbatim import config
from verbatim.arxiv import (
    ARXIV_OAI_METADATA_NAMESPACE,
    ARXIV_OAI_URL,
    MAX_IDS_PER_REQUEST,
    OAI_NAMESPACE,
    ArxivAPIError,
    Paper,
    fetch_papers_by_id,
)
from verbatim.corpus import CorpusStore, split_arxiv_version
from verbatim.fetching import fetch

# arXiv's top-level OAI-PMH sets; "physics" holds all 13 physics archives.
SUBJECTS = ("cs", "math", "physics", "q-bio", "q-fin", "stat", "eess", "econ")
OAI_ID_PREFIX = "oai:arXiv.org:"
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Since 2007 "YYMM.NNNNN"; before, "archive/YYMMNNN".
NEW_STYLE_ID = re.compile(r"^(?P<yy>\d{2})(?P<mm>\d{2})\.\d{4,5}$")
OLD_STYLE_ID = re.compile(r"^[a-z-]+(?:\.[A-Za-z-]+)?/(?P<yy>\d{2})(?P<mm>\d{2})\d{3}$")


@dataclass(frozen=True)
class OaiRecord:
    base_arxiv_id: str
    # The day arXiv last changed the record: a new paper, version or correction.
    datestamp: str
    license_url: str | None


@dataclass(frozen=True)
class OaiPage:
    records: list[OaiRecord]
    resumption_token: str | None


@dataclass
class HarvestReport:
    subject: str
    since: str
    harvested_from: str
    records: int = 0
    looked_up: int = 0
    kept: int = 0
    added: int = 0
    refreshed: int = 0
    licences: int = 0
    # Identifiers the query API did not return, such as withdrawn papers.
    missing: int = 0
    requests: int = 0
    newest_datestamp: str | None = None
    failed: bool = False
    stopped_because: str = "read every record"


def submission_month(base_arxiv_id: str) -> tuple[int, int] | None:
    """The year and month an identifier was assigned, or None for an unknown form."""
    match = NEW_STYLE_ID.match(base_arxiv_id) or OLD_STYLE_ID.match(base_arxiv_id)
    if match is None:
        return None
    year = int(match["yy"])
    # Old-style identifiers began in 1991.
    year += 1900 if (match.re is OLD_STYLE_ID and year >= 91) else 2000
    return year, int(match["mm"])


def parse_oai_page(xml: bytes) -> OaiPage:
    """Read one ListRecords response; an empty window is no error."""
    root = ElementTree.fromstring(xml)
    namespaces = {"oai": OAI_NAMESPACE, "arxiv": ARXIV_OAI_METADATA_NAMESPACE}
    error = root.find("oai:error", namespaces)
    if error is not None:
        if error.get("code") == "noRecordsMatch":
            return OaiPage(records=[], resumption_token=None)
        raise ArxivAPIError(
            f"arXiv OAI-PMH error: {error.get('code')}: {(error.text or '').strip()}"
        )
    records: list[OaiRecord] = []
    for record in root.iterfind(".//oai:record", namespaces):
        header = record.find("oai:header", namespaces)
        if header is None or header.get("status") == "deleted":
            continue
        identifier = header.findtext("oai:identifier", "", namespaces).strip()
        datestamp = header.findtext("oai:datestamp", "", namespaces).strip()[:10]
        if not identifier.startswith(OAI_ID_PREFIX) or not DATE.match(datestamp):
            raise ArxivAPIError(f"unexpected OAI-PMH record: {identifier!r} on {datestamp!r}")
        licence = record.findtext(".//arxiv:license", "", namespaces).strip()
        records.append(
            OaiRecord(identifier[len(OAI_ID_PREFIX):], datestamp, licence or None)
        )
    token = root.findtext(".//oai:resumptionToken", "", namespaces).strip()
    return OaiPage(records=records, resumption_token=token or None)


def fetch_oai_page(
    subject: str,
    *,
    since: str | None = None,
    token: str | None = None,
    http_get: Callable[[str], bytes] | None = None,
) -> OaiPage:
    """One page of a subject's records changed since a date, or the page a token names."""
    if token is not None:
        query = urlencode({"verb": "ListRecords", "resumptionToken": token})
    else:
        query = urlencode(
            {"verb": "ListRecords", "set": subject, "metadataPrefix": "arXiv", "from": since}
        )
    get = http_get or fetch
    return parse_oai_page(get(f"{ARXIV_OAI_URL}?{query}"))


def harvest_subject(
    corpus: CorpusStore,
    subject: str,
    *,
    since: str | None = None,
    delay: float | None = None,
    fetch_page: Callable[..., OaiPage] | None = None,
    fetch_by_id: Callable[[list[str]], list[Paper]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[HarvestReport], None] | None = None,
) -> HarvestReport:
    """Add a subject's papers first submitted since its start date, and new versions of held ones.

    The first harvest needs `since`; later ones resume from the last day the
    previous one finished, keeping its start date. Passing `since` again reads
    from that date. A harvest that fails keeps what it stored but does not
    move on, so the next one reads the same window again.
    """
    if subject not in SUBJECTS:
        raise ValueError(f"unknown subject {subject!r}; arXiv's subjects are {', '.join(SUBJECTS)}")
    if since is not None and not DATE.match(since):
        raise ValueError(f"--since must be a date, YYYY-MM-DD, not {since!r}")
    state = corpus.subject_harvest(subject)
    start = since or (state.since if state else None)
    if start is None:
        raise ValueError(f"the first harvest of {subject} needs a start date: --since YYYY-MM-DD")
    read_from = since or (state.harvested_through if state and state.harvested_through else start)
    first_month = (int(start[:4]), int(start[5:7]))
    pause = config.SYNC_DELAY if delay is None else delay
    page_of = fetch_page or fetch_oai_page
    look_up = fetch_by_id or fetch_papers_by_id
    report = HarvestReport(subject=subject, since=start, harvested_from=read_from)

    def request(call: Callable[[], object]):
        if report.requests:
            sleep(pause)  # arXiv asks callers to pace themselves
        report.requests += 1
        return call()

    token: str | None = None
    try:
        while True:
            page = request(lambda: page_of(subject, since=read_from, token=token))
            report.records += len(page.records)
            if page.records:
                newest = max(record.datestamp for record in page.records)
                report.newest_datestamp = max(report.newest_datestamp or "", newest)
            held = corpus.last_seen_by_base([r.base_arxiv_id for r in page.records])
            wanted = [r for r in page.records if _worth_a_lookup(r, held, first_month)]
            for position in range(0, len(wanted), MAX_IDS_PER_REQUEST):
                batch = wanted[position : position + MAX_IDS_PER_REQUEST]
                papers = request(lambda: look_up([r.base_arxiv_id for r in batch]))
                report.looked_up += len(batch)
                report.missing += len(batch) - len(papers)
                _store(corpus, papers, batch, held, start, report)
            if on_progress is not None:
                on_progress(report)
            token = page.resumption_token
            if token is None:
                break
    except (OSError, RuntimeError, ValueError) as error:
        # Stop cleanly: what earlier pages stored is kept.
        report.failed = True
        report.stopped_because = f"failed after {report.requests} request(s): {error}"
    corpus.record_subject_harvest(
        subject, since=start,
        through=None if report.failed else (report.newest_datestamp or read_from),
        kept=report.kept,
    )
    return report


def _worth_a_lookup(
    record: OaiRecord, held: dict[str, str], first_month: tuple[int, int]
) -> bool:
    """Held papers whose record changed since a sync last saw them, and recent identifiers."""
    seen = held.get(record.base_arxiv_id)
    if seen is not None:
        return seen[:10] <= record.datestamp
    # An identifier is assigned no earlier than the paper is submitted.
    month = submission_month(record.base_arxiv_id)
    return month is None or month >= first_month


def _store(
    corpus: CorpusStore,
    papers: list[Paper],
    batch: list[OaiRecord],
    held: dict[str, str],
    start: str,
    report: HarvestReport,
) -> None:
    licences = {record.base_arxiv_id: record.license_url for record in batch}
    keep = [
        paper for paper in papers
        if paper.published[:10] >= start or split_arxiv_version(paper.arxiv_id)[0] in held
    ]
    if not keep:
        return
    summary = corpus.upsert_papers(keep)
    report.added += summary.inserted
    report.refreshed += summary.updated
    report.kept += len(keep)
    for paper in keep:
        base_id = split_arxiv_version(paper.arxiv_id)[0]
        # The licence came with the record, so the one-by-one lookup is not needed.
        licence = licences.get(base_id)
        corpus.record_license(base_id, licence)
        report.licences += licence is not None


def format_harvest_report(report: HarvestReport) -> str:
    line = (
        f"{report.subject}: read {report.records:,} records changed since "
        f"{report.harvested_from} in {report.requests:,} request(s); kept {report.kept:,} "
        f"first submitted since {report.since} or new versions of held papers "
        f"({report.added:,} added, {report.refreshed:,} refreshed, "
        f"{report.licences:,} with a licence)"
    )
    if report.missing:
        line += f"; {report.missing:,} identifiers the query API did not return"
    if report.failed:
        return f"{line}; {report.stopped_because}. Re-run to read the same window again."
    return f"{line}; harvested through {report.newest_datestamp or report.harvested_from}"
