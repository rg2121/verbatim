"""Small, polite client for arXiv's metadata API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from collections.abc import Sequence
from typing import Callable
from urllib.parse import urlencode, urlparse, urlunparse
from xml.etree import ElementTree

from verbatim.fetching import fetch

ARXIV_API_URL = "https://export.arxiv.org/api/query"
# Licence lives only here; the query API omits it entirely.
ARXIV_OAI_URL = "https://export.arxiv.org/oai2"
OAI_NAMESPACE = "http://www.openarchives.org/OAI/2.0/"
ARXIV_OAI_METADATA_NAMESPACE = "http://arxiv.org/OAI/arXiv/"
ARXIV_HOSTS = frozenset({"arxiv.org", "www.arxiv.org", "export.arxiv.org"})
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
OPENSEARCH_NAMESPACE = "http://a9.com/-/spec/opensearch/1.1/"
ARXIV_NAMESPACE = "http://arxiv.org/schemas/atom"


class ArxivAPIError(ValueError):
    """An error entry returned by arXiv's API in an otherwise valid feed."""


@dataclass(frozen=True)
class Paper:
    """Metadata needed to trace a paper back to its source."""

    arxiv_id: str
    title: str
    summary: str
    authors: list[str]
    categories: list[str]
    published: str
    updated: str
    source_url: str
    pdf_url: str | None
    license_url: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FeedPage:
    """One page of search results, and how many exist in total."""

    papers: list[Paper]
    total_results: int
    start: int


# arXiv rejects an id_list far longer than this, and a rejected batch loses
# every identifier in it.
MAX_IDS_PER_REQUEST = 100


def fetch_metadata_page(
    category: str,
    *,
    start: int = 0,
    page_size: int = 100,
    sort_by: str = "lastUpdatedDate",
    http_get: Callable[[str], bytes] | None = None,
) -> FeedPage:
    """Fetch one page of a category, for callers that page through results.

    arXiv caps a single request well below the size of a real corpus, so
    anything past a few hundred papers has to be paged.  `total_results` comes
    from the feed itself, which is what lets a caller know when to stop.
    """
    if start < 0:
        raise ValueError("start must not be negative")
    if page_size < 1:
        raise ValueError("page_size must be at least 1")
    if sort_by not in {"lastUpdatedDate", "submittedDate"}:
        raise ValueError("sort_by must be 'lastUpdatedDate' or 'submittedDate'")

    query = urlencode(
        {
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": page_size,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
    )
    get = http_get or _get
    xml = get(f"{ARXIV_API_URL}?{query}")
    return FeedPage(
        papers=parse_feed(xml), total_results=parse_total_results(xml), start=start
    )


def fetch_papers_by_id(
    arxiv_ids: Sequence[str],
    *,
    http_get: Callable[[str], bytes] | None = None,
) -> list[Paper]:
    """Fetch metadata for specific papers, named by identifier.

    Category queries return what is newest; this returns what was asked for,
    which is what growing a corpus along its own citations needs. arXiv caps a
    single id_list request, so callers batch.
    """
    if not arxiv_ids:
        return []
    if len(arxiv_ids) > MAX_IDS_PER_REQUEST:
        raise ValueError(
            f"arXiv accepts at most {MAX_IDS_PER_REQUEST} identifiers per request"
        )
    query = urlencode({"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)})
    get = http_get or _get
    return parse_feed(get(f"{ARXIV_API_URL}?{query}"))


def parse_total_results(xml: bytes) -> int:
    """Read how many papers match the query, across all pages."""
    root = ElementTree.fromstring(xml)
    element = root.find(f"{{{OPENSEARCH_NAMESPACE}}}totalResults")
    if element is None or not (element.text or "").strip():
        return 0
    return int(element.text.strip())


def fetch_metadata(
    category: str,
    max_results: int,
    *,
    sort_by: str = "lastUpdatedDate",
    http_get: Callable[[str], bytes] | None = None,
) -> list[Paper]:
    """Fetch recent papers in one arXiv category.

    `http_get` exists to keep parsing deterministic and network-free in tests.
    """
    if max_results < 1:
        raise ValueError("max_results must be at least 1")
    if sort_by not in {"lastUpdatedDate", "submittedDate"}:
        raise ValueError("sort_by must be 'lastUpdatedDate' or 'submittedDate'")

    query = urlencode(
        {
            "search_query": f"cat:{category}",
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
    )
    url = f"{ARXIV_API_URL}?{query}"
    get = http_get or _get
    return parse_feed(get(url))


def parse_feed(xml: bytes) -> list[Paper]:
    """Parse an arXiv Atom response into provenance-preserving paper records."""
    root = ElementTree.fromstring(xml)
    namespace = {"atom": ATOM_NAMESPACE, "arxiv": ARXIV_NAMESPACE}
    papers: list[Paper] = []

    for entry in root.findall("atom:entry", namespace):
        raw_source_url = _required_text(entry, "atom:id", namespace)
        arxiv_id = _arxiv_id_from_source(raw_source_url)
        source_url = _canonical_arxiv_url(raw_source_url, required_path_prefix="/abs/")
        pdf_url = _find_pdf_url(entry, namespace)
        license_element = entry.find("arxiv:license", namespace)
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=_clean(_required_text(entry, "atom:title", namespace)),
                summary=_clean(_required_text(entry, "atom:summary", namespace)),
                authors=[
                    _required_text(author, "atom:name", namespace)
                    for author in entry.findall("atom:author", namespace)
                ],
                categories=_categories(entry, namespace),
                published=_required_text(entry, "atom:published", namespace),
                updated=_required_text(entry, "atom:updated", namespace),
                source_url=source_url,
                pdf_url=pdf_url,
                license_url=(license_element.text if license_element is not None else None),
            )
        )
    return papers


def license_class(license_url: str | None) -> str:
    """Group a licence URL by what it actually permits.

    The distinction matters more than presence: arXiv's default
    non-exclusive-distrib licence grants *arXiv* the right to distribute and
    grants a third party nothing, so treating "has a licence" as "is reusable"
    would be wrong. Nor is every Creative Commons licence equivalent:

    - `open` — CC0, CC-BY, CC-BY-SA. Redistribution and derivatives permitted,
      subject to attribution and, for SA, share-alike.
    - `noncommercial` — carries -NC. No commercial use.
    - `no-derivatives` — carries -ND. A trained model is arguably a derivative,
      so this excludes training regardless of commercial intent.
    - `arxiv-nonexclusive` — arXiv may distribute it; you may not.

    Only `open` is safe for redistribution or training.
    """
    if not license_url:
        return "unknown"
    url = license_url.lower()
    if "creativecommons.org/publicdomain" in url or "/zero/" in url:
        return "open"
    if "creativecommons.org/licenses/" in url:
        # NC and ND are not "open" for the uses this project cares about, and
        # lumping them in with CC-BY would misreport a third of the licensed
        # corpus as reusable. The stricter term wins when both are present.
        if "-nd" in url:
            return "no-derivatives"
        if "-nc" in url:
            return "noncommercial"
        return "open"
    if "arxiv.org/licenses/nonexclusive-distrib" in url:
        return "arxiv-nonexclusive"
    return "other"


def fetch_license(
    base_arxiv_id: str, *, http_get: Callable[[str], bytes] | None = None
) -> str | None:
    """Look up one paper's licence via arXiv's OAI-PMH interface.

    The query API does not return `<arxiv:license>` at all — measured across 59
    papers, every one came back without it — so licence has to come from the
    OAI-PMH record, which does carry it. Takes the base identifier without a
    version, which is what OAI-PMH is keyed on.
    """
    query = urlencode(
        {
            "verb": "GetRecord",
            "identifier": f"oai:arXiv.org:{base_arxiv_id}",
            "metadataPrefix": "arXiv",
        }
    )
    get = http_get or _get
    return parse_license(get(f"{ARXIV_OAI_URL}?{query}"))


def parse_license(xml: bytes) -> str | None:
    """Read the licence out of an OAI-PMH record, if the record carries one."""
    root = ElementTree.fromstring(xml)
    error = root.find("oai:error", {"oai": OAI_NAMESPACE})
    if error is not None:
        raise ArxivAPIError(f"arXiv OAI-PMH error: {error.get('code')}: {error.text}")
    element = root.find(f".//{{{ARXIV_OAI_METADATA_NAMESPACE}}}license")
    if element is None or element.text is None:
        return None
    return element.text.strip() or None


def _get(url: str, *, sleep: Callable[[float], None] = time.sleep) -> bytes:
    return fetch(url, sleep=sleep)


def _required_text(element: ElementTree.Element, path: str, namespace: dict[str, str]) -> str:
    found = element.find(path, namespace)
    if found is None or found.text is None:
        raise ValueError(f"Missing required arXiv API field: {path}")
    return found.text


def _arxiv_id_from_source(source_url: str) -> str:
    """Extract modern and legacy arXiv identifiers without losing archive names."""
    parsed = urlparse(source_url)
    if parsed.path == "/api/errors":
        detail = parsed.fragment or source_url
        raise ArxivAPIError(f"arXiv API returned an error entry: {detail}")
    if not parsed.path.startswith("/abs/"):
        raise ValueError(f"Unexpected arXiv entry URL: {source_url}")
    arxiv_id = parsed.path.removeprefix("/abs/")
    if not arxiv_id:
        raise ValueError(f"Missing arXiv identifier in URL: {source_url}")
    return arxiv_id


def _canonical_arxiv_url(url: str, *, required_path_prefix: str) -> str:
    """Accept known arXiv URLs and normalize them to HTTPS."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if parsed.scheme not in {"http", "https"} or hostname not in ARXIV_HOSTS:
        raise ValueError(f"Unexpected arXiv URL: {url}")
    if not parsed.path.startswith(required_path_prefix):
        raise ValueError(f"Unexpected arXiv URL path: {url}")
    return urlunparse(parsed._replace(scheme="https", netloc=hostname))


def _find_pdf_url(entry: ElementTree.Element, namespace: dict[str, str]) -> str | None:
    for link in entry.findall("atom:link", namespace):
        if link.attrib.get("title") != "pdf":
            continue
        href = link.attrib.get("href")
        if href is None:
            raise ValueError("arXiv PDF link is missing an href")
        return _canonical_arxiv_url(href, required_path_prefix="/pdf/")
    return None


def _categories(entry: ElementTree.Element, namespace: dict[str, str]) -> list[str]:
    categories: list[str] = []
    for category in entry.findall("atom:category", namespace):
        term = category.attrib.get("term")
        if term is None:
            raise ValueError("arXiv category is missing a term")
        categories.append(term)
    return categories


def _clean(text: str) -> str:
    return " ".join(text.split())
