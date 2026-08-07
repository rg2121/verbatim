"""Small, polite client for arXiv's metadata API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_HOSTS = frozenset({"arxiv.org", "www.arxiv.org", "export.arxiv.org"})
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
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


def fetch_metadata(
    category: str,
    max_results: int,
    *,
    http_get: Callable[[str], bytes] | None = None,
) -> list[Paper]:
    """Fetch recent papers in one arXiv category.

    `http_get` exists to keep parsing deterministic and network-free in tests.
    """
    if max_results < 1:
        raise ValueError("max_results must be at least 1")

    query = urlencode(
        {
            "search_query": f"cat:{category}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
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


def _get(url: str) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "verbatim/0.1 (personal research project)"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS API host
        return response.read()


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
