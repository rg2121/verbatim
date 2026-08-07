"""Small, polite client for arXiv's metadata API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
ARXIV_NAMESPACE = "http://arxiv.org/schemas/atom"


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
        source_url = _required_text(entry, "atom:id", namespace)
        arxiv_id = source_url.rstrip("/").rsplit("/", 1)[-1]
        pdf_url = next(
            (
                link.attrib["href"]
                for link in entry.findall("atom:link", namespace)
                if link.attrib.get("title") == "pdf"
            ),
            None,
        )
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
                categories=[
                    category.attrib["term"]
                    for category in entry.findall("atom:category", namespace)
                ],
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


def _clean(text: str) -> str:
    return " ".join(text.split())

