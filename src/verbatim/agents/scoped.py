"""What an agent can read of the corpus: an allow-list, never a passthrough (I11).

ScopedCorpus offers only the store methods tools.py reaches, each with the
hidden papers taken out, and `search`, which runs the store's hybrid search with
them filtered out of every leg: the opening search and every search tool go
through it. It has no `__getattr__` and no connection, so a method nobody listed
raises instead of leaking. A paper is hidden in every version, as PaperFilter
hides it. With nothing hidden it filters nothing, and every method returns what
the store does.

Counts over the whole corpus (holdings, reference_count, citation_graph_stats,
and the citation counts of papers still shown) include the hidden papers: a
number names no paper and quotes nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from typing import Any, TypeVar

from verbatim.arxiv import Paper
from verbatim.chunking import Chunk
from verbatim.corpus import ChunkMatch, CorpusStore, PaperFilter, unversioned
from verbatim.retrieval import hybrid_search

Row = TypeVar("Row")

# The store methods tools.py reaches. Nothing else is offered.
ALLOWED = (
    "get_paper", "papers_by_id", "latest_version_of", "papers_cited_by", "citation_counts",
    "chunks_by_id", "reference_count", "passage_span", "papers_updated_since",
    "papers_citing", "most_cited_held_papers", "most_cited_absent_papers", "holdings",
    "find_papers_by_title", "citation_graph_stats", "chunks_on_pages", "chunks_for_paper",
)


class ScopedCorpus:
    """The corpus with some papers hidden, offering only what agents' tools read."""

    def __init__(self, store: CorpusStore, hidden: Iterable[str] = ()) -> None:
        self._store = store
        self._hidden = frozenset(unversioned(arxiv_id) for arxiv_id in hidden)

    @property
    def hidden(self) -> tuple[str, ...]:
        """The hidden papers, without versions."""
        return tuple(sorted(self._hidden))

    def search(self, embedder: Any, query: str, *, papers: PaperFilter | None = None,
               **options: Any) -> list[ChunkMatch]:
        """hybrid_search on the store, the hidden papers excluded from every leg.
        Any filter stands the title and graph legs down, so with nothing hidden the
        caller's own filter goes through untouched, None included."""
        if self._hidden:
            given = papers or PaperFilter()
            papers = replace(given, exclude_ids=tuple(dict.fromkeys(
                (*given.exclude_ids, *self.hidden))))
        return hybrid_search(self._store, embedder, query, papers=papers, **options)

    # -- papers --

    def get_paper(self, arxiv_id: str) -> Paper | None:
        return self._paper(self._store.get_paper(arxiv_id))

    def latest_version_of(self, base_arxiv_id: str) -> Paper | None:
        return self._paper(self._store.latest_version_of(base_arxiv_id))

    def papers_by_id(self, arxiv_ids: Sequence[str]) -> dict[str, Paper]:
        return {arxiv_id: found for arxiv_id, found in self._store.papers_by_id(arxiv_ids).items()
                if self._shown(arxiv_id)}

    def find_papers_by_title(self, text: str, *, limit: int = 5) -> list[Paper]:
        return self._top(lambda n: self._store.find_papers_by_title(text, limit=n), limit)

    def papers_updated_since(self, since: str | None = None, *, category: str | None = None,
                             limit: int = 20, latest_only: bool = True) -> list[Paper]:
        return self._top(lambda n: self._store.papers_updated_since(
            since, category=category, limit=n, latest_only=latest_only), limit)

    # -- the citation graph --

    def papers_cited_by(self, arxiv_id: str, *, limit: int = 50) -> list[Paper]:
        if not self._shown(arxiv_id):
            return []
        return self._top(lambda n: self._store.papers_cited_by(arxiv_id, limit=n), limit)

    def papers_citing(self, base_arxiv_id: str, *, limit: int = 25) -> list[Paper]:
        if not self._shown(base_arxiv_id):
            return []
        return self._top(lambda n: self._store.papers_citing(base_arxiv_id, limit=n), limit)

    def citation_counts(self, base_arxiv_ids: Sequence[str]) -> dict[str, int]:
        return {base: count for base, count in self._store.citation_counts(base_arxiv_ids).items()
                if self._shown(base)}

    def most_cited_held_papers(self, *, limit: int = 25) -> list[tuple[Paper, int]]:
        return self._top(lambda n: self._store.most_cited_held_papers(limit=n), limit,
                         key=lambda pair: pair[0].arxiv_id)

    def most_cited_absent_papers(self, *, limit: int = 50) -> list[tuple[str, int]]:
        # Papers cited and not held; a hidden paper is held, so it is never one of them.
        return self._store.most_cited_absent_papers(limit=limit)

    def reference_count(self) -> int:
        return self._store.reference_count()

    def citation_graph_stats(self) -> dict[str, int]:
        return self._store.citation_graph_stats()

    # -- passages --

    def chunks_by_id(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return [chunk for chunk in self._store.chunks_by_id(chunk_ids)
                if self._shown(chunk.arxiv_id)]

    def chunks_for_paper(self, arxiv_id: str) -> list[Chunk]:
        return self._store.chunks_for_paper(arxiv_id) if self._shown(arxiv_id) else []

    def chunks_on_pages(self, arxiv_id: str, pages: Sequence[int]) -> list[Chunk]:
        return self._store.chunks_on_pages(arxiv_id, pages) if self._shown(arxiv_id) else []

    def passage_span(self, arxiv_id: str) -> tuple[int, int | None, int | None]:
        return self._store.passage_span(arxiv_id) if self._shown(arxiv_id) else (0, None, None)

    def holdings(self, *, categories: int = 10) -> dict[str, object]:
        return self._store.holdings(categories=categories)

    # -- filtering --

    def _shown(self, arxiv_id: str) -> bool:
        return unversioned(arxiv_id) not in self._hidden

    def _paper(self, found: Paper | None) -> Paper | None:
        return found if found is not None and self._shown(found.arxiv_id) else None

    def _top(self, fetch: Callable[[int], list[Row]], limit: int,
             key: Callable[[Row], str] = lambda found: found.arxiv_id) -> list[Row]:  # type: ignore[attr-defined]
        """The store's first `limit` rows still shown, asking it for more while
        hidden papers crowd them out."""
        if not self._hidden:
            return fetch(limit)
        asked = limit + len(self._hidden)
        while True:
            rows = fetch(asked)
            shown = [row for row in rows if self._shown(key(row))]
            if len(shown) >= limit or len(rows) < asked:
                return shown[:limit]
            asked *= 2
