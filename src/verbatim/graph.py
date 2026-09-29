"""The in-corpus citation graph, and its retrieval leg for definitional questions.

Authority is in-corpus in-degree: how many held papers cite a paper. For a
"what is X" question, passage search is drowned by papers that merely use X,
so this leg adds passages from the papers X is named after: a title match that
held papers actually cite. It fires only on definition-shaped questions.
"""

from __future__ import annotations

import re

from verbatim.arxiv import Paper
from verbatim.corpus import ChunkMatch, CorpusStore, split_arxiv_version

# Questions asking what something is. Deliberately tight: other questions
# retrieve fine from passages.
DEFINITIONAL = re.compile(
    r"^\s*(?:what\s+(?:is|are)|what's|define|definition\s+of)\s+(?P<subject>.+?)[\s?]*$",
    re.IGNORECASE,
)

# Citations a title match needs to count as an anchor; `grow` uses the same bar.
MIN_ANCHOR_CITATIONS = 2
MAX_ANCHORS = 2
ANCHOR_CANDIDATES = 8
PASSAGES_PER_ANCHOR = 3


# "what is a tensor" names the subject "tensor".
LEADING_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)


def definitional_subject(question: str) -> str | None:
    """The subject of a definition-shaped question, or None for any other question."""
    match = DEFINITIONAL.match(question)
    if match is None:
        return None
    subject = LEADING_ARTICLE.sub("", match.group("subject").strip()).strip()
    return subject or None


def anchor_papers(
    corpus: CorpusStore,
    subject: str,
    *,
    limit: int = MAX_ANCHORS,
    min_citations: int = MIN_ANCHOR_CITATIONS,
) -> list[tuple[Paper, int]]:
    """Papers a subject is likely named after, best-cited first.

    Title search proposes candidates; only those held papers cite are kept.
    """
    candidates = corpus.search_titles(subject, limit=ANCHOR_CANDIDATES)
    if not candidates:
        return []
    papers = corpus.papers_by_id([arxiv_id for arxiv_id, _ in candidates])
    counts = corpus.citation_counts(
        [split_arxiv_version(arxiv_id)[0] for arxiv_id in papers]
    )
    anchored = [
        (paper, counts.get(split_arxiv_version(arxiv_id)[0], 0))
        for arxiv_id, _ in candidates
        if (paper := papers.get(arxiv_id)) is not None
    ]
    kept = [(paper, cited) for paper, cited in anchored if cited >= min_citations]
    # Ties keep title-match order, which sorted() preserves by being stable.
    kept.sort(key=lambda pair: -pair[1])
    return kept[:limit]


def anchored_matches(
    corpus: CorpusStore,
    query: str,
    *,
    per_paper: int = PASSAGES_PER_ANCHOR,
    include_references: bool = False,
) -> list[ChunkMatch]:
    """Passages from the papers a definitional question asks after.

    Empty for any other question, so retrieval is then unchanged.
    """
    subject = definitional_subject(query)
    if subject is None:
        return []
    matches: list[ChunkMatch] = []
    for paper, _cited in anchor_papers(corpus, subject):
        matches.extend(
            corpus.search_chunks(
                query,
                limit=per_paper,
                arxiv_id=paper.arxiv_id,
                match="any",
                include_references=include_references,
            )
        )
    return matches
