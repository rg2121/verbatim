"""The citation graph the bibliographies already imply, and what it is for.

Measured over all 398,802 bibliography entries, 12,781 distinct edges resolve
into the corpus and 43% of papers touch at least one. (An earlier note said a
graph could not be built here; that was measured on a biased 11% sample - the
alphabetically first ids, i.e. the oldest papers, whose references point at
work still absent.)

Authority here is in-corpus in-degree: how many held papers cite this one.
It is deterministic, explainable in one sentence, and at an average degree of
1.61 a PageRank over these edges is in-degree with extra steps.

The graph's retrieval job is the definitional question, which passage ranking
loses roughly 60:1 - 3,152 papers mention Adam and the paper defining it holds
53 passages of hyperparameter-free prose. Title matching alone finds the Adam
paper but was measured worse overall (mrr 0.866 -> 0.822) because it also runs
on questions like "why does PPO clip its objective", where `clip` matches
CLIP. Anchoring differs in two ways, both gates:

- it fires only on questions *shaped* like definitions ("what is X",
  "define X"), so ordinary questions never see the extra leg;
- a title match anchors only when held papers actually cite it, so the corpus
  itself testifies that this is the paper a subject is named after, rather
  than one that happens to share a word with the question.
"""

from __future__ import annotations

import re

from verbatim.arxiv import Paper
from verbatim.corpus import ChunkMatch, CorpusStore, split_arxiv_version

# Questions that ask what something is, rather than how or why it behaves.
# Deliberately tight: "how does X work" retrieves fine as passages, and the
# always-on version of title matching was measured worse for exactly the
# questions this pattern excludes.
DEFINITIONAL = re.compile(
    r"^\s*(?:what\s+(?:is|are)|what's|define|definition\s+of)\s+(?P<subject>.+?)[\s?]*$",
    re.IGNORECASE,
)

# A single stray citation should not crown a paper the origin of a subject;
# two held papers citing it is the same bar `grow` uses for adding one.
MIN_ANCHOR_CITATIONS = 2
MAX_ANCHORS = 2
ANCHOR_CANDIDATES = 8
PASSAGES_PER_ANCHOR = 3


def definitional_subject(question: str) -> str | None:
    """The subject of a definition-shaped question, or None for any other.

    "What is the Adam optimizer?" yields "the Adam optimizer"; "why does PPO
    clip its objective" yields None, and that None is the point: it is the
    gate that keeps the anchor leg away from questions it was measured to
    hurt.
    """
    match = DEFINITIONAL.match(question)
    if match is None:
        return None
    subject = match.group("subject").strip()
    return subject or None


def anchor_papers(
    corpus: CorpusStore,
    subject: str,
    *,
    limit: int = MAX_ANCHORS,
    min_citations: int = MIN_ANCHOR_CITATIONS,
) -> list[tuple[Paper, int]]:
    """Papers a subject is likely named after, with their in-corpus citations.

    Candidates come from title search; the graph then decides. A recent paper
    whose title happens to share the subject's words has no held citers and is
    dropped, while the paper a field keeps citing rises regardless of where
    BM25 put it. Best-cited first.
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
    """Passages from the papers a definitional question is asking after.

    Empty for any question that is not definition-shaped, and for one whose
    subject no cited title matches - in both cases retrieval proceeds exactly
    as it would have without the graph. Within an anchor the query still
    chooses the passages, so the leg contributes evidence, not an arbitrary
    excerpt.
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
