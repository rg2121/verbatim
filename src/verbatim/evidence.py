"""What an answer rests on, stated so a reader can judge it.

No detector tried could tell a covered-but-unanswered question from an answered
one, so each answer carries facts instead of a verdict: how many passages and
papers it used, what they are about, and, for a "what is X" question, which
held papers are named after X. Nothing here changes the answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from verbatim.corpus import ChunkMatch, CorpusStore
from verbatim.graph import anchor_papers, definitional_subject

# Fewer distinct papers than this means the answer rests on one source.
THIN_EVIDENCE_PAPERS = 2


@dataclass(frozen=True)
class Evidence:
    """What an answer was written from, in facts rather than a verdict."""

    passages: int
    papers: int
    subject: str | None = None
    closeness: float | None = None
    definitional_subject: str | None = None
    named_after: tuple[str, ...] = ()

    def lines(self) -> list[str]:
        """The disclosure, as lines to print under an answer."""
        if not self.passages:
            return []
        told = [f"Answered from {self.passages} passages in {self.papers} paper"
                f"{'s' if self.papers != 1 else ''}."]
        if self.papers < THIN_EVIDENCE_PAPERS:
            told.append("All of it from a single paper.")
        if self.subject:
            told.append(f"Those passages are about: {self.subject}")
        # The negative ("no paper is named after X") was measured and dropped:
        # true, but it fired on most good answers too.
        if self.named_after:
            told.append("Named after it here: " + ", ".join(self.named_after) + ".")
        return told


def describe_evidence(
    corpus: CorpusStore,
    question: str,
    matches: list[ChunkMatch],
    *,
    subject: str | None = None,
    closeness: float | None = None,
) -> Evidence:
    """Account for the evidence behind an answer; `subject` comes from the caller."""
    papers = {match.chunk.arxiv_id for match in matches}
    asked_after = definitional_subject(question)
    named: tuple[str, ...] = ()
    if asked_after and matches:
        named = tuple(
            f"{paper.title[:60]} (arXiv:{paper.arxiv_id})"
            for paper, _cited in anchor_papers(corpus, asked_after)
        )
    return Evidence(
        passages=len(matches),
        papers=len(papers),
        subject=subject,
        closeness=closeness,
        definitional_subject=asked_after,
        named_after=named,
    )
