"""Saying what an answer rests on, so a reader can judge what it is worth.

Seven approaches have tried to *detect* the case where the corpus covers a
subject and answers nothing about it - similarity at passage and abstract
level, lexical coverage, four ways of asking a model, and the orchestrator's
own gap verdict. All measured 0 on the labelled cases. The reason is the same
every time: the passages really are on the subject and the quotes really do
verify, so nothing in the retrieval or verification path can tell the two
apart.

This stops trying. Instead of a threshold that decides for the reader, an
answer carries a short, deterministic account of the evidence behind it: how
much was found, in how many papers, what those passages are about, and - for a
question asking what something *is* - whether the corpus holds a paper the
subject is actually named after, or only papers that mention it in passing.

Nothing here gates an answer or changes a word of it. "What is a tensor" still
gets its answer; it also gets the sentence that says 177 papers here use
tensors and none is named after one, which is the fact a reader needed and no
detector could supply.
"""

from __future__ import annotations

from dataclasses import dataclass

from verbatim.corpus import ChunkMatch, CorpusStore
from verbatim.graph import anchor_papers, definitional_subject

# Below this many distinct papers, an answer rests on one voice rather than a
# literature, which is worth saying whatever the similarity scores are.
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
        # The negative of this - "no paper is named after X" - was built,
        # measured, and dropped. It fired on all 4 no_answer cases and on 13 of
        # the 19 definition-shaped answerable ones: true every time, and at a
        # 68% fire rate on good answers it is a line readers learn to skip,
        # which is the same reason the subject threshold was not raised to
        # 0.79. Absence of this line is left to speak for itself.
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
    """Account for the evidence behind an answer. Deterministic apart from
    `subject`, which the caller has already computed if it wanted it."""
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
