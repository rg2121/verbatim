"""A literature review written section by section, every claim still verified.

Composition, not new machinery. Planning the sections is one model call that
returns flat lines, the shape a local model actually follows. Each section is
then a retrieval with the measured defaults and an answer written by the
existing verified-citation path, so a claim cannot appear in the review unless
its quoted words are in the passage it names - the property the whole project
exists for, carried unchanged into a longer document.

What a report adds over `ask` run several times is the assembly: one
bibliography numbered across sections with the inline markers rewritten to
match, each section's subject check carried beside it as a warning rather
than silently dropped, and a section the corpus cannot support saying so in
place instead of being papered over. A hosted "deep research" product cannot
show its retrieval or verify its quotes; this one can, and the markdown it
writes is the evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import re

from verbatim import config
from verbatim.answers import (
    Answer,
    Completion,
    answer_from_matches,
    describe_subject,
    subject_similarity,
)
from verbatim.citations import render_bibliography
from verbatim.corpus import CorpusStore
from verbatim.embedding import Embedder
from verbatim.research import parse_plan_lines
from verbatim.retrieval import hybrid_search

MAX_SECTIONS = 6
MARKER = re.compile(r"\[(\d+)\]")

SECTION_SYSTEM = (
    "You outline a short literature review on a topic. Reply with one section "
    "per line and nothing else. Each line is a specific question about the "
    "topic that a search over research papers can answer, ordered so the "
    "sections read as a review: what it is, how it works or is used, what has "
    "been measured, what remains open. Write between three and five lines."
)


@dataclass(frozen=True)
class ReportSection:
    """One section: the question it asks, the answer, and its subject check."""

    heading: str
    answer: Answer
    subject: str | None = None
    subject_closeness: float | None = None


@dataclass(frozen=True)
class Report:
    topic: str
    sections: list[ReportSection]
    generated_at: str


def plan_sections(
    complete_text: Callable[[str, str], str], topic: str, *, max_sections: int = 4
) -> list[str]:
    """The questions the review will answer, one per section.

    Falls back to the topic as a single section when planning fails or
    returns nothing usable, so a report can always be produced.
    """
    if max_sections < 1:
        raise ValueError("max_sections must be at least 1")
    try:
        reply = complete_text(SECTION_SYSTEM, topic)
    except (RuntimeError, OSError):
        return [topic]
    return parse_plan_lines(reply, min(max_sections, MAX_SECTIONS)) or [topic]


def build_report(
    corpus: CorpusStore,
    embedder: Embedder,
    completion: Completion,
    complete_text: Callable[[str, str], str],
    topic: str,
    *,
    max_sections: int = 4,
    per_section: int = 6,
    max_per_paper: int | None = 2,
    use_compare: bool = False,
    check_subject: bool = True,
    on_section: Callable[[str, int, int], None] | None = None,
) -> Report:
    """Plan, retrieve, and write every section of a review on one topic.

    Each section runs the same retrieval `ask` would - hybrid with the
    definitional graph gate on, the comparison gate only if the caller turned
    it on - and the same subject check, kept per section because a review's
    sections can drift off-corpus one at a time.
    """
    headings = plan_sections(complete_text, topic, max_sections=max_sections)
    sections: list[ReportSection] = []
    for position, heading in enumerate(headings, start=1):
        if on_section is not None:
            on_section(heading, position, len(headings))
        matches = hybrid_search(
            corpus,
            embedder,
            heading,
            limit=per_section,
            max_per_paper=max_per_paper,
            use_graph=True,
            use_compare=use_compare,
        )
        answer = answer_from_matches(corpus, completion, heading, matches)
        subject = closeness = None
        if matches and check_subject:
            subject = describe_subject(completion, matches)
            closeness = subject_similarity(embedder, heading, subject)
        sections.append(
            ReportSection(
                heading=heading, answer=answer,
                subject=subject, subject_closeness=closeness,
            )
        )
    return Report(
        topic=topic,
        sections=sections,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d"),
    )


def render_markdown(
    report: Report, *, warn_below: float = config.SUBJECT_WARN_BELOW
) -> str:
    """The review as one markdown document with a single bibliography.

    Sections cite locally as [1], [2]; here every citation is renumbered into
    one reference list, a passage cited by two sections appearing once. The
    inline markers were already rewritten to match each section's own
    citations, so the remap is position to position, never a guess.
    """
    citations = []
    number_of: dict[str, int] = {}
    lines = [
        f"# {report.topic}",
        "",
        f"*Generated {report.generated_at} from a local arXiv corpus. Every "
        "quotation was verified character-for-character against the passage "
        "it cites; a claim that could not be verified is marked "
        "[unverified].*",
    ]
    for section in report.sections:
        mapping: dict[int, int] = {}
        for local, citation in enumerate(section.answer.citations, start=1):
            if citation.chunk_id not in number_of:
                citations.append(citation)
                number_of[citation.chunk_id] = len(citations)
            mapping[local] = number_of[citation.chunk_id]

        def remap(match: re.Match[str]) -> str:
            local = int(match.group(1))
            return f"[{mapping[local]}]" if local in mapping else match.group(0)

        lines += ["", f"## {section.heading}", "", MARKER.sub(remap, section.answer.text)]
        if (
            section.subject_closeness is not None
            and section.subject_closeness < warn_below
        ):
            lines += [
                "",
                f"> The passages behind this section are about: "
                f"{section.subject} - which may not be what the section asks "
                f"({section.subject_closeness:.2f} similarity).",
            ]
        if section.answer.rejected:
            lines += [
                "",
                f"> {len(section.answer.rejected)} citation(s) in this section "
                "failed verification and are marked [unverified].",
            ]
    if citations:
        lines += ["", "## References"]
        # A blank line between references, or markdown folds them into one
        # paragraph.
        for reference in render_bibliography(citations).splitlines():
            lines += ["", reference]
    return "\n".join(lines)
