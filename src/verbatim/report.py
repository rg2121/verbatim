"""A literature review written section by section, every claim verified.

One model call plans the sections; each is answered through the verified-citation
path. The report merges them under one bibliography and keeps each section's
evidence and subject warnings beside it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any

from verbatim import config
from verbatim.answers import Answer, Completion
from verbatim.citations import render_bibliography
from verbatim.corpus import CorpusStore
from verbatim.embedding import Embedder
from verbatim.research import parse_plan_lines

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
    """One section: the question it asks, the answer, and what it rests on."""

    heading: str
    answer: Answer
    subject: str | None = None
    subject_closeness: float | None = None
    # The evidence account, already rendered to lines.
    evidence_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class Report:
    topic: str
    sections: list[ReportSection]
    generated_at: str


def plan_sections(
    complete_text: Callable[[str, str], str], topic: str, *, max_sections: int = 4
) -> list[str]:
    """The questions the review will answer, one per section; the topic alone if planning fails."""
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
    """Plan and write a review: the orchestrator with one rung and no retries."""
    # Imported here: the orchestrator imports this module.
    from verbatim.orchestrator import orchestrate

    report, _ledger = orchestrate(
        corpus,
        embedder,
        completion,
        complete_text,
        topic,
        caller=None,
        max_sections=max_sections,
        per_section=per_section,
        max_per_paper=max_per_paper,
        workers=1,
        check_subject=check_subject,
        ladder=("compare",) if use_compare else ("hybrid",),
        on_section=on_section,
    )
    return report


def render_markdown(
    report: Report,
    *,
    warn_below: float = config.SUBJECT_WARN_BELOW,
    ledger: Any = None,
) -> str:
    """The review as one markdown document with a single, renumbered bibliography.

    A passage cited by two sections appears once.
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
        if section.evidence_lines:
            # Always shown, so the reader can judge what the section rests on.
            lines += ["", "> " + " ".join(section.evidence_lines)]
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
    if ledger is not None:
        # The orchestrator's trace: strategies tried, what verified, what was retried.
        lines += ["", ledger.render_markdown()]
    return "\n".join(lines)
