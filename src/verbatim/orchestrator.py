"""The orchestrator behind `ask --deep` and `report --agents`.

Each section is routed by its question shape to a ladder of retrieval
strategies. After each attempt a deterministic verdict decides: accept when a
citation verified and the passages are on-subject, otherwise retry with the
next strategy, never the same one. Every attempt is recorded in a ledger. The
supervisor is code; models only plan sections, write prose, name subjects
and, on the last rung, choose a tool.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
import re
import sqlite3
import threading
from typing import Any

from verbatim import config
from verbatim.agent import ToolCaller, gather, records_answer
from verbatim.answers import Answer, Completion, answer_from_matches, subject_check
from verbatim.contracts import Budget, BudgetGuard, Cost, Critique, Gathered, Verdict
from verbatim.corpus import CorpusStore
from verbatim.critique import critique as run_critique
from verbatim.embedding import Embedder
from verbatim.evidence import describe_evidence
from verbatim.graph import definitional_subject
from verbatim.report import Report, ReportSection, plan_sections
from verbatim.retrieval import comparison_shaped, default_search_options, hybrid_search

# Questions about what is new, which passage similarity cannot answer and the
# tool-using agent can, by choosing the recency tool.
RECENCY = re.compile(
    r"\b(?:latest|newest|most recent|recently|this (?:week|month|year)|last (?:week|month|year))\b",
    re.IGNORECASE,
)

# The strategies each shape tries, in order. Every rung retrieves differently, so
# a retry is never blind; "agent" lets a model choose a tool.
LADDERS: dict[str, tuple[str, ...]] = {
    "definitional": ("hybrid", "titles", "agent"),
    "comparison": ("compare", "titles", "agent"),
    "recency": ("agent", "hybrid"),
    "open": ("hybrid", "titles", "agent"),
}
# Each is the default search with its own legs added.
STRATEGY_FLAGS: dict[str, dict[str, bool]] = {
    "hybrid": {},
    "compare": {"use_compare": True},
    "titles": {"use_titles": True},
}
ACCEPTED = "accepted"
RETRY = "retry"


def question_shape(question: str) -> str:
    """Which ladder a section climbs. Deterministic, like the retrieval gates."""
    if RECENCY.search(question):
        return "recency"
    if comparison_shaped(question):
        return "comparison"
    if definitional_subject(question) is not None:
        return "definitional"
    return "open"


@dataclass(frozen=True)
class Attempt:
    """One rung of the ladder: what was tried and what the verifier found.

    `route` is the agent rung's tool trace; `error` marks a rung that could not run.
    """

    strategy: str
    passages: int
    verified: int
    rejected: int
    closeness: float | None
    verdict: str
    route: str = ""
    error: str = ""
    # The passages this attempt stood on.
    papers: tuple[str, ...] = ()
    chunks: tuple[str, ...] = ()
    cost: Cost = Cost()
    critique: Critique | None = None


@dataclass(frozen=True)
class SectionLedger:
    heading: str
    shape: str
    attempts: tuple[Attempt, ...]
    outcome: str


@dataclass(frozen=True)
class Ledger:
    """What every role did for every section - the trace, kept for the reader."""

    sections: list[SectionLedger]

    @property
    def escalations(self) -> int:
        return sum(len(section.attempts) - 1 for section in self.sections)

    def render_markdown(self) -> str:
        lines = [
            "## How this review was made",
            "",
            "Each section was routed by its shape, retrieved by a specialist "
            "strategy, written by the model, and checked mechanically: a "
            "section whose citations failed verification, or whose passages "
            "drifted off-subject, was retried with the next strategy rather "
            "than the same one again. The best attempt is the one printed.",
            "",
        ]
        for section in self.sections:
            lines.append(f"- **{section.heading}** ({section.shape}) - {section.outcome}")
            for attempt in section.attempts:
                subject = (
                    f"subject {attempt.closeness:.2f}"
                    if attempt.closeness is not None
                    else "subject unchecked"
                )
                if attempt.error:
                    lines.append(
                        f"  - `{attempt.strategy}`: failed - {attempt.error}"
                    )
                    continue
                route = f" via {attempt.route}" if attempt.route else ""
                cost = f", {attempt.cost.render()}" if attempt.cost.calls else ""
                lines.append(
                    f"  - `{attempt.strategy}`{route}: {attempt.passages} passages, "
                    f"{attempt.verified} verified, {attempt.rejected} rejected, "
                    f"{subject}{cost} -> {attempt.verdict}"
                )
                if attempt.papers:
                    shown = ", ".join(f"arXiv:{paper}" for paper in attempt.papers[:4])
                    more = "" if len(attempt.papers) <= 4 else f" (+{len(attempt.papers) - 4})"
                    lines.append(f"    - stood on {shown}{more}")
                if attempt.critique is not None:
                    found = attempt.critique
                    if found.unaddressed:
                        lines.append(
                            "    - no passage covers: " + ", ".join(found.unaddressed)
                        )
                    if found.unsupported:
                        lines.append(
                            f"    - {len(found.unsupported)} sentence(s) carry a claim "
                            "with no citation"
                        )
                    if found.conflicts:
                        lines.append(
                            f"    - {len(found.conflicts)} pair(s) of cited passages disagree"
                        )
        return "\n".join(lines)


def verdict_for(
    answer: Answer, closeness: float | None, warn_below: float, *, listed: bool = False
) -> Verdict:
    """Accept or retry, from numbers only.

    Accepted when a citation verified and the passages are on-subject;
    `listed` accepts a recency answer listed from the catalogue.
    """
    if listed:
        return Verdict(True, "listed from the catalogue; nothing was generated")
    if not answer.citations:
        return Verdict(False, "no citation survived verification")
    if closeness is not None and closeness < warn_below:
        return Verdict(False, f"passages are off-subject ({closeness:.2f})")
    return Verdict(True, "a citation verified and the passages are on-subject")


def retrieve(
    strategy: str,
    corpus: CorpusStore,
    embedder: Embedder,
    question: str,
    *,
    limit: int,
    max_per_paper: int | None,
    caller: ToolCaller | None,
) -> Gathered:
    """Run one strategy; the route is the tool trace on the agent rung."""
    if strategy == "agent":
        if caller is None:
            return Gathered(question, (), strategy, "no tool-choosing model available")
        matches, trace = gather(corpus, embedder, caller, question)
        return Gathered(
            question, tuple(matches[:limit]), strategy, trace.describe(),
            records=tuple(trace.metadata),
        )
    found = hybrid_search(
        corpus, embedder, question, limit=limit, max_per_paper=max_per_paper,
        **default_search_options(), **STRATEGY_FLAGS[strategy],
    )
    return Gathered(question, tuple(found), strategy)


def answered_from_records(gathered: Gathered, shape: str) -> bool:
    """Whether this attempt answers a recency question from catalogue rows."""
    return shape == "recency" and not gathered.matches and bool(gathered.records)


def write(
    corpus: CorpusStore,
    completion: Completion,
    heading: str,
    gathered: Gathered,
    shape: str,
) -> Answer:
    """The answer one attempt can give. Shared by both supervisors."""
    if answered_from_records(gathered, shape):
        return records_answer(heading, list(gathered.records))
    return answer_from_matches(corpus, completion, heading, list(gathered.matches))


class Meters:
    """What an attempt spent across every client, each counted once."""

    def __init__(self, *clients: object) -> None:
        counters = (getattr(client, "usage", None) for client in clients)
        self.usages = list({id(u): u for u in counters if u is not None}.values())

    def mark(self) -> list[tuple[int, int, int, float]]:
        return [usage.snapshot() for usage in self.usages]

    def since(self, marks: list[tuple[int, int, int, float]] | None) -> Cost:
        if not marks:
            return Cost()
        deltas = [usage.since(mark) for usage, mark in zip(self.usages, marks, strict=True)]
        return Cost(
            sum(d.calls for d in deltas),
            sum(d.prompt_tokens for d in deltas),
            sum(d.output_tokens for d in deltas),
            round(sum(d.seconds for d in deltas), 3),
        )


def work_section(
    corpus: CorpusStore,
    embedder: Embedder,
    completion: Completion,
    heading: str,
    *,
    caller: ToolCaller | None = None,
    per_section: int = 6,
    max_per_paper: int | None = 2,
    check_subject: bool = True,
    warn_below: float = config.SUBJECT_WARN_BELOW,
    ladder: tuple[str, ...] | None = None,
    guard: BudgetGuard | None = None,
    complete_text: Callable[[str, str], str] | None = None,
    critic: bool = False,
    on_attempt: Callable[[str, int, int], None] | None = None,
) -> tuple[ReportSection, SectionLedger]:
    """Climb the ladder for one section until the verifier accepts.

    The best attempt is kept, not the last. `ladder` overrides the router (the
    plain report uses one rung). `on_attempt` reports each rung before it runs.
    A rung that raises is recorded as failed and the ladder continues. With
    `critic`, uncovered subjects force a retry; uncited sentences are only
    reported, since another retrieval cannot fix them.
    """
    shape = question_shape(heading)
    meters = Meters(completion, caller)
    candidates: list[tuple[Attempt, ReportSection]] = []
    rungs = ladder or LADDERS[shape]
    for rung, strategy in enumerate(rungs, start=1):
        if on_attempt is not None:
            on_attempt(strategy, rung, len(rungs))
        if guard is not None:
            stop = guard.exhausted()
            if stop is not None:
                candidates.append((
                    Attempt(strategy=strategy, passages=0, verified=0, rejected=0,
                            closeness=None, verdict=RETRY, error=f"budget: {stop}"),
                    ReportSection(
                        heading=heading,
                        answer=Answer(
                            question=heading,
                            text=f"This section was not written: {stop}.",
                            citations=[], rejected=[], declined=True,
                        ),
                    ),
                ))
                break
            guard.take()
        spent_before = meters.mark()
        try:
            gathered = retrieve(
                strategy, corpus, embedder, heading,
                limit=per_section, max_per_paper=max_per_paper, caller=caller,
            )
            matches = list(gathered.matches)
            answer = write(corpus, completion, heading, gathered, shape)
            subject = closeness = None
            if matches and check_subject:
                subject, closeness = subject_check(
                    completion, embedder, heading, matches
                )
            found = (
                run_critique(heading, answer, gathered, complete_text=complete_text)
                if critic else None
            )
        except (RuntimeError, OSError, sqlite3.Error) as error:
            candidates.append((
                Attempt(strategy=strategy, passages=0, verified=0, rejected=0,
                        closeness=None, verdict=RETRY, error=str(error)[:200]),
                ReportSection(
                    heading=heading,
                    answer=Answer(
                        question=heading,
                        text=f"This section could not be written: {error}",
                        citations=[], rejected=[], declined=True,
                    ),
                ),
            ))
            continue
        verdict = verdict_for(
            answer, closeness, warn_below, listed=answered_from_records(gathered, shape)
        )
        # A critique escalates only where a different retrieval could help.
        accepted = verdict.accepted and not (found.unaddressed if found else ())
        evidence = describe_evidence(
            corpus, heading, matches, subject=subject, closeness=closeness
        )
        spent = meters.since(spent_before)
        candidates.append((
            Attempt(
                strategy=strategy, passages=len(matches),
                verified=len(answer.citations), rejected=len(answer.rejected),
                closeness=closeness, verdict=ACCEPTED if accepted else RETRY,
                route=gathered.route, papers=gathered.paper_ids,
                chunks=gathered.chunk_ids, cost=spent, critique=found,
            ),
            ReportSection(
                heading=heading, answer=answer,
                subject=subject, subject_closeness=closeness,
                evidence_lines=tuple(evidence.lines()),
            ),
        ))
        if accepted:
            break
    return choose_best(candidates, heading, shape)


def choose_best(
    candidates: list[tuple[Attempt, ReportSection]], heading: str, shape: str
) -> tuple[ReportSection, SectionLedger]:
    """Pick the best attempt, not the last, and state the section's outcome.

    Shared by both supervisors, so they differ only in control flow.
    """
    best_attempt, best_section = max(
        candidates,
        key=lambda pair: (
            pair[0].verdict == ACCEPTED,
            pair[0].verified,
            pair[0].closeness or 0.0,
            pair[0].passages,
        ),
    )
    errored = [attempt for attempt, _ in candidates if attempt.error]
    if best_attempt.verdict == ACCEPTED:
        outcome = "accepted"
    elif best_attempt.verified and best_attempt.critique and best_attempt.critique.unaddressed:
        # Held back by the critic, not the subject check.
        outcome = (
            "accepted, though no passage covers: "
            + ", ".join(best_attempt.critique.unaddressed)
        )
    elif best_attempt.verified:
        outcome = "accepted with a subject warning"
    elif errored:
        outcome = (
            f"failed: {len(errored)} of {len(candidates)} strategies errored "
            f"({errored[0].error})"
        )
    else:
        outcome = "gap: no strategy produced a verified citation"
    return best_section, SectionLedger(
        heading=heading, shape=shape,
        attempts=tuple(attempt for attempt, _ in candidates), outcome=outcome,
    )


class _Serialized:
    """One embedder shared by worker threads, one call at a time."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._lock = threading.Lock()

    @property
    def model(self) -> Any:
        with self._lock:
            return self._embedder.model

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            return self._embedder.embed_query(text)

    def embed_passages(self, texts: Any) -> list[list[float]]:
        with self._lock:
            return self._embedder.embed_passages(texts)


SUPERVISORS = ("builtin", "langgraph")


def _supervisor(name: str) -> Callable[..., tuple[ReportSection, SectionLedger]]:
    """The function that climbs one section's ladder; LangGraph is imported only if chosen."""
    if name == "builtin":
        return work_section
    if name == "langgraph":
        from verbatim.graph_supervisor import work_section_with_graph

        return work_section_with_graph
    raise ValueError(
        f"unknown supervisor {name!r}; known: {', '.join(SUPERVISORS)}"
    )


def orchestrate(
    corpus: CorpusStore,
    embedder: Embedder,
    completion: Completion,
    complete_text: Callable[[str, str], str],
    topic: str,
    *,
    caller: ToolCaller | None = None,
    max_sections: int = 4,
    per_section: int = 6,
    max_per_paper: int | None = 2,
    workers: int = 1,
    check_subject: bool = True,
    warn_below: float = config.SUBJECT_WARN_BELOW,
    ladder: tuple[str, ...] | None = None,
    budget: Budget | None = None,
    critic: bool = False,
    on_section: Callable[[str, int, int], None] | None = None,
    on_section_done: Callable[[ReportSection, SectionLedger], None] | None = None,
    supervisor: str = "builtin",
) -> tuple[Report, Ledger]:
    """Plan a review, then work its sections, in parallel when `workers` > 1.

    Each worker thread opens its own corpus connection and shares one
    serialised embedder. `on_section_done` receives each section as it
    finishes, from a worker thread when `workers` > 1.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    climb = _supervisor(supervisor)
    headings = plan_sections(complete_text, topic, max_sections=max_sections)
    shared = _Serialized(embedder) if workers > 1 else embedder
    guard = (
        BudgetGuard(
            budget, getattr(completion, "usage", None), getattr(caller, "usage", None)
        )
        if budget is not None else None
    )

    def run(position: int, heading: str) -> tuple[ReportSection, SectionLedger]:
        if on_section is not None:
            on_section(heading, position, len(headings))
        options = dict(
            caller=caller, per_section=per_section, max_per_paper=max_per_paper,
            check_subject=check_subject, warn_below=warn_below, ladder=ladder,
            guard=guard, complete_text=complete_text, critic=critic,
        )
        if workers == 1:
            worked = climb(corpus, shared, completion, heading, **options)
        else:
            with CorpusStore(corpus.path) as own:
                worked = climb(own, shared, completion, heading, **options)
        if on_section_done is not None:
            on_section_done(*worked)
        return worked

    if workers == 1:
        results = [run(position, heading) for position, heading in enumerate(headings, start=1)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda pair: run(*pair), enumerate(headings, start=1)))

    return (
        Report(
            topic=topic,
            sections=[section for section, _ in results],
            generated_at=datetime.now(UTC).strftime("%Y-%m-%d"),
        ),
        Ledger(sections=[ledger for _, ledger in results]),
    )
