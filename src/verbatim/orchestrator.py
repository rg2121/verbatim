"""Roles that differ in kind, coordinated in code, and measured against not doing it.

The 2026 literature on multi-agent systems agrees on more than it argues
about, and this module is built from the parts that agree:

- **Orchestrator and workers.** A lead coordinates specialists that work in
  parallel on independent facets; a published orchestrator-workers research
  system reports a 90% gain over a single agent on breadth-first tasks,
  and the LangGraph
  benchmark finds supervisor designs stay accurate as tools proliferate
  where a single agent degrades. The cost is tokens - 3 to 15 times a
  single agent - so the work has to be worth it.
- **Divide by context, not by problem type.** Facets that share context
  belong to one agent; facets that share nothing may be split. This is why
  splitting one question into sub-questions lost here (measured: worse on
  its own compound cases) while a review's *sections* are the right split -
  they share a topic and nothing else.
- **A verifier that is not the writer.** The separate citation agent in those
  systems exists because one agent both writing and checking plays "a game of
  telephone". Here the verifier is stronger than an agent: it is the
  deterministic quote check, and it never asks a model whether the answer is
  good, because models asked that always say yes (measured, four ways).
- **A ledger, and no blind retries.** Magentic-One's orchestrator keeps a
  progress ledger and retries on error; OrchestraBench (arXiv:2608.05263)
  finds blind retry reproduces latent faults and that detection *with
  attribution* is what makes containment possible. So every attempt here is
  recorded with what it retrieved, what verified, and why it was rejected,
  and a retry always changes strategy.
- **The supervisor never paraphrases.** The LangGraph benchmark's main
  failure mode was the supervisor's "translation layer" rewriting worker
  output. The supervisor here forwards each section's verified answer
  untouched.

The supervisor is code rather than a model for the same reason the retrieval
gates are: routing by question shape was measured to win, and code can be
tested offline in two seconds. Models do the narrow jobs - planning sections,
writing prose, naming what passages are about, and, as the last rung of the
ladder, choosing a tool - and the ledger says which one did what.
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
from verbatim.agent import ToolCaller, gather
from verbatim.answers import (
    Answer,
    Completion,
    answer_from_matches,
    subject_check,
)
from verbatim.contracts import Budget, BudgetGuard, Cost, Critique, Gathered, Verdict
from verbatim.corpus import CorpusStore
from verbatim.critique import critique as run_critique
from verbatim.embedding import Embedder
from verbatim.evidence import describe_evidence
from verbatim.graph import definitional_subject
from verbatim.report import Report, ReportSection, plan_sections
from verbatim.retrieval import comparison_shaped, hybrid_search

# Questions about what is new, which passage similarity cannot answer and the
# tool-using agent can, by choosing the recency tool.
RECENCY = re.compile(
    r"\b(?:latest|newest|most recent|recently|this (?:week|month|year)|last (?:week|month|year))\b",
    re.IGNORECASE,
)

# The strategies a section may be retrieved with, in the order each shape
# tries them. Every rung is a different retrieval, so a retry is never blind.
# "hybrid" is the measured default (the definitional graph gate is inside it);
# "compare" and "titles" are the legs measured best on multi-paper questions;
# "agent" lets a model choose a tool, which is the only rung that can answer a
# recency or named-paper question, and the most expensive.
LADDERS: dict[str, tuple[str, ...]] = {
    "definitional": ("hybrid", "titles", "agent"),
    "comparison": ("compare", "titles", "agent"),
    "recency": ("agent", "hybrid"),
    "open": ("hybrid", "titles", "agent"),
}
STRATEGY_FLAGS: dict[str, dict[str, bool]] = {
    "hybrid": {"use_graph": True},
    "compare": {"use_graph": True, "use_compare": True},
    "titles": {"use_graph": True, "use_titles": True},
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
    """One rung of the ladder: what was tried, and what the verifier found.

    `route` records how the agent rung actually gathered its passages - which
    tools it called, in order. Without it the ledger says a rung ran and not
    what it did, and this module's own argument is that containment needs
    detection *with attribution*.

    `error` records a rung that could not run at all, so a model that fell over
    is distinguishable in the ledger from one that answered badly.
    """

    strategy: str
    passages: int
    verified: int
    rejected: int
    closeness: float | None
    verdict: str
    route: str = ""
    error: str = ""
    # Which passages this attempt actually stood on, so the ledger can be
    # traced to sources rather than only to counts.
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
                    # Which sources, not only how many: a ledger that cannot be
                    # traced back to passages records that something happened
                    # rather than what it stood on.
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


def verdict_for(answer: Answer, closeness: float | None, warn_below: float) -> Verdict:
    """Accept or retry, from numbers only.

    A section is accepted when at least one citation survived verification
    and the passages are about what the section asks. No model is asked
    whether the answer is good: asked that, every model tested says yes.
    """
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
    """Run one strategy, returning its passages and how it got them.

    The route is empty for the deterministic strategies, whose name already
    says what they did, and is the agent's tool trace on the agent rung, which
    is the only rung whose behaviour varies from question to question.
    """
    if strategy == "agent":
        if caller is None:
            return Gathered(question, (), strategy, "no tool-choosing model available")
        matches, trace = gather(corpus, embedder, caller, question)
        return Gathered(question, tuple(matches[:limit]), strategy, trace.describe())
    found = hybrid_search(
        corpus, embedder, question,
        limit=limit, max_per_paper=max_per_paper, **STRATEGY_FLAGS[strategy],
    )
    return Gathered(question, tuple(found), strategy)


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

    The best attempt is kept, not the last: an escalation can retrieve worse
    than what it replaced, and the reader should get the strongest section
    the ladder found, with the ledger showing the rest.

    `ladder` overrides the router's choice, which is how the plain report runs
    this same loop with a single rung and no escalation.

    `on_attempt` is called with (strategy, rung, rungs) before each rung runs,
    for callers that must show progress while the loop blocks - a page waiting
    silently through three retrievals and three generations reads as a broken
    one. It is told what is *about* to be tried, because what was tried is
    already in the ledger by the time the loop returns.

    A rung that raises - an unreachable model, a corpus fault - is recorded as
    a failed attempt and the ladder continues. One failure in one section must
    not discard a review that may have taken ten minutes to write.

    With `critic`, each written section is also checked for claims carrying no
    citation and for subjects the question raises that no passage covers. Only
    the second escalates: a different retrieval can fix missing coverage and
    cannot fix a sentence the writer left uncited, so the first is reported to
    the reader instead of being retried.
    """
    shape = question_shape(heading)
    usage = getattr(completion, "usage", None)
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
        spent_before = usage.snapshot() if usage is not None else None
        try:
            gathered = retrieve(
                strategy, corpus, embedder, heading,
                limit=per_section, max_per_paper=max_per_paper, caller=caller,
            )
            matches = list(gathered.matches)
            answer = answer_from_matches(corpus, completion, heading, matches)
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
        verdict = verdict_for(answer, closeness, warn_below)
        # A critique escalates only where a different retrieval could help.
        accepted = verdict.accepted and not (found.unaddressed if found else ())
        evidence = describe_evidence(
            corpus, heading, matches, subject=subject, closeness=closeness
        )
        spent = Cost()
        if usage is not None and spent_before is not None:
            delta = usage.since(spent_before)
            spent = Cost(delta.calls, delta.prompt_tokens, delta.output_tokens,
                         delta.seconds)
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
    """Pick the attempt to keep, and say what the ladder came to.

    The best attempt, not the last: an escalation can retrieve worse than what
    it replaced, and the reader should get the strongest section found.

    Shared by both supervisors rather than written twice. They are meant to
    differ in control flow and in nothing else, so any selection rule living
    in one of them would make a comparison between them meaningless.
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
    """One embedder shared by worker threads, one call at a time.

    ONNX inference is cheap next to generation, so serialising it costs
    nothing measurable and removes a class of race the threads would
    otherwise have on first load.
    """

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
    """The function that climbs one section's ladder.

    Imported lazily, because the LangGraph one needs an optional dependency
    and a default install does not have it - naming a supervisor nobody
    installed should say so, not fail at import time for everybody else.
    """
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
    """Plan a review, then work its sections - in parallel when asked.

    Sections share a topic and nothing else, which is the division of labour
    the literature says parallel workers are for. Each worker thread opens
    its own connection to the corpus, because SQLite connections belong to
    the thread that made them, and shares one serialised embedder.

    `on_section_done` receives each finished section and its ledger entry as
    it lands, which is what lets a review be streamed rather than waited for.

    `supervisor` chooses which implementation climbs the ladder: the built-in
    loop, or the same ladder as a LangGraph state machine. They share every
    role and the selection rule and are meant to differ only in control flow;
    the point of keeping both is that two implementations behind one set of
    contracts is what shows the contracts are a boundary.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    climb = _supervisor(supervisor)
    headings = plan_sections(complete_text, topic, max_sections=max_sections)
    shared = _Serialized(embedder) if workers > 1 else embedder
    guard = (
        BudgetGuard(budget, getattr(completion, "usage", None))
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
            # Fired as each section lands, so a caller streaming a review has
            # something to show before the whole thing is finished. With
            # several workers this arrives from a worker thread, so anything
            # the caller does here must be safe to call from one.
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
