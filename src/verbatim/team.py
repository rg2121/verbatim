"""The `team` command: workers around a shared evidence board.

A model plans the tasks. After that, code reads the board for what is missing
(no evidence, too little, no adversarial search yet) and sends the matching
worker, so one worker's findings decide the next dispatch. A worker is never
sent twice at the same task; a task whose workers are exhausted is abandoned.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re

from verbatim import config
from verbatim.answers import Answer, Completion, answer_from_matches
from verbatim.board import (
    ABANDONED,
    ADVERSARIAL,
    DIRECT,
    SATISFIED,
    Missing,
    ResearchState,
    Task,
)
from verbatim.corpus import ChunkMatch, CorpusStore
from verbatim.embedding import Embedder
from verbatim.retrieval import hybrid_search

RETRIEVAL = "retrieval"
ADVERSARIAL_SEARCH = "adversarial-search"

NO_EVIDENCE = "no evidence"
THIN = "thinner than asked"
NO_ADVERSARIAL = "no adversarial search run"

# Recorded as a decision, but no worker runs.
ABANDON = "abandon"

PLANNER_SYSTEM = """You are given a research question. Break it into the things that must be established before it can be answered.

Rules:
- Two to four tasks. Fewer is better; a question needing one thing gets one task.
- Each task is something a passage from a paper could support or undermine.
- Name subjects concretely. Someone must be able to search for a task using only
  its own words.
- Do not write tasks about writing, summarising, or checking. Those are not
  evidence.

Write one per line, in this exact form, and nothing else:

TASK: <objective>"""

TASK_LINE = re.compile(r"^\s*TASK:\s*(.+?)\s*$", re.MULTILINE)

ADVERSARIAL_SYSTEM = """You are given a research claim that other people are trying to support.

Write search queries that would surface its limitations: failure cases, negative results, criticism, papers reporting it does not work, or conditions under which it breaks.

Rules:
- Write queries, not questions. Use the words such a paper would use.
- Do not write queries that would find support for the claim.
- Vary the vocabulary; a critic and an advocate use different words.

Write one per line, in this exact form, and nothing else:

QUERY: <query>"""

QUERY_LINE = re.compile(r"^\s*QUERY:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class TeamReport:
    """What the team established, and what the board made it do."""

    state: ResearchState
    answers: dict[str, Answer]
    final: Answer | None = None

    @property
    def redirections(self) -> int:
        """Dispatches the board caused after every task had been tried once.

        Zero means the board recorded work rather than directing it. Abandoning
        a task dispatches nothing and is not counted.
        """
        seen: set[str] = set()
        extra = 0
        for decision in self.state.decisions:
            if decision.action == ABANDON:
                continue
            if decision.task_id in seen or decision.action != RETRIEVAL:
                extra += 1
            seen.add(decision.task_id)
        return extra

    @property
    def citations_from_adversarial(self) -> tuple[int, int]:
        """Citations resting on a passage only the adversarial search found, and the total."""
        adversarial_chunks = {
            e.chunk_id for e in self.state.evidence if e.stance == ADVERSARIAL
        }
        cited = [
            c.chunk_id for answer in self.answers.values() for c in answer.citations
        ]
        return sum(1 for chunk_id in cited if chunk_id in adversarial_chunks), len(cited)

    def render(self) -> str:
        lines = [self.state.render(), ""]
        lines.append(
            f"  the board redirected work {self.redirections} time(s)"
        )
        from_adversarial, total = self.citations_from_adversarial
        lines.append(
            f"  {from_adversarial} of {total} citation(s) rest on a passage only "
            "the adversarial search found"
        )
        if self.final is not None:
            lines.append("")
            lines.append(f"  {self.state.question}")
            lines.append(f"    {self.final.text}")
        for task in self.state.tasks:
            answer = self.answers.get(task.task_id)
            if answer is None:
                continue
            lines.append("")
            lines.append(f"  {task.objective}")
            lines.append(f"    {answer.text}")
        return "\n".join(lines)


def plan_tasks(
    complete: Callable[[str, str], str], question: str, *, max_tasks: int = 4
) -> list[str]:
    reply = complete(PLANNER_SYSTEM, f"QUESTION: {question}")
    found = [m.group(1) for m in TASK_LINE.finditer(reply or "")]
    kept = [t for t in dict.fromkeys(found) if len(t.split()) >= 4]
    return kept[:max_tasks] or [question]


def inspect(state: ResearchState) -> list[Missing]:
    """What the board can see is absent. Arithmetic, in a fixed order."""
    missing: list[Missing] = []
    for task in state.open_tasks():
        direct = state.evidence_for(task.task_id, stance=DIRECT)
        if not direct:
            missing.append(Missing(task.task_id, NO_EVIDENCE, "nothing filed yet"))
            continue
        if len(direct) < task.needs:
            missing.append(Missing(
                task.task_id, THIN, f"{len(direct)} of {task.needs} asked for"
            ))
            continue
        if not state.attempted(task.task_id, ADVERSARIAL_SEARCH):
            # The attempt closes this gap, even if it found nothing new: the
            # board dedupes by passage, so rows alone could never close it.
            missing.append(Missing(
                task.task_id, NO_ADVERSARIAL, "an adversarial search has not run"
            ))
    return missing


def worker_for(kind: str) -> str:
    """Which worker a structural gap implies. A rule, not a judgement."""
    return ADVERSARIAL_SEARCH if kind == NO_ADVERSARIAL else RETRIEVAL


def retrieval_worker(
    corpus: CorpusStore,
    embedder: Embedder,
    state: ResearchState,
    task: Task,
    *,
    limit: int = 6,
) -> int:
    """Search for passages that bear on the task, and file them."""
    filed = 0
    for match in hybrid_search(
        corpus, embedder, task.objective, limit=limit,
        max_per_paper=config.ASK_MAX_PER_PAPER, use_graph=True,
    ):
        if state.add_evidence(
            task.task_id, chunk_id=match.chunk.chunk_id,
            arxiv_id=match.chunk.arxiv_id, quote=match.chunk.text[:400],
            found_by=RETRIEVAL, stance=DIRECT,
        ):
            filed += 1
    return filed


def adversarial_worker(
    corpus: CorpusStore,
    embedder: Embedder,
    complete: Callable[[str, str], str],
    state: ResearchState,
    task: Task,
    *,
    queries: int = 3,
    limit: int = 4,
) -> int:
    """Search for what would undermine the task, and file it as adversarial.

    The model only writes the queries; it never judges whether a passage
    contradicts the claim.
    """
    reply = complete(ADVERSARIAL_SYSTEM, f"CLAIM: {task.objective}")
    searches = list(dict.fromkeys(
        m.group(1) for m in QUERY_LINE.finditer(reply or "")
    ))[:queries] or [f"limitations of {task.objective}"]
    filed = 0
    for query in searches:
        for match in hybrid_search(
            corpus, embedder, query, limit=limit, max_per_paper=2, use_graph=True,
        ):
            if state.add_evidence(
                task.task_id, chunk_id=match.chunk.chunk_id,
                arxiv_id=match.chunk.arxiv_id, quote=match.chunk.text[:400],
                found_by=ADVERSARIAL_SEARCH, stance=ADVERSARIAL,
            ):
                filed += 1
    return filed


def run_team(
    corpus: CorpusStore,
    embedder: Embedder,
    completion: Completion,
    complete: Callable[[str, str], str],
    question: str,
    *,
    max_tasks: int = 4,
    max_steps: int = 12,
    passages: int = 6,
    needs: int = 2,
) -> TeamReport:
    """Plan, then let the board decide what happens next until nothing is missing."""
    state = ResearchState(question=question)
    for objective in plan_tasks(complete, question, max_tasks=max_tasks):
        state.add_task(objective, needs=needs)

    for _step in range(max_steps):
        missing = inspect(state)
        if not missing:
            break
        gap = missing[0]
        worker = worker_for(gap.kind)
        task = next(t for t in state.tasks if t.task_id == gap.task_id)
        if state.attempted(gap.task_id, worker):
            # This worker already ran on this task; retrying it would be blind.
            state.record(ABANDON, gap.task_id, f"{gap.kind}; {worker} already ran")
            state.close(gap.task_id, ABANDONED)
            continue
        state.record(worker, gap.task_id, f"{gap.kind}: {gap.detail}")
        if worker == RETRIEVAL:
            retrieval_worker(corpus, embedder, state, task, limit=passages)
        else:
            adversarial_worker(corpus, embedder, complete, state, task)
        if not inspect_task(state, task):
            state.close(gap.task_id, SATISFIED)

    answers = write_answers(corpus, completion, state)
    final = write_final(corpus, completion, state, answers)
    return TeamReport(state=state, answers=answers, final=final)


def write_final(
    corpus: CorpusStore,
    completion: Completion,
    state: ResearchState,
    answers: dict[str, Answer],
    *,
    limit: int = 12,
) -> Answer:
    """One answer to the original question, from passages a task answer already cited.

    Only cited passages are offered, so every one already passed verification.
    When there are none, the answer is an explicit decline.
    """
    approved = {
        citation.chunk_id for answer in answers.values() for citation in answer.citations
    }
    on_board = [e for e in state.evidence if e.chunk_id in approved]
    chunks = corpus.chunks_by_id(
        list(dict.fromkeys(e.chunk_id for e in on_board))[:limit]
    )
    if not chunks:
        return Answer(
            question=state.question,
            text=(
                "The team found no passage it could cite for this question. "
                + why_nothing_was_cited(state)
            ),
            citations=[], rejected=[], declined=True,
        )
    matches = [
        ChunkMatch(chunk=chunk, score=1.0, snippet=chunk.text[:160])
        for chunk in chunks
    ]
    return answer_from_matches(corpus, completion, state.question, matches)


def why_nothing_was_cited(state: ResearchState) -> str:
    """Why no passage was cited, distinguishing unfinished tasks from empty ones."""
    unfinished = state.open_tasks()
    finished = len(state.tasks) - len(unfinished)
    if not state.decisions:
        return "No task was worked before the team stopped."
    if unfinished and not finished:
        return f"It stopped before finishing any of its {len(state.tasks)} task(s)."
    if unfinished:
        return (
            f"{len(unfinished)} of {len(state.tasks)} task(s) were not finished "
            "before the team stopped, and the finished ones produced no verified "
            "citation."
        )
    return "Every task it set came back without a verified citation."


def inspect_task(state: ResearchState, task: Task) -> list[Missing]:
    return [m for m in inspect(state) if m.task_id == task.task_id]


def write_answers(
    corpus: CorpusStore, completion: Completion, state: ResearchState
) -> dict[str, Answer]:
    """Answer each task only from the evidence filed for it on the board."""
    answers: dict[str, Answer] = {}
    for task in state.tasks:
        rows = state.evidence_for(task.task_id)
        if not rows:
            continue
        chunks = corpus.chunks_by_id([r.chunk_id for r in rows])
        matches = [
            ChunkMatch(chunk=chunk, score=1.0, snippet=chunk.text[:160])
            for chunk in chunks
        ]
        if not matches:
            continue
        answer = answer_from_matches(corpus, completion, task.objective, matches)
        answers[task.task_id] = answer
        by_chunk = {r.chunk_id: r.evidence_id for r in rows}
        for citation in answer.citations:
            state.add_claim(
                task.task_id, answer.text[:200],
                [by_chunk[citation.chunk_id]]
                if citation.chunk_id in by_chunk else [],
            )
    return answers
