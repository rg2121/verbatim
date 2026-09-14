"""Workers that redirect each other through the board, and code that delegates.

The ladder retries a fixed sequence of strategies. This does something
different: workers append to a shared board, code reads the board for what is
structurally missing, and *that* decides which worker runs next. A worker's
finding therefore changes another worker's next action, which is the line
between a workflow and a team.

**The coordinator plans with a model and delegates without one.** Turning a
question into tasks is writing, which models do well. Choosing what to do next
is routing, and routing by model was measured here to lose to routing by rule -
so the choice is made by reading the board. That also makes the loop
reproducible: the same board always produces the same next action.

**The critic is arithmetic.** It reports tasks with no evidence, tasks thinner
than they asked to be, and tasks where counter-evidence was never sought. It
does not report that evidence is weak, or that a claim rests on one paper -
that second one was measured here firing on 5 of 14 good answers while catching
2 of 6 bad ones, so it is not a gap worth acting on and is not raised.

**A worker is never sent twice at the same task.** Blind retry reproduces the
fault that caused the first failure; every dispatch here is a worker that has
not yet been tried on that task, and a task whose workers are exhausted is
abandoned rather than reattempted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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

        The number that says whether the board is doing anything. A run where
        every task is satisfied by its first retrieval has a board that
        recorded work rather than directing it, and this reports zero.
        """
        first_pass = {t.task_id for t in self.state.tasks}
        seen: set[str] = set()
        extra = 0
        for decision in self.state.decisions:
            if decision.task_id in seen or decision.action != RETRIEVAL:
                extra += 1
            seen.add(decision.task_id)
        return extra if first_pass else 0

    @property
    def citations_from_adversarial(self) -> tuple[int, int]:
        """Citations resting on a passage an adversarial search found, of the total.

        The number that says whether the adversarial worker earns its
        cost. It can file as many rows as it likes; what matters is whether
        the written answer rests on any of them. Zero means the board paid for
        a search whose results nothing used.
        """
        counter = {
            e.chunk_id for e in self.state.evidence if e.stance == ADVERSARIAL
        }
        cited = [
            c.chunk_id for answer in self.answers.values() for c in answer.citations
        ]
        return sum(1 for chunk_id in cited if chunk_id in counter), len(cited)

    def render(self) -> str:
        lines = [self.state.render(), ""]
        lines.append(
            f"  the board redirected work {self.redirections} time(s)"
        )
        from_counter, total = self.citations_from_adversarial
        lines.append(
            f"  {from_counter} of {total} citation(s) rest on a passage only "
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
            # Sought, not found. An adversarial search that turns up only the
            # passages the supporting search already filed has done its job and
            # come back empty, and the board dedupes by passage - so asking
            # whether counter rows exist would send the worker out for ever on
            # a task where there is genuinely nothing against. The gap is
            # closed by the attempt, which is what its name says.
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
    """Search for what would undermine the task, and file it as counter.

    It files what it finds; it does not decide that a passage contradicts
    anything. Asking a model whether evidence undermines a claim is the
    judgement this project has measured failing, so the model's job here ends
    at writing adversarial queries - which is retrieval, and it is good at it.
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
            # Every worker this gap implies has already run on this task.
            # Retrying the same one is the blind retry the ledger forbids.
            state.record("abandon", gap.task_id, f"{gap.kind}; {worker} already ran")
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
    """One answer to the question that was asked, from approved evidence only.

    The per-task answers are working notes: they answer what the coordinator
    decided to establish, not what the reader asked. This writes the answer to
    the original question.

    "Approved" is doing real work in that sentence. The passages offered here
    are only those a per-task answer *cited* - which means each one already
    carried a quote through verification. Evidence a worker filed and no task
    answer used is left out, so a passage that survived retrieval but nothing
    else cannot reach the final text.

    When nothing was approved, the answer is a declined one in words - never
    nothing. The first version returned None there, and the three-arm run
    caught it: on one question in thirty the team handed the reader working
    notes and no answer, where both simpler arms always answered, even if only
    to say nothing matched. Falling back to unapproved evidence would have
    filled the gap by breaking the rule above, so the fix is to decline.
    """
    approved = {
        citation.chunk_id for answer in answers.values() for citation in answer.citations
    }
    on_board = [e for e in state.evidence if e.chunk_id in approved]
    chunks = corpus.chunks_by_id(
        list(dict.fromkeys(e.chunk_id for e in on_board))[:limit]
    ) if on_board else []
    if not chunks:
        return Answer(
            question=state.question,
            text=(
                "The team found no passage it could cite for this question. "
                "Every task it set came back without a verified citation."
            ),
            citations=[], rejected=[], declined=True,
        )
    matches = [
        ChunkMatch(chunk=chunk, score=1.0, snippet=chunk.text[:160])
        for chunk in chunks
    ]
    return answer_from_matches(corpus, completion, state.question, matches)


def inspect_task(state: ResearchState, task: Task) -> list[Missing]:
    return [m for m in inspect(state) if m.task_id == task.task_id]


def write_answers(
    corpus: CorpusStore, completion: Completion, state: ResearchState
) -> dict[str, Answer]:
    """Answer each task from the evidence on the board and nothing else.

    The writer is handed the board's rows rebuilt as passages, so it cannot
    reach for anything a worker did not file, and every quote still goes
    through the same verification as the rest of the project.
    """
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
