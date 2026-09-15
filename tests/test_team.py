"""Workers that redirect each other through a shared board."""

from __future__ import annotations

from pathlib import Path


from verbatim.board import ADVERSARIAL, ResearchState
from verbatim.team import (
    ADVERSARIAL_SEARCH,
    NO_ADVERSARIAL,
    NO_EVIDENCE,
    RETRIEVAL,
    THIN,
    inspect,
    run_team,
    worker_for,
    write_answers,
)

from test_orchestrator import StubEmbedder, cite, flux_corpus


def board_with(needs: int = 2) -> tuple[ResearchState, str]:
    state = ResearchState(question="does flux improve charge?")
    task = state.add_task("establish that flux improves charge", needs=needs)
    return state, task.task_id


def test_the_board_refuses_a_claim_resting_on_evidence_nobody_filed() -> None:
    """The one place the board enforces rather than records.

    A claim citing an evidence id that is not on the board cannot be traced to
    anything, which defeats the point of having a board at all.
    """
    state, task = board_with()
    row = state.add_evidence(
        task, chunk_id="a", arxiv_id="p1", quote="q", found_by=RETRIEVAL
    )

    assert state.add_claim(task, "flux improves charge", ["e404"]) is None
    assert state.add_claim(task, "flux improves charge", []) is None
    assert state.add_claim(task, "flux improves charge", [row.evidence_id]) is not None


def test_two_workers_finding_the_same_passage_file_it_once() -> None:
    """A task's evidence count decides whether it is done.

    Counting one passage twice would close a task nobody finished, and two
    workers searching the same objective will collide.
    """
    state, task = board_with()

    first = state.add_evidence(
        task, chunk_id="a", arxiv_id="p1", quote="q", found_by=RETRIEVAL
    )
    again = state.add_evidence(
        task, chunk_id="a", arxiv_id="p1", quote="q", found_by=ADVERSARIAL_SEARCH,
        stance=ADVERSARIAL,
    )

    assert first is not None and again is None
    assert len(state.evidence_for(task)) == 1


def test_the_critic_reports_absence_in_a_fixed_order() -> None:
    """Structural gaps only, and the same board always gives the same answer."""
    state, task = board_with(needs=2)

    assert inspect(state)[0].kind == NO_EVIDENCE
    state.add_evidence(task, chunk_id="a", arxiv_id="p1", quote="q", found_by=RETRIEVAL)
    assert inspect(state)[0].kind == THIN
    state.add_evidence(task, chunk_id="b", arxiv_id="p2", quote="q", found_by=RETRIEVAL)
    assert inspect(state)[0].kind == NO_ADVERSARIAL
    # Closed by the attempt, not by what it found: an adversarial search that
    # turns up only passages already filed has done its job and come back
    # empty, and the board dedupes by passage.
    state.record(ADVERSARIAL_SEARCH, task, "sought")
    assert inspect(state) == []


def test_a_single_paper_is_not_reported_as_a_gap() -> None:
    """Deliberately absent, on this project's own measurement.

    A lone-paper warning was measured firing on 5 of 14 good answers while
    catching 2 of 6 bad ones. It reads like a quality signal and is not one,
    so the board does not raise it and no worker is dispatched for it.
    """
    state, task = board_with(needs=2)
    for chunk in ("a", "b"):
        state.add_evidence(
            task, chunk_id=chunk, arxiv_id="p1", quote="q", found_by=RETRIEVAL
        )
    state.record(ADVERSARIAL_SEARCH, task, "sought")

    assert state.papers_for(task) == {"p1"}   # everything from one paper
    assert inspect(state) == []               # and the board says nothing


def test_a_gap_chooses_its_worker_by_rule() -> None:
    assert worker_for(NO_EVIDENCE) == RETRIEVAL
    assert worker_for(THIN) == RETRIEVAL
    assert worker_for(NO_ADVERSARIAL) == ADVERSARIAL_SEARCH


def test_a_worker_is_never_sent_twice_at_the_same_task(tmp_path: Path) -> None:
    """Blind retry reproduces the fault that caused the first failure.

    A retrieval that files nothing leaves the task with no evidence, so the
    critic raises the same gap for ever. The loop has to abandon the task
    rather than run the same worker again.
    """
    class NoAnswers:
        def complete(self, system, user, schema):
            return {"answer": "The passages do not say.", "citations": []}

        def judge(self, system, user):
            return "flux"

    def plans(system, user):
        if "QUERY" in system:
            return "QUERY: flux limitations\n"
        return "TASK: establish how magnetic flux threads a ring\n"

    with flux_corpus(tmp_path) as corpus:
        # More evidence demanded than a two-paper corpus can supply, so the
        # task stays thin however often retrieval runs. Retrieval finding
        # nothing at all is unreachable here: vector search always returns
        # its nearest neighbours, however far away they are.
        report = run_team(
            corpus, StubEmbedder(), NoAnswers(), plans,
            "how does flux thread a ring?", max_steps=6, needs=99,
        )

    actions = [d.action for d in report.state.decisions]
    # It tried, then abandoned. It did not try the same worker twice.
    assert actions.count(RETRIEVAL) <= 1
    assert "abandon" in actions
    assert report.state.tasks[0].status == "abandoned"


def test_the_board_redirects_work_and_says_how_often(tmp_path: Path) -> None:
    """The number that says whether the board is doing anything.

    A run where every task is satisfied by its first retrieval has a board
    that recorded work rather than directing it.
    """
    class Answers:
        def complete(self, system, user, schema):
            return {"answer": "Flux threads it.", "citations": []}

        def judge(self, system, user):
            return "flux and charge"

    def plans(system, user):
        if "QUERY" in system:
            return "QUERY: flux limitations\nQUERY: charge failure\n"
        return "TASK: establish how magnetic flux threads a ring\n"

    with flux_corpus(tmp_path) as corpus:
        report = run_team(
            corpus, StubEmbedder(), Answers(), plans,
            "how does flux thread a ring?", max_steps=6,
        )

    actions = [d.action for d in report.state.decisions]
    assert RETRIEVAL in actions
    # Retrieval satisfied the evidence count, and the board then sent a
    # different worker for what was still missing - that is the redirection.
    assert ADVERSARIAL_SEARCH in actions
    assert report.redirections >= 1
    # The task closed because the adversarial search ran, not because it
    # happened to find something the supporting search had missed.
    assert report.state.tasks[0].status == "satisfied"


def test_the_writer_may_use_only_what_is_on_the_board(tmp_path: Path) -> None:
    """The synthesis role cannot reach for a passage no worker filed."""
    with flux_corpus(tmp_path) as corpus:
        chunk = next(c for c in corpus.all_chunks() if c.arxiv_id == "2608.11111v1")
        other = next(c for c in corpus.all_chunks() if c.arxiv_id != "2608.11111v1")

        seen: dict[str, str] = {}

        class Records:
            def complete(self, system, user, schema):
                seen["prompt"] = user
                return cite(chunk.chunk_id, "Magnetic flux threads the ring.")

            def judge(self, system, user):
                return "flux"

        state = ResearchState(question="q")
        task = state.add_task("how flux threads a ring")
        state.add_evidence(
            task.task_id, chunk_id=chunk.chunk_id, arxiv_id=chunk.arxiv_id,
            quote=chunk.text, found_by=RETRIEVAL,
        )
        answers = write_answers(corpus, Records(), state)

    assert task.task_id in answers
    assert chunk.chunk_id in seen["prompt"]
    # The passage nobody filed never reached the writer.
    assert other.chunk_id not in seen["prompt"]
    # And the claim it produced is traceable to the board row it rests on.
    assert state.claims and state.claims[0].evidence_ids


def test_the_report_says_whether_the_adversarial_search_was_used(
    tmp_path: Path,
) -> None:
    """A worker that files rows nothing cites has not earned its cost.

    The adversarial worker can fill the board and change nothing. What
    matters is whether the written answer rests on any of what it found, so
    that is counted rather than assumed.
    """
    from verbatim.team import TeamReport

    with flux_corpus(tmp_path) as corpus:
        chunk = next(c for c in corpus.all_chunks() if c.arxiv_id == "2608.11111v1")

        class CitesTheAdversarialRow:
            def complete(self, system, user, schema):
                return cite(chunk.chunk_id, "Magnetic flux threads the ring.")

            def judge(self, system, user):
                return "flux"

        state = ResearchState(question="q")
        task = state.add_task("how flux threads a ring")
        # Filed by the adversarial search, not the supporting one.
        state.add_evidence(
            task.task_id, chunk_id=chunk.chunk_id, arxiv_id=chunk.arxiv_id,
            quote=chunk.text, found_by=ADVERSARIAL_SEARCH, stance=ADVERSARIAL,
        )
        answers = write_answers(corpus, CitesTheAdversarialRow(), state)
        report = TeamReport(state=state, answers=answers)

    used, total = report.citations_from_adversarial
    assert total == 1 and used == 1
    assert "1 of 1 citation(s) rest on a passage only the adversarial search found" \
        in report.render()


def test_the_final_answer_uses_only_evidence_a_task_answer_cited(
    tmp_path: Path,
) -> None:
    """"Approved" means it already carried a quote through verification.

    A passage a worker filed and no task answer ever cited survived retrieval
    and nothing else. It must not reach the final text, or the board's
    provenance stops meaning anything at the last step.
    """
    from verbatim.team import write_final

    with flux_corpus(tmp_path) as corpus:
        cited = next(c for c in corpus.all_chunks() if c.arxiv_id == "2608.11111v1")
        filed_only = next(c for c in corpus.all_chunks() if c.chunk_id != cited.chunk_id)

        seen: dict[str, str] = {}

        class Writes:
            def complete(self, system, user, schema):
                seen["prompt"] = user
                return cite(cited.chunk_id, "Magnetic flux threads the ring.")

            def judge(self, system, user):
                return "flux"

        state = ResearchState(question="how does flux thread a ring?")
        task = state.add_task("how flux threads a ring")
        for chunk in (cited, filed_only):
            state.add_evidence(
                task.task_id, chunk_id=chunk.chunk_id, arxiv_id=chunk.arxiv_id,
                quote=chunk.text, found_by=RETRIEVAL,
            )
        answers = write_answers(corpus, Writes(), state)
        final = write_final(corpus, Writes(), state, answers)

    assert final is not None
    # The task answer cited one chunk; only that one reached the final prompt.
    assert cited.chunk_id in seen["prompt"]
    assert filed_only.chunk_id not in seen["prompt"]
    assert final.question == "how does flux thread a ring?"


def test_nothing_approved_declines_in_words_rather_than_returning_nothing(tmp_path: Path) -> None:
    """A board whose task answers cited nothing has nothing to synthesise."""
    from verbatim.team import write_final

    class CitesNothing:
        def complete(self, system, user, schema):
            return {"answer": "The passages do not say.", "citations": []}

        def judge(self, system, user):
            return "flux"

    with flux_corpus(tmp_path) as corpus:
        chunk = corpus.all_chunks()[0]
        state = ResearchState(question="q")
        task = state.add_task("something")
        state.add_evidence(
            task.task_id, chunk_id=chunk.chunk_id, arxiv_id=chunk.arxiv_id,
            quote=chunk.text, found_by=RETRIEVAL,
        )
        answers = write_answers(corpus, CitesNothing(), state)

        final = write_final(corpus, CitesNothing(), state, answers)

    # Declined in words, never nothing: the reader always gets an answer to the
    # question they asked, and the approved-only rule still holds, because no
    # passage reached it.
    assert final is not None
    assert final.declined is True
    assert final.citations == []
    assert "no passage it could cite" in final.text


def test_a_decline_does_not_claim_work_the_step_limit_prevented(
    tmp_path: Path,
) -> None:
    """Found in audit: the decline described work that never happened.

    With the step limit at zero no worker is ever dispatched, yet the answer
    said every task came back without a verified citation. It must say what
    actually happened.
    """
    class Silent:
        def complete(self, system, user, schema):
            return {"answer": "The passages do not say.", "citations": []}

        def judge(self, system, user):
            return "flux"

    def plans(system, user):
        return (
            "TASK: establish how magnetic flux threads a ring\n"
            "TASK: establish how charge moves along a wire\n"
        )

    with flux_corpus(tmp_path) as corpus:
        report = run_team(
            corpus, StubEmbedder(), Silent(), plans,
            "how does flux thread a ring?", max_steps=0,
        )

    assert report.final is not None and report.final.declined
    assert "Every task it set came back" not in report.final.text
    assert "No task was worked before the team stopped." in report.final.text
    assert all(t.status == "open" for t in report.state.tasks)


def test_abandoning_a_task_is_not_counted_as_a_redirection() -> None:
    """No worker runs when a task is abandoned, so nothing was redirected.

    It was counted, which inflated the number on exactly the runs that gave up.
    """
    from verbatim.team import ABANDON, TeamReport

    state, task = board_with()
    state.record(RETRIEVAL, task, "no evidence: nothing filed yet")
    state.record(ABANDON, task, "thinner than asked; retrieval already ran")

    assert TeamReport(state=state, answers={}).redirections == 0

    state.record(ADVERSARIAL_SEARCH, task, "no adversarial search run")
    assert TeamReport(state=state, answers={}).redirections == 1
