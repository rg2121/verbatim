"""M1: the LangGraph runtime v2 stands on, proved on a scripted report-shaped slice.

Each test pins one property the plan's design rests on (graph_slice.py holds the
slice). If one fails, the design changes before any agent is built.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, replace
import multiprocessing
import operator
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
import pytest

import graph_slice
from graph_slice import BUDGET_EXHAUSTED, COMPLETE, Deps, FakeClock, Script, ScriptedModel, build
from verbatim.agents.runtime import (
    BUSY_TIMEOUT,
    Deadline,
    checkpoints,
    refuse_used_thread,
    require_checkpointable,
    serializer,
    shares,
    time_used,
    v1_events,
)
from verbatim.agents.state import (
    Allowance,
    Evidence,
    Found,
    Record,
    SectionResult,
    Step,
    ledger_append,
    spent,
)
from verbatim.contracts import Budget, Cost

SLICE = Path(graph_slice.__file__)
# editor 1 + 3 sections x (2 searches + 1 draft) + review 1
CALLS = 11
SECTION = ["researcher", "researcher", "writer"]


def run(deps: Deps, *, saver: Any = None, thread: str = "t", topic: str = "topic") -> dict:
    config = {"configurable": {"thread_id": thread}}
    if saver is None:  # LangGraph 1.2.11 fails on `durability` when there is no checkpointer
        return build().invoke({"topic": topic}, config, context=deps)
    refuse_used_thread(saver, config)
    return build(saver).invoke({"topic": topic}, config, context=deps, durability="sync")


def section_steps(ledger: list[Step], index: int) -> list[Step]:
    return [step for step in ledger if step.key[1:3] == ("section_team", index)]


def without_seconds(state: dict) -> dict:
    """I9 compares the ledger without its times, which no two runs share."""
    return {**state, "ledger": [replace(s, cost=replace(s.cost, seconds=0.0))
                                for s in state["ledger"]]}


def wait_for(condition: Callable[[], bool], why: Callable[[], str], seconds: float = 60.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    pytest.fail(f"timed out: {why()}")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def child(*args: str, err: Path) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, str(SLICE), *args], stderr=err.open("w"))


# -- wrappers, Send and Command ---------------------------------------------------------


def test_each_section_runs_its_own_graph_behind_a_wrapper_and_returns_one_result() -> None:
    out = run(Deps(model=ScriptedModel()))
    assert [(s.index, s.heading, s.status) for s in out["sections"]] == [
        (0, "Background", COMPLETE), (1, "Methods", COMPLETE), (2, "Results", COMPLETE)]
    assert out["status"] == COMPLETE and out["result"].startswith("Background, round 0: ")
    assert len({step.key for step in out["ledger"]}) == len(out["ledger"]) == CALLS
    # Every section found the shared passage: the board holds it once, with every finder.
    shared = [e for e in out["board"] if e.chunk_id == graph_slice.SHARED_CHUNK]
    assert len(shared) == 1 and len(shared[0].found) == 6
    assert len(out["board"]) == 7


def test_parallel_branches_sharing_one_state_fail_as_expected() -> None:
    # The naive version: every branch runs on the run's own state and returns it.
    class Shared(TypedDict, total=False):
        status: str

    def section(state: Shared) -> dict:
        return {"status": COMPLETE}

    naive = StateGraph(Shared)
    naive.add_node("section", section)
    naive.add_conditional_edges(START, lambda state: [Send("section", state)] * 3, ["section"])
    naive.add_edge("section", END)
    with pytest.raises(InvalidUpdateError, match="Can receive only one value per step"):
        naive.compile().invoke({})


@pytest.mark.parametrize(("reducer", "calls"), [(operator.add, 7), (ledger_append, 4)])
def test_a_branch_that_returns_its_copy_of_the_ledger_counts_it_twice(reducer, calls) -> None:
    # A branch given the run's ledger hands all of it back, not just its own step.
    # Added naively the editor's step is counted four times; keyed, once.
    # Built here, not as a class: the reducer is a local this module's annotations cannot see.
    Shared = TypedDict("Shared", {"ledger": Annotated[list[Step], reducer]}, total=False)

    def step(path: str, branch: int) -> Step:
        return Step(key=(0, path, branch, 0), kind="model", name=path, cost=Cost(calls=1))

    def section(copy: dict) -> dict:
        return {"ledger": [*copy["ledger"], step("section", copy["index"])]}

    graph = StateGraph(Shared)
    graph.add_node("editor", lambda state: {"ledger": [step("editor", -1)]})
    graph.add_node("section", section)
    graph.add_edge(START, "editor")
    graph.add_conditional_edges(
        "editor", lambda state: [Send("section", {"ledger": state["ledger"], "index": i})
                                 for i in range(3)], ["section"])
    graph.add_edge("section", END)
    assert spent(graph.compile().invoke({})["ledger"]).calls == calls


def test_sections_are_joined_in_plan_order_whatever_order_they_were_started() -> None:
    # LangGraph applies a step's writes in the order its Sends were made, never the
    # order branches finish; keys are what put the plan's order back.
    out = run(Deps(model=ScriptedModel(), script=Script(order=(2, 0, 1))))
    assert [s.index for s in out["sections"]] == [0, 1, 2]
    assert out["result"].startswith("Background, round 0: ")


def test_the_ledger_takes_a_step_once_and_refuses_two_under_one_key() -> None:
    first = Step(key=(0, "editor", -1, 0), kind="model", name="editor", cost=Cost(calls=1))
    assert ledger_append([first], [first]) == [first]
    with pytest.raises(ValueError, match="two different ledger steps"):
        ledger_append([first], [replace(first, name="writer")])


def test_the_ledger_reads_in_the_order_the_run_went() -> None:
    # Keys start with LangGraph's step numbers, not node names: the review comes after
    # the work it reviewed, a section's searches before its draft, a second round last.
    out = run(Deps(model=ScriptedModel(), script=Script(send_back=(1,))))
    assert [s.name for s in out["ledger"]] == [
        "editor", *SECTION * 3, "review", *SECTION, "review"]
    assert [s.key for s in out["ledger"] if s.name == "review"] == [
        (3, "review", -1, 0), (5, "review", -1, 0)]
    assert out["ledger"][1].key == (2, "section_team", 0, 1, "research", -1, 0)


def test_a_command_sends_a_section_back_and_the_report_shows_its_second_round() -> None:
    out = run(Deps(model=ScriptedModel(), script=Script(send_back=(1,))))
    # Both rounds are kept; which one the report shows is decided after the join.
    assert [(s.index, s.round, s.status) for s in out["sections"]] == [
        (0, 0, COMPLETE), (1, 0, COMPLETE), (1, 1, COMPLETE), (2, 0, COMPLETE)]
    assert out["result"].split("\n\n")[1].startswith("Methods, round 1: ")
    assert {step.key[0] for step in section_steps(out["ledger"], 1)} == {2, 4}


def test_a_new_run_on_a_thread_that_holds_one_is_refused(tmp_path) -> None:
    db, work = tmp_path / "checkpoints.db", tmp_path / "work"
    assert graph_slice.main(["start", str(db), "job", str(work)]) == 0
    with pytest.raises(SystemExit, match="already holds a run"):
        graph_slice.main(["start", str(db), "job", str(work)])
    # Run anyway, the thread stays readable, since keys keep counting, but the new run
    # inherits the old one's ledger, and with it the old run's spend.
    config = {"configurable": {"thread_id": "job"}}
    with checkpoints(db) as saver:
        graph = build(saver)
        again = graph.invoke({"topic": "again"}, config, context=Deps(model=ScriptedModel()),
                             durability="sync")
        assert len(again["ledger"]) == 2 * CALLS
        assert graph.get_state(config).values["topic"] == "again"


# -- reservations and the deadline ------------------------------------------------------


def test_shares_split_what_is_left_evenly_and_never_below_zero() -> None:
    ledger = [Step(key=(0, "editor", -1, 0), kind="model", name="editor",
                   cost=Cost(calls=1, tokens_in=10, tokens_out=5))]
    budget = Budget(max_calls=11, max_tokens=45)
    assert shares(budget, ledger, 3) == (Allowance(4, 10), Allowance(3, 10), Allowance(3, 10))
    assert shares(budget, ledger, 3, keep_calls=1, keep_tokens=15) == (Allowance(3, 5),) * 3
    assert shares(Budget(), ledger, 2) == (Allowance(None, None),) * 2
    assert shares(Budget(max_calls=1), ledger, 2) == (Allowance(0, None),) * 2
    assert shares(budget, ledger, 0) == ()  # no branches, no shares


def test_unused_shares_come_back_at_the_join() -> None:
    # 15 calls: the editor's 1 and the 1 kept for the review leave 5, 4, 4; each
    # section spends 3, and the 4 left over let the section sent back redo all 3.
    out = run(Deps(model=ScriptedModel(), script=Script(send_back=(1,)),
                   budget=Budget(max_calls=15)))
    assert len([s for s in section_steps(out["ledger"], 1) if s.key[0] == 4]) == 3
    assert out["status"] == COMPLETE and spent(out["ledger"]).calls == 15


def test_a_section_cut_short_keeps_the_draft_it_had() -> None:
    # 14 calls leave the section sent back 2: it searches twice and stops before its
    # draft. The report shows its first draft, marked budget_exhausted, never a blank.
    out = run(Deps(model=ScriptedModel(), script=Script(send_back=(1,)),
                   budget=Budget(max_calls=14)))
    assert [(s.index, s.round, s.status) for s in out["sections"]] == [
        (0, 0, COMPLETE), (1, 0, COMPLETE), (1, 1, BUDGET_EXHAUSTED), (2, 0, COMPLETE)]
    assert out["result"].split("\n\n")[1].startswith("Methods, round 0: ")
    assert out["status"] == BUDGET_EXHAUSTED and spent(out["ledger"]).calls == 14


def test_a_branch_past_its_call_share_ends_budget_exhausted() -> None:
    # 13 calls leave 4, 4, 3: the section that wants 5 searches stops after 3.
    out = run(Deps(model=ScriptedModel(), script=Script(searches=((2, 5),)),
                   budget=Budget(max_calls=13)))
    assert [s.status for s in out["sections"]] == [COMPLETE, COMPLETE, BUDGET_EXHAUSTED]
    assert len(section_steps(out["ledger"], 2)) == 3


def test_a_branch_stops_once_its_token_share_is_spent() -> None:
    # 15 tokens a call: the editor's leaves 25 per section, two searches spend 30.
    out = run(Deps(model=ScriptedModel(), budget=Budget(max_tokens=90)))
    assert [s.status for s in out["sections"]] == [BUDGET_EXHAUSTED] * 3
    for index in range(3):
        assert spent(section_steps(out["ledger"], index)).tokens == 30


def test_the_step_after_a_join_keeps_its_call() -> None:
    # 10 calls: the editor's 1 and the review's 1 leave 3, 3, 2, so the last section
    # stops before its draft, and the review still runs.
    out = run(Deps(model=ScriptedModel(), budget=Budget(max_calls=10)))
    assert [s.status for s in out["sections"]] == [COMPLETE, COMPLETE, BUDGET_EXHAUSTED]
    assert out["ledger"][-1].name == "review"
    assert out["status"] == BUDGET_EXHAUSTED and spent(out["ledger"]).calls == 10


def test_a_run_whose_review_has_no_budget_left_is_not_complete() -> None:
    # Tokens are counted once spent, so the sections can use the last of them: each
    # is complete, the review cannot run, and the run says so.
    out = run(Deps(model=ScriptedModel(), budget=Budget(max_tokens=150)))
    assert [s.status for s in out["sections"]] == [COMPLETE] * 3
    assert "review" not in [s.name for s in out["ledger"]]
    assert out["status"] == BUDGET_EXHAUSTED


def test_a_passed_deadline_stops_every_branch_before_its_first_call() -> None:
    clock = FakeClock()
    out = run(Deps(model=ScriptedModel(clock=clock, seconds_per_call=10.0),
                   deadline=Deadline(at=5.0, clock=clock)))
    assert [s.status for s in out["sections"]] == [BUDGET_EXHAUSTED] * 3
    assert [s.name for s in out["ledger"]] == ["editor"]


def test_every_branch_checks_the_deadline_before_each_call() -> None:
    clock = FakeClock()
    out = run(Deps(model=ScriptedModel(clock=clock, seconds_per_call=10.0),
                   script=Script(plan=("Only",)), deadline=Deadline(at=25.0, clock=clock)))
    # editor 0 -> 10, searches 10 -> 20 -> 30, and the draft finds the time up
    assert [s.name for s in out["ledger"]] == ["editor", "researcher", "researcher"]
    assert out["sections"][0].status == BUDGET_EXHAUSTED


def test_the_deadline_is_the_budget_less_the_time_a_run_already_used() -> None:
    clock = FakeClock()
    clock.advance(100.0)
    assert Deadline.for_budget(Budget(max_seconds=600), clock=clock).at == 700.0
    resumed = Deadline.for_budget(Budget(max_seconds=600), used=540.0, clock=clock)
    assert resumed.at == 160.0 and not resumed.passed()
    assert Deadline.for_budget(Budget(max_seconds=600), used=700.0, clock=clock).passed()
    assert Deadline.for_budget(Budget(), used=700.0, clock=clock).at is None


def test_a_run_has_used_the_time_between_its_first_and_latest_checkpoints(tmp_path) -> None:
    with checkpoints(tmp_path / "checkpoints.db") as saver:
        started = time.time()
        run(Deps(model=ScriptedModel(delay=lambda key: 0.02)), saver=saver)
        wall = time.time() - started
        used, unknown = time_used(saver, "t"), time_used(saver, "never-ran")
    # The editor, a section's 3 calls (sections run at once) and the review: 0.1 s at least.
    assert 0.1 <= used <= wall and unknown == 0.0


# -- checkpoints ------------------------------------------------------------------------


def test_a_killed_run_resumes_without_rerunning_finished_branches(tmp_path) -> None:
    db, log, err = tmp_path / "checkpoints.db", tmp_path / "calls.log", tmp_path / "child.err"

    def finished_sections() -> int:
        try:
            with closing(sqlite3.connect(db, timeout=5)) as conn:
                return conn.execute(
                    "SELECT count(*) FROM writes WHERE thread_id = 'job' AND checkpoint_ns = '' "
                    "AND channel = 'sections'").fetchone()[0]
        except sqlite3.OperationalError:
            return 0

    process = child("start", str(db), "job", str(tmp_path), "--block", "1", "--delay", "0.1",
                    err=err)
    try:
        wait_for(lambda: "waiting 1" in read(log) and finished_sections() == 2,
                 lambda: f"log={read(log)!r} stderr={read(err)!r}")
    finally:
        process.kill()  # SIGKILL: no cleanup runs
        process.wait()
    before = read(log).splitlines()
    with checkpoints(db) as saver:
        # The killed run's time is on file, counted to its latest nested checkpoint: the
        # editor's call and two searches at least, where its own graph saw only the editor.
        assert time_used(saver, "job") >= 0.3
    (tmp_path / "release").write_text("go")
    resumed = subprocess.run([sys.executable, str(SLICE), "resume", str(db), "job", str(tmp_path)],
                             capture_output=True, text=True, timeout=120)
    assert resumed.returncode == 0, resumed.stderr
    after = [line.split(" ", 1)[1] for line in read(log).splitlines()[len(before):]]
    # Only the unfinished draft and the review ran again: no editor, no finished
    # section, and not the blocked section's searches, saved in its own checkpoint.
    assert after == ["writer 1", "review -1"]
    with checkpoints(db) as saver:
        state = build(saver).get_state({"configurable": {"thread_id": "job"}}).values
    assert state["status"] == COMPLETE and [s.status for s in state["sections"]] == [COMPLETE] * 3
    assert len({step.key for step in state["ledger"]}) == len(state["ledger"]) == CALLS
    assert len(state["board"]) == len({e.chunk_id for e in state["board"]}) == 7


def test_each_branch_checkpoints_its_graph_under_its_own_namespace(tmp_path) -> None:
    db = tmp_path / "checkpoints.db"
    with checkpoints(db) as saver:
        run(Deps(model=ScriptedModel(), script=Script(send_back=(1,))), saver=saver)
        with closing(sqlite3.connect(db)) as conn:
            namespaces = [row[0] for row in conn.execute(
                "SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE checkpoint_ns != ''")]
        last = [saver.get_tuple({"configurable": {"thread_id": "t", "checkpoint_ns": ns}})
                for ns in namespaces]
    assert len(namespaces) == 4 and all(ns.startswith("section_team:") for ns in namespaces)
    held = {(t.checkpoint["channel_values"]["index"], t.checkpoint["channel_values"]["round"])
            for t in last}
    assert held == {(0, 0), (1, 0), (2, 0), (1, 1)}


def test_processes_opening_a_new_checkpoint_file_at_once_all_get_in(tmp_path) -> None:
    # A new file's switch to WAL needs the only lock on it, and SQLite fails the
    # loser at once, without the busy timeout, so the switch is retried.
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(6)
    processes = [context.Process(target=graph_slice.open_at_once,
                                 args=(str(tmp_path / "new.db"), barrier)) for _ in range(6)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
    assert [process.exitcode for process in processes] == [0] * 6


def test_two_processes_share_one_checkpoint_file(tmp_path) -> None:
    db, threads = tmp_path / "checkpoints.db", ("job-a", "job-b")
    processes = [child("start", str(db), thread, str(tmp_path / thread), "--sections", "8",
                       "--block", "0", err=tmp_path / f"{thread}.err")
                 for thread in threads]
    try:
        # Each stops before one draft, so both are part-way through a run at once ...
        wait_for(lambda: all("waiting 0" in read(tmp_path / t / "calls.log") for t in threads),
                 lambda: " ".join(read(tmp_path / f"{t}.err") for t in threads))
        for thread in threads:  # ... and then both finish, writing to the file together.
            (tmp_path / thread / "release").write_text("go")
        for thread, process in zip(threads, processes, strict=True):
            assert process.wait(timeout=120) == 0, read(tmp_path / f"{thread}.err")
    finally:
        for process in processes:
            process.kill()
    with checkpoints(db) as saver:
        assert saver.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert saver.conn.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT * 1000
        graph = build(saver)
        for thread in threads:
            state = graph.get_state({"configurable": {"thread_id": thread}}).values
            assert state["topic"] == thread and state["status"] == COMPLETE
            assert len(state["sections"]) == 8 and len(state["ledger"]) == 1 + 8 * 3 + 1


def test_two_threads_in_one_file_see_nothing_of_each_other(tmp_path) -> None:
    # Two MCP jobs, or two web requests: one process, one checkpoint file, at once.
    a, b = ({"configurable": {"thread_id": t}} for t in ("job-a", "job-b"))
    with checkpoints(tmp_path / "checkpoints.db") as saver:
        graph = build(saver)
        with ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(graph.invoke, {"topic": "alpha"}, a,
                                context=Deps(model=ScriptedModel()), durability="sync"),
                    pool.submit(graph.invoke, {"topic": "beta"}, b,
                                context=Deps(model=ScriptedModel(), script=Script(plan=("Only",))),
                                durability="sync")]
            for job in jobs:
                job.result()
        first, second = graph.get_state(a).values, graph.get_state(b).values
        history = list(graph.get_state_history(a))
    assert first["topic"] == "alpha" and len(first["sections"]) == 3
    assert second["topic"] == "beta" and len(second["sections"]) == 1
    assert all(s.config["configurable"]["thread_id"] == "job-a" for s in history)
    assert not any(s.values.get("topic") == "beta" for s in history)


def test_deleting_a_finished_run_clears_its_nested_graphs_too(tmp_path) -> None:
    # What M3's graphs.run does once a run reaches a terminal state.
    with checkpoints(tmp_path / "checkpoints.db") as saver:
        run(Deps(model=ScriptedModel(), script=Script(send_back=(1,))), saver=saver, thread="done")
        run(Deps(model=ScriptedModel()), saver=saver, thread="kept")
        saver.delete_thread("done")
        held = [dict(saver.conn.execute(
            f"SELECT thread_id, count(*) FROM {table} GROUP BY thread_id").fetchall())
            for table in ("checkpoints", "writes")]
    assert all("done" not in rows and rows["kept"] > 0 for rows in held)


def test_every_record_comes_back_from_a_checkpoint_as_itself(tmp_path) -> None:
    db = tmp_path / "checkpoints.db"
    with checkpoints(db) as saver:
        expected = run(Deps(model=ScriptedModel()), saver=saver)
    with checkpoints(db) as saver:  # a new connection, as a new process opens it
        loaded = build(saver).get_state({"configurable": {"thread_id": "t"}}).values
    assert loaded == expected
    step, evidence, section = loaded["ledger"][0], loaded["board"][0], loaded["sections"][0]
    assert type(step) is Step and type(step.key) is tuple and type(step.cost) is Cost
    assert type(evidence) is Evidence and type(evidence.found) is tuple
    assert all(type(found) is Found for found in evidence.found)
    assert type(section) is SectionResult
    assert len({*loaded["ledger"], *loaded["board"], *loaded["sections"]}) == CALLS + 7 + 3


def test_a_checkpoint_rebuilds_only_the_records_it_was_told_about() -> None:
    step = Step(key=(0, "editor", -1, 0), kind="model", name="editor", cost=Cost(calls=1))
    assert serializer().loads_typed(serializer().dumps_typed(step)) == step
    stranger = serializer(records=())
    assert not isinstance(stranger.loads_typed(stranger.dumps_typed(step)), Step)


@dataclass(frozen=True)
class Tagged(Record):
    """A record left off RECORDS."""

    index: int
    tags: tuple[str, ...] = ()


def test_state_a_checkpoint_would_not_give_back_is_refused_before_any_run() -> None:
    # Either would pass a first run and break only on resume, as a dict or a list.
    class Forgotten(TypedDict, total=False):
        items: list[Tagged]

    class Bare(TypedDict, total=False):
        at: tuple[int, ...]

    with pytest.raises(TypeError, match="Tagged is not in RECORDS"):
        require_checkpointable(Forgotten)
    with pytest.raises(TypeError, match="Bare.at is a tuple"):
        require_checkpointable(Bare)
    require_checkpointable(graph_slice.RunState, graph_slice.SectionState, graph_slice.SectionBrief)


# -- streaming and determinism ----------------------------------------------------------


def test_events_stream_from_inside_the_wrappers_as_v1_names_in_order() -> None:
    deps = Deps(model=ScriptedModel(delay=lambda key: random.uniform(0, 0.005)),
                script=Script(send_back=(1,)))
    events = list(v1_events(build(), {"topic": "t"}, {"configurable": {"thread_id": "s"}},
                            context=deps))
    names = [name for name, _ in events]
    assert set(names) == {"working", "section", "report", "done"}
    assert names[-2:] == ["report", "done"]
    for index in range(3):
        assert [n for n, p in events if p.get("index") == index] == (
            ["working", "section"] * (2 if index == 1 else 1))
    # "working" comes from inside the section graph the wrapper runs.
    raw = [(ns, data) for ns, mode, data in build().stream(
        {"topic": "t"}, {"configurable": {"thread_id": "r"}}, context=Deps(model=ScriptedModel()),
        stream_mode=["custom"], subgraphs=True) if data["event"] == "working"]
    assert len(raw) == 3 and all(ns and ns[0].startswith("section_team:") for ns, _ in raw)


def test_a_failure_ends_the_stream_with_failed_so_a_partial_run_never_looks_finished() -> None:
    events = list(v1_events(build(), {"topic": "t"}, {"configurable": {"thread_id": "f"}},
                            context=Deps(model=ScriptedModel(fail_on="writer"))))
    names = [name for name, _ in events]
    assert names[-2:] == ["failed", "done"] and "report" not in names
    assert "scripted failure in writer" in events[-2][1]["message"]


def test_a_run_that_ends_without_a_terminal_state_does_not_look_finished() -> None:
    # An Editor that plans nothing fans out to no one: the run just stops.
    events = list(v1_events(build(), {"topic": "t"}, {"configurable": {"thread_id": "e"}},
                            context=Deps(model=ScriptedModel(), script=Script(plan=()),
                                         budget=Budget(max_calls=5))))
    assert events == [("failed", {"message": "the run stopped without a terminal state"}),
                      ("done", {})]


def test_a_run_waiting_at_an_interrupt_does_not_look_finished(tmp_path) -> None:
    class Plain(TypedDict, total=False):
        answer: str

    graph = StateGraph(Plain)
    graph.add_node("ask", lambda state: {"answer": interrupt("approve the outline?")})
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    with checkpoints(tmp_path / "checkpoints.db") as saver:
        events = list(v1_events(graph.compile(checkpointer=saver), {},
                                {"configurable": {"thread_id": "i"}}))
    assert events == [("failed", {"message": "the run stopped to wait for input"}), ("done", {})]


def test_a_stream_item_without_an_event_name_is_skipped() -> None:
    class Plain(TypedDict, total=False):
        status: str

    def chatty(state: Plain) -> dict:
        get_stream_writer()({"token": "hello"})
        return {}

    def finish(state: Plain) -> dict:
        get_stream_writer()({"event": "report"})
        return {"status": COMPLETE}

    graph = StateGraph(Plain)
    graph.add_node("chatty", chatty)
    graph.add_node("finish", finish)
    graph.add_edge(START, "chatty")
    graph.add_edge("chatty", "finish")
    graph.add_edge("finish", END)
    assert list(v1_events(graph.compile(), {}, {"configurable": {"thread_id": "c"}})) == [
        ("report", {}), ("done", {})]


def test_the_same_replies_give_the_same_state_whatever_order_branches_finish(tmp_path) -> None:
    # I9 over 20 runs with randomized timing, through real checkpoints.
    states, orders = [], set()
    with checkpoints(tmp_path / "checkpoints.db") as saver:
        for attempt in range(20):
            rng, lock = random.Random(attempt), threading.Lock()

            def jitter(key: tuple, rng: random.Random = rng, lock: threading.Lock = lock) -> float:
                with lock:
                    return rng.uniform(0, 0.01)

            model = ScriptedModel(delay=jitter)
            state = run(Deps(model=model, script=Script(send_back=(1,))), saver=saver,
                        thread=f"run-{attempt}")
            states.append(without_seconds(state))
            orders.add(tuple(key[2] for key in model.calls if key[0] == 2 and key[4:5] == ("write",)))
    assert all(state == states[0] for state in states)
    assert len(orders) > 1  # the branches really did finish in different orders


def test_checkpointing_overhead_is_measured(tmp_path) -> None:
    # Reported, not judged: a bound on time would fail on a slow machine.
    def per_run(saver: Any, label: str) -> float:
        run(Deps(model=ScriptedModel()), saver=saver, thread=f"{label}-warm-up")
        started = time.perf_counter()
        for attempt in range(5):
            run(Deps(model=ScriptedModel()), saver=saver, thread=f"{label}-{attempt}")
        return (time.perf_counter() - started) / 5

    bare = per_run(None, "bare")
    with checkpoints(tmp_path / "checkpoints.db") as saver:
        saved = per_run(saver, "saved")
        writes = saver.conn.execute("SELECT count(*) FROM checkpoints").fetchone()[0] / 6
    each = max(saved - bare, 0.0) / writes
    print(f"\ncheckpointing: {bare * 1000:.1f} ms a run without, {saved * 1000:.1f} ms with; "
          f"{writes:.0f} checkpoints a run, {each * 1000:.2f} ms each")
    assert writes > 0
