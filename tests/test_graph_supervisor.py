"""The LangGraph supervisor, and whether it agrees with the built-in one.

The point of a second implementation is not that a framework is better. It is
that two implementations behind one set of role contracts is the only thing
that shows the contracts are a real boundary. So the test that matters is
equivalence: given the same corpus, the same scripted model and the same
ladder, both supervisors must reach the same section by the same route.
"""

from pathlib import Path

import pytest

from verbatim.orchestrator import ACCEPTED, RETRY, work_section

from test_orchestrator import (
    QueueCompletion,
    ScriptedAnswers,
    StubEmbedder,
    cite,
    flux_corpus,
)

langgraph = pytest.importorskip(
    "langgraph.graph",
    reason="the optional langgraph extra: uv sync --extra langgraph",
)

from verbatim.graph_supervisor import (  # noqa: E402
    MISSING,
    work_section_with_graph,
)


def both_supervisors(corpus, payloads, question, ladder):
    """Run the same scripted attempt sequence through each supervisor."""
    builtin = work_section(
        corpus, StubEmbedder(), QueueCompletion(list(payloads)), question,
        ladder=ladder, per_section=4,
    )
    graphed = work_section_with_graph(
        corpus, StubEmbedder(), QueueCompletion(list(payloads)), question,
        ladder=ladder, per_section=4,
    )
    return builtin, graphed


def test_both_supervisors_reach_the_same_section_by_the_same_route(
    tmp_path: Path,
) -> None:
    """The equivalence that makes the comparison meaningful.

    A first strategy whose quote cannot be verified, then one whose quote can.
    Both must retry rather than accept, both must land on the second rung, and
    both must keep the same answer.
    """
    with flux_corpus(tmp_path) as corpus:
        chunk = corpus.all_chunks()[0]
        payloads = [
            cite(chunk.chunk_id, "Flux never threads anything."),
            cite(chunk.chunk_id, "Magnetic flux threads the ring."),
        ]
        (section_a, ledger_a), (section_b, ledger_b) = both_supervisors(
            corpus, payloads, "how does flux behave?", ("hybrid", "titles")
        )

    assert ledger_a.outcome == ledger_b.outcome == "accepted"
    assert [a.strategy for a in ledger_a.attempts] == ["hybrid", "titles"]
    assert [a.strategy for a in ledger_b.attempts] == ["hybrid", "titles"]
    assert [a.verdict for a in ledger_a.attempts] == [RETRY, ACCEPTED]
    assert [a.verdict for a in ledger_b.attempts] == [RETRY, ACCEPTED]
    assert section_a.answer.text == section_b.answer.text
    assert [c.chunk_id for c in section_a.answer.citations] == \
           [c.chunk_id for c in section_b.answer.citations]
    assert ledger_a.shape == ledger_b.shape


def test_both_supervisors_declare_the_same_gap(tmp_path: Path) -> None:
    """Agreement has to hold when nothing works, not only when something does."""
    with flux_corpus(tmp_path) as corpus:
        chunk = corpus.all_chunks()[0]
        payloads = [cite(chunk.chunk_id, "Flux never threads anything.")] * 2
        (_a, ledger_a), (_b, ledger_b) = both_supervisors(
            corpus, payloads, "how does flux behave?", ("hybrid", "titles")
        )

    assert ledger_a.outcome == ledger_b.outcome
    assert "gap" in ledger_a.outcome
    assert len(ledger_a.attempts) == len(ledger_b.attempts) == 2


def test_the_graph_stops_at_the_first_accepted_rung(tmp_path: Path) -> None:
    """The conditional edge is the `break`, and must not climb past success."""
    with flux_corpus(tmp_path) as corpus:
        chunk = corpus.all_chunks()[0]
        _section, ledger = work_section_with_graph(
            corpus, StubEmbedder(),
            QueueCompletion([cite(chunk.chunk_id, "Magnetic flux threads the ring.")]),
            "how does flux behave?", ladder=("hybrid", "titles", "agent"),
            per_section=4,
        )

    assert len(ledger.attempts) == 1
    assert ledger.attempts[0].verdict == ACCEPTED


def test_a_rung_that_raises_is_contained_in_the_graph_too(tmp_path: Path) -> None:
    """A dead model must escalate, not abort the graph."""
    class Dies:
        def complete(self, system, user, schema):
            raise RuntimeError("model went away")

        def judge(self, system, user):
            return "flux and charge"

    with flux_corpus(tmp_path) as corpus:
        _section, ledger = work_section_with_graph(
            corpus, StubEmbedder(), Dies(), "how does flux behave?",
            ladder=("hybrid", "titles"), per_section=4,
        )

    assert len(ledger.attempts) == 2
    assert all(a.error for a in ledger.attempts)
    assert "failed" in ledger.outcome


def test_the_reported_nodes_are_the_roles(tmp_path: Path) -> None:
    """The graph's shape is the argument for it, so it is asserted.

    Retriever, writer and verifier are separate nodes rather than one step,
    because they are separate Protocols. If they were fused the graph would
    say nothing the `for` loop does not.
    """
    from verbatim.graph_supervisor import build_graph

    with flux_corpus(tmp_path) as corpus:
        compiled = build_graph(corpus, StubEmbedder(), QueueCompletion([]))

    nodes = set(compiled.get_graph().nodes)
    assert {"route", "retrieve", "write", "verify", "escalate"} <= nodes


def test_a_missing_extra_says_how_to_install_it(monkeypatch, tmp_path: Path) -> None:
    """An opaque ImportError is a worse answer than an instruction."""
    import builtins

    from verbatim.graph_supervisor import build_graph

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("langgraph"):
            raise ImportError("no langgraph here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(RuntimeError) as raised:
        build_graph(None, None, None)  # type: ignore[arg-type]

    assert "uv sync --extra langgraph" in str(raised.value)
    assert MISSING in str(raised.value)


def test_orchestrate_can_drive_the_graph_with_every_option_it_passes(
    tmp_path: Path,
) -> None:
    """`report --agents --supervisor langgraph` crashed before any section ran.

    orchestrate hands every supervisor the critic's options, and the graph did
    not accept them - a TypeError on the command the README shows. Each
    supervisor was tested on its own, so nothing ran the two together.
    """
    from verbatim.orchestrator import orchestrate

    with flux_corpus(tmp_path) as corpus:
        chunk_of = {c.arxiv_id: c for c in corpus.all_chunks()}
        completion = ScriptedAnswers({
            "What is flux?": cite(
                chunk_of["2608.11111v1"].chunk_id, "Magnetic flux threads the ring."
            ),
        })
        _report, ledger = orchestrate(
            corpus, StubEmbedder(), completion,
            lambda system, user: "What is flux?", "electromagnetism",
            supervisor="langgraph", critic=True,
        )

    (section,) = ledger.sections
    assert section.outcome == "accepted"
    assert section.attempts[0].critique is not None


def test_the_critic_escalates_the_same_way_in_both_supervisors(tmp_path: Path) -> None:
    """A subject no passage covers sends both up the whole ladder."""
    with flux_corpus(tmp_path) as corpus:
        chunk = next(c for c in corpus.all_chunks() if c.arxiv_id == "2608.11111v1")
        payload = cite(chunk.chunk_id, "Magnetic flux threads the ring.")
        options = dict(
            caller=None, critic=True,
            complete_text=lambda system, user: "superconductivity",
        )
        _a, builtin = work_section(
            corpus, StubEmbedder(), QueueCompletion([payload] * 3), "what is flux?",
            **options,
        )
        _b, graphed = work_section_with_graph(
            corpus, StubEmbedder(), QueueCompletion([payload] * 3), "what is flux?",
            **options,
        )

    assert [a.strategy for a in graphed.attempts] == [a.strategy for a in builtin.attempts]
    assert [a.verdict for a in graphed.attempts] == [a.verdict for a in builtin.attempts]
    assert graphed.attempts[0].critique.unaddressed == ("superconductivity",)


def test_both_supervisors_answer_a_recency_question_from_the_catalogue(
    tmp_path: Path,
) -> None:
    from test_orchestrator import ListingCaller

    with flux_corpus(tmp_path) as corpus:
        question = "what came out this week on flux?"
        builtin = work_section(
            corpus, StubEmbedder(), QueueCompletion([]), question,
            caller=ListingCaller(),
        )
        graphed = work_section_with_graph(
            corpus, StubEmbedder(), QueueCompletion([]), question,
            caller=ListingCaller(),
        )

    for section, ledger in (builtin, graphed):
        assert ledger.outcome == "accepted"
        assert "2608.22222v1" in section.answer.text
    assert builtin[0].answer.text == graphed[0].answer.text


class _JudgeDies(QueueCompletion):
    """Writes an answer, then goes away before the subject check."""

    def judge(self, system, user):
        raise RuntimeError("Cannot reach Ollama (subject check)")


def test_both_supervisors_contain_a_model_that_fails_while_checking(
    tmp_path: Path,
) -> None:
    """The built-in loop recorded the failure and carried on; the graph raised,
    which would have taken a whole review down with one section."""
    with flux_corpus(tmp_path) as corpus:
        chunk = corpus.all_chunks()[0]
        payloads = [cite(chunk.chunk_id, chunk.text)]
        builtin = work_section(
            corpus, StubEmbedder(), _JudgeDies(list(payloads)), "what threads the ring",
            ladder=("hybrid",), per_section=4,
        )
        graphed = work_section_with_graph(
            corpus, StubEmbedder(), _JudgeDies(list(payloads)), "what threads the ring",
            ladder=("hybrid",), per_section=4,
        )

    assert builtin[1].outcome == graphed[1].outcome
    assert graphed[1].outcome.startswith("failed")


def test_both_supervisors_stop_at_the_first_rung_a_spent_budget_refuses(
    tmp_path: Path,
) -> None:
    from verbatim.contracts import Budget, BudgetGuard

    ladder = ("hybrid", "titles", "compare")
    with flux_corpus(tmp_path) as corpus:
        builtin = work_section(
            corpus, StubEmbedder(), QueueCompletion([]), "what threads the ring",
            ladder=ladder, per_section=4, guard=BudgetGuard(Budget(max_attempts=0)),
        )
        graphed = work_section_with_graph(
            corpus, StubEmbedder(), QueueCompletion([]), "what threads the ring",
            ladder=ladder, per_section=4, guard=BudgetGuard(Budget(max_attempts=0)),
        )

    assert len(builtin[1].attempts) == len(graphed[1].attempts) == 1
    assert graphed[1].attempts[0].error.startswith("budget")
