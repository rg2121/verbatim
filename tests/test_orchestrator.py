"""The supervisor: routing by shape, retrying by strategy, judging by numbers."""

from pathlib import Path
import threading

from verbatim.answers import Answer
from verbatim.corpus import EmbeddingModel
from verbatim.embedding import embed_pending
from verbatim.orchestrator import (
    ACCEPTED,
    LADDERS,
    RETRY,
    Ledger,
    orchestrate,
    question_shape,
    verdict_for,
    work_section,
)
from verbatim.report import build_report, render_markdown

from helpers import corpus_with


def test_question_shape_routes_each_kind_of_question_to_its_ladder() -> None:
    assert question_shape("what is the Adam optimizer?") == "definitional"
    assert question_shape("how does GRPO differ from PPO?") == "comparison"
    assert question_shape("what came out this week on reasoning models?") == "recency"
    assert question_shape("how are verifiers trained for math problems?") == "open"
    # Every ladder starts somewhere different and ends at the agent, except
    # recency, which starts there because nothing else can answer it.
    assert LADDERS["recency"][0] == "agent"
    assert all(ladder[0] != "agent" for shape, ladder in LADDERS.items() if shape != "recency")


def test_verdicts_come_from_numbers_alone() -> None:
    cited = Answer(question="q", text="t [1]", citations=[object()], rejected=[])  # type: ignore[list-item]
    uncited = Answer(question="q", text="t", citations=[], rejected=[])

    assert verdict_for(cited, 0.9, warn_below=0.65) == ACCEPTED
    assert verdict_for(cited, None, warn_below=0.65) == ACCEPTED   # unchecked is not a failure
    assert verdict_for(cited, 0.4, warn_below=0.65) == RETRY       # off-subject passages
    assert verdict_for(uncited, 0.9, warn_below=0.65) == RETRY     # nothing verified


class StubEmbedder:
    def __init__(self) -> None:
        self.model = EmbeddingModel(
            model_id="stub", dimensions=3, normalized=True, revision="v0"
        )

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        raw = [float("flux" in lowered), float("charge" in lowered), 0.0]
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]

    def embed_passages(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


class QueueCompletion:
    """Answers in the order asked; the ledger test scripts one payload per attempt."""

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.lock = threading.Lock()

    def complete(self, system, user, schema):
        with self.lock:
            return self.payloads.pop(0)

    def judge(self, system, user):
        return "flux and charge"


class ScriptedAnswers:
    """Answers by the question in the prompt, safe to call from several threads."""

    def __init__(self, by_question: dict[str, dict]) -> None:
        self.by_question = by_question

    def complete(self, system, user, schema):
        for question, payload in self.by_question.items():
            if question in user:
                return payload
        raise AssertionError(f"no scripted answer for: {user[:80]}")

    def judge(self, system, user):
        return "flux and charge"


def flux_corpus(tmp_path: Path):
    corpus = corpus_with(
        tmp_path,
        {
            "2608.11111v1": ("Flux Paper", ["Magnetic flux threads the ring."]),
            "2608.22222v1": ("Charge Paper", ["Charge moves along the wire."]),
        },
    )
    embed_pending(corpus, StubEmbedder())
    return corpus


def cite(chunk_id: str, quote: str) -> dict:
    return {
        "relevant": True,
        "answer": "Claim [1].",
        "citations": [{"marker": 1, "chunk_id": chunk_id, "quoted_text": quote}],
    }


def test_a_section_whose_citation_fails_is_retried_with_a_different_strategy(
    tmp_path: Path,
) -> None:
    with flux_corpus(tmp_path) as corpus:
        chunk = next(c for c in corpus.all_chunks() if c.arxiv_id == "2608.11111v1")
        completion = QueueCompletion(
            [
                cite(chunk.chunk_id, "Flux never threads anything."),   # rejected
                cite(chunk.chunk_id, "Magnetic flux threads the ring."),  # verified
            ]
        )
        section, ledger = work_section(
            corpus, StubEmbedder(), completion,
            "how does flux differ from charge?", caller=None,
        )

    assert ledger.shape == "comparison"
    assert [a.strategy for a in ledger.attempts] == ["compare", "titles"]
    assert [a.verdict for a in ledger.attempts] == [RETRY, ACCEPTED]
    assert ledger.attempts[0].rejected == 1 and ledger.attempts[1].verified == 1
    assert ledger.outcome == "accepted"
    assert [c.arxiv_id for c in section.answer.citations] == ["2608.11111v1"]


def test_a_section_no_strategy_can_support_is_declared_a_gap(tmp_path: Path) -> None:
    with flux_corpus(tmp_path) as corpus:
        nothing = {"relevant": True, "answer": "The passages do not say.", "citations": []}
        # Two written attempts; the agent rung has no tool caller and so
        # retrieves nothing, which needs no model call at all.
        completion = QueueCompletion([nothing, nothing])
        section, ledger = work_section(
            corpus, StubEmbedder(), completion, "how does flux behave?", caller=None,
        )

    assert [a.strategy for a in ledger.attempts] == ["hybrid", "titles", "agent"]
    assert ledger.attempts[-1].passages == 0
    assert ledger.outcome.startswith("gap")
    assert section.answer.citations == []


def test_orchestrate_works_sections_in_parallel_and_keeps_their_order(tmp_path: Path) -> None:
    with flux_corpus(tmp_path) as corpus:
        chunk_of = {c.arxiv_id: c for c in corpus.all_chunks()}
        completion = ScriptedAnswers(
            {
                "What is flux?": cite(
                    chunk_of["2608.11111v1"].chunk_id, "Magnetic flux threads the ring."
                ),
                "How does charge move?": cite(
                    chunk_of["2608.22222v1"].chunk_id, "Charge moves along the wire."
                ),
            }
        )
        seen: list[str] = []
        report, ledger = orchestrate(
            corpus, StubEmbedder(), completion,
            lambda system, user: "What is flux?\nHow does charge move?",
            "electromagnetism",
            workers=2,
            on_section=lambda heading, done, total: seen.append(heading),
        )
        markdown = render_markdown(report, ledger=ledger)

    assert [s.heading for s in report.sections] == ["What is flux?", "How does charge move?"]
    assert [s.shape for s in ledger.sections] == ["definitional", "open"]
    assert all(s.outcome == "accepted" for s in ledger.sections)
    assert ledger.escalations == 0
    assert sorted(seen) == ["How does charge move?", "What is flux?"]
    assert "## How this review was made" in markdown
    assert "`hybrid`: " in markdown and "-> accepted" in markdown


class Unreachable:
    """A model that is down, to prove one dead section does not kill a run."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on

    def complete(self, system, user, schema):
        if self.fail_on is None or self.fail_on in user:
            raise RuntimeError("Cannot reach Ollama at http://localhost:11434")
        return {"relevant": True, "answer": "Fine.", "citations": []}

    def judge(self, system, user):
        return "flux and charge"


class TracingCaller:
    """A tool-choosing model that calls search once, so a route exists."""

    def __init__(self) -> None:
        self.calls = 0

    def choose_tool(self, system, messages, tools):
        self.calls += 1
        return ("search_chunks", {"query": "flux"}) if self.calls == 1 else None


def test_a_section_whose_every_rung_errors_is_contained_not_fatal(tmp_path: Path) -> None:
    with flux_corpus(tmp_path) as corpus:
        section, ledger = work_section(
            corpus, StubEmbedder(), Unreachable(), "how does flux behave?", caller=None,
        )

    assert section.answer.declined
    assert "could not be written" in section.answer.text
    assert [a.strategy for a in ledger.attempts] == ["hybrid", "titles", "agent"]
    # The two writing rungs errored; the agent rung had no tool-caller, so it
    # retrieved nothing and never reached the model at all.
    assert [bool(a.error) for a in ledger.attempts] == [True, True, False]
    assert ledger.outcome.startswith("failed: 2 of 3 strategies errored")
    # A failure is reported as a failure, never as a gap: a gap is a claim
    # about the corpus, and an unreachable model makes no such claim.
    assert "gap" not in ledger.outcome
    assert "failed - Cannot reach Ollama" in Ledger(sections=[ledger]).render_markdown()


def test_one_dead_section_does_not_discard_the_rest_of_the_review(tmp_path: Path) -> None:
    with flux_corpus(tmp_path) as corpus:
        report, ledger = orchestrate(
            corpus, StubEmbedder(), Unreachable(fail_on="charge"),
            lambda system, user: "What is flux?\nHow does charge move?",
            "electromagnetism",
        )

    assert [s.heading for s in report.sections] == ["What is flux?", "How does charge move?"]
    assert not report.sections[0].answer.declined
    assert report.sections[1].answer.declined
    assert ledger.sections[1].outcome.startswith("failed")


def test_the_agent_rung_records_the_tools_it_called(tmp_path: Path) -> None:
    with flux_corpus(tmp_path) as corpus:
        chunk = next(c for c in corpus.all_chunks() if c.arxiv_id == "2608.11111v1")
        # Two rungs decline to cite, so the ladder reaches the agent, which
        # cites correctly and is accepted.
        nothing = {"relevant": True, "answer": "The passages do not say.", "citations": []}
        completion = QueueCompletion([nothing, nothing, cite(chunk.chunk_id, "Magnetic flux threads the ring.")])
        caller = TracingCaller()
        _section, ledger = work_section(
            corpus, StubEmbedder(), completion, "how does flux behave?", caller=caller,
        )

    agent_attempt = ledger.attempts[-1]
    assert agent_attempt.strategy == "agent"
    assert "search_chunks" in agent_attempt.route
    assert f"via {agent_attempt.route}" in Ledger(sections=[ledger]).render_markdown()
    # A deterministic rung has no route to report; its name is what it did.
    assert ledger.attempts[0].route == ""


def test_every_section_says_what_it_rests_on(tmp_path: Path) -> None:
    with flux_corpus(tmp_path) as corpus:
        chunk_of = {c.arxiv_id: c for c in corpus.all_chunks()}
        completion = ScriptedAnswers({
            "What is flux?": cite(chunk_of["2608.11111v1"].chunk_id, "Magnetic flux threads the ring."),
        })
        report, ledger = orchestrate(
            corpus, StubEmbedder(), completion,
            lambda system, user: "What is flux?", "electromagnetism",
        )
        markdown = render_markdown(report, ledger=ledger)

    assert report.sections[0].evidence_lines
    assert "Answered from" in markdown
    assert "Those passages are about:" in markdown


def test_the_plain_report_is_the_orchestrator_with_one_rung(tmp_path: Path) -> None:
    """build_report is no longer a second copy of the loop."""
    with flux_corpus(tmp_path) as corpus:
        chunk_of = {c.arxiv_id: c for c in corpus.all_chunks()}
        payload = cite(chunk_of["2608.11111v1"].chunk_id, "Magnetic flux threads the ring.")
        plain = build_report(
            corpus, StubEmbedder(), ScriptedAnswers({"What is flux?": payload}),
            lambda system, user: "What is flux?", "electromagnetism",
        )
        laddered, ledger = orchestrate(
            corpus, StubEmbedder(), ScriptedAnswers({"What is flux?": payload}),
            lambda system, user: "What is flux?", "electromagnetism",
        )

    assert [s.heading for s in plain.sections] == [s.heading for s in laddered.sections]
    assert plain.sections[0].answer.text == laddered.sections[0].answer.text
    assert plain.sections[0].evidence_lines == laddered.sections[0].evidence_lines
    # One rung means no escalation was even possible.
    assert len(ledger.sections[0].attempts) == 1
