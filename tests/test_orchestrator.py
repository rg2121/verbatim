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
    orchestrate,
    question_shape,
    verdict_for,
    work_section,
)
from verbatim.report import render_markdown

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
