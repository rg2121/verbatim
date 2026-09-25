"""Claims of absence, and the agent paid to destroy them."""

from __future__ import annotations

from pathlib import Path

import pytest

from verbatim.gaps import (
    MIN_PAPERS,
    MIN_QUERIES,
    REFUTED,
    STANDS,
    UNPROVEN,
    Effort,
    Gap,
    checkable,
    falsify,
    propose_gaps,
)

from test_orchestrator import StubEmbedder, cite, flux_corpus


@pytest.mark.parametrize(
    "claim, ok",
    [
        ("no paper evaluates sparse attention on multilingual retrieval", True),
        ("nothing here measures the energy cost of speculative decoding", True),
        # Unfalsifiable: no single paper can disprove any of these, so they
        # would survive every attempt and be reported as confirmed absence.
        ("evaluation of long-context models is under-explored", False),
        ("more work is needed on tokenizer design", False),
        ("this remains an open question", False),
        ("these passages do not cover evaluation", False),
        ("gaps exist", False),
    ],
)
def test_a_gap_must_be_something_one_paper_could_disprove(claim: str, ok: bool) -> None:
    """The filter that matters most.

    An unfalsifiable claim cannot be destroyed by the Falsifier, so it stands
    for ever and is reported as confirmed absence - a worse outcome than
    proposing nothing at all.
    """
    assert checkable(claim) is ok


def test_the_gap_check_is_not_the_question_check(caplog) -> None:
    """"Nothing here measures X" is right for a gap and wrong for a question.

    `distill` rejects "here" because a retrieval question must stand alone
    without its source passage. A gap claim is a statement about this corpus,
    so the same rule borrowed wholesale would reject the correct phrasing.
    """
    from verbatim.distill import standalone

    claim = "nothing here measures the energy cost of speculative decoding"

    assert standalone(claim) is False  # the question rule rejects it
    assert checkable(claim) is True    # the gap rule keeps it


def test_a_lazy_falsifier_cannot_confirm_a_gap(tmp_path: Path) -> None:
    """The referee is arithmetic, and this is why it exists.

    A Falsifier that runs one query and gives up would confirm every gap it
    was handed. Its failure only counts once it has demonstrably tried, and no
    model is asked whether the attempt was serious.
    """
    class Lazy:
        """Writes a single query and finds nothing to cite."""

        def complete(self, system, user, schema):
            return {"answer": "The passages do not say.", "citations": []}

        def judge(self, system, user):
            return "flux"

    def one_query(system, user):
        return "QUERY: flux\n"

    with flux_corpus(tmp_path) as corpus:
        finding = falsify(
            corpus, StubEmbedder(), Lazy(), one_query,
            Gap(topic="flux", claim="no paper measures flux in cold rings"),
        )

    assert finding.verdict == UNPROVEN
    assert len(finding.effort.queries) < MIN_QUERIES
    assert "did not try hard enough" in finding.reason


def test_a_refutation_has_to_be_a_verified_quote(tmp_path: Path) -> None:
    """Saying "I found one" is not refuting. Quoting a real passage is.

    The Falsifier goes through the same mechanical quote check as every other
    citation in this project, so a hallucinated refutation removes a gap only
    if the words are genuinely in the passage.
    """
    with flux_corpus(tmp_path) as corpus:
        chunk = next(c for c in list(corpus.iter_chunks()) if c.arxiv_id == "2608.11111v1")

        class Invents:
            def complete(self, system, user, schema):
                return cite(chunk.chunk_id, "words that are not in any passage")

            def judge(self, system, user):
                return "flux"

        class Quotes:
            def complete(self, system, user, schema):
                return cite(chunk.chunk_id, "Magnetic flux threads the ring.")

            def judge(self, system, user):
                return "flux"

        def four_queries(system, user):
            return "".join(
                f"QUERY: flux {n}\n" for n in ("ring", "thread", "magnet", "charge")
            )

        gap = Gap(topic="flux", claim="no paper measures flux in cold rings")
        invented = falsify(corpus, StubEmbedder(), Invents(), four_queries, gap)
        genuine = falsify(corpus, StubEmbedder(), Quotes(), four_queries, gap)

    # The invented quote fails verification, so it refutes nothing.
    assert invented.verdict != REFUTED
    assert genuine.verdict == REFUTED
    assert genuine.citation is not None
    assert genuine.citation.arxiv_id == "2608.11111v1"


def test_effort_is_counted_not_judged() -> None:
    """Whether an attempt was genuine is arithmetic, by contract."""
    thin = Effort(queries=("a", "b"), papers=tuple(f"p{n}" for n in range(20)),
                  passages=40)
    narrow = Effort(queries=("a", "b", "c", "d"), papers=("p1",), passages=3)
    real = Effort(queries=tuple("abcd"), papers=tuple(f"p{n}" for n in range(MIN_PAPERS)),
                  passages=30)

    assert thin.genuine is False     # too few queries
    assert narrow.genuine is False   # too few papers reached
    assert real.genuine is True


def test_a_hidden_paper_is_out_of_the_falsifiers_reach(tmp_path: Path) -> None:
    """How a gap with a known answer is planted, without deleting anything.

    The experiment needs a gap whose answer exists: hide the paper, and a
    correct Falsifier must fail; unhide it, and the same Falsifier must
    succeed. Deleting the paper would be destructive and unrepeatable.
    """
    with flux_corpus(tmp_path) as corpus:
        chunk = next(c for c in list(corpus.iter_chunks()) if c.arxiv_id == "2608.11111v1")

        class Quotes:
            def complete(self, system, user, schema):
                return cite(chunk.chunk_id, "Magnetic flux threads the ring.")

            def judge(self, system, user):
                return "flux"

        def four_queries(system, user):
            return "".join(f"QUERY: flux {n}\n" for n in ("ring", "a", "b", "c"))

        gap = Gap(topic="flux", claim="no paper measures flux threading a ring")
        visible = falsify(corpus, StubEmbedder(), Quotes(), four_queries, gap)
        planted = falsify(
            corpus, StubEmbedder(), Quotes(), four_queries, gap,
            hidden=["2608.11111v1"],
        )

    assert visible.verdict == REFUTED
    # With the paper hidden the Falsifier cannot reach it, so it cannot cite it.
    assert "2608.11111v1" not in planted.effort.papers
    assert planted.verdict != REFUTED


def test_the_prospector_reports_what_it_rejected(tmp_path: Path) -> None:
    """The rejection rate is the signal that the instruction is followed."""
    def proposes(system, user):
        return (
            "GAP: no paper measures flux in superconducting rings below one kelvin\n"
            "GAP: more work is needed on flux\n"
            "GAP: this remains an open question\n"
        )

    with flux_corpus(tmp_path) as corpus:
        gaps, rejected = propose_gaps(corpus, StubEmbedder(), proposes, "flux")

    assert [g.claim for g in gaps] == [
        "no paper measures flux in superconducting rings below one kelvin"
    ]
    assert len(rejected) == 2


def test_a_verdict_does_not_claim_more_than_it_established(tmp_path: Path) -> None:
    """The quote check proves the words are there, not that they answer.

    Whether a cited passage is responsive to the gap claim is the model's
    judgement, and this project has measured seven times that models judge
    relevance badly. A refutation that said "a paper in the corpus does this"
    asserted the part that was never verified, so the wording is pinned here:
    a standing gap must never be described as absence.
    """
    from verbatim.gaps import Effort, Finding, render

    stands = Finding(
        gap=Gap(topic="t", claim="no paper measures flux in cold rings"),
        verdict=STANDS,
        effort=Effort(queries=tuple("abcd"), papers=tuple(f"p{n}" for n in range(9)),
                      passages=20),
        reason="survived a bounded adversarial search - which is not the same as "
               "proof of absence",
    )

    text = render([stands], rejected=[])

    assert "adversarial search" in text
    assert "not" in text and "proof" in text
    # The claim the verification cannot support must not appear anywhere.
    assert "does not exist" not in text
    assert "proof the literature lacks it" in text
