"""Scoring the answers, and measuring the judge that scores them.

Asking a model whether evidence supports an answer measured 0/6 seven times.
So every judged number here is reported beside a control: the same claims
against quotes from other answers, which cannot be supported. If the judge
calls both supported it has told us nothing, and the report must say so
rather than quoting the first figure alone.
"""

from verbatim.answer_eval import (
    AnsweredCase,
    JudgeReport,
    claim_quote_pairs,
    judge_support,
    judge_with_control,
    score_answers,
)
from verbatim.answers import Answer
from verbatim.citations import Citation


def citation(chunk_id: str, quote: str, arxiv_id: str = "2608.11111v1") -> Citation:
    return Citation(
        arxiv_id=arxiv_id, base_arxiv_id=arxiv_id.split("v")[0], version=1,
        title="A paper", authors=["Ada Lovelace"], page_number=1, char_start=0,
        char_end=len(quote), quoted_text=quote, chunk_id=chunk_id,
        source_sha256="s", source_url="u", license_url=None,
        retrieved_at="2026-09-09T00:00:00Z",
    )


def answered(text: str, citations, expected=("2608.11111v1",), declined=False) -> AnsweredCase:
    return AnsweredCase(
        question="q",
        answer=Answer(question="q", text=text, citations=list(citations),
                      rejected=[], declined=declined),
        expected_papers=expected,
    )


def test_the_countable_half_needs_no_model() -> None:
    cases = [
        answered("Flux threads the ring [1].", [citation("a", "flux threads")]),
        answered("No passages in the corpus match this question.", [], declined=True),
        # Cited, but one sentence carries a claim with no marker.
        answered("Charge moves [1]. It was first measured in 1897 by Thomson.",
                 [citation("b", "charge moves")]),
        # Cites a paper, but not the one the question was written from.
        answered("Something else entirely [1].",
                 [citation("c", "something", arxiv_id="2608.99999v1")]),
    ]

    score = score_answers(cases)

    assert score.cases == 4 and score.declined == 1
    assert score.with_verified_citation == 3
    assert score.fully_cited == 2          # the uncited-sentence one does not count
    assert score.cited_expected_paper == 2


def test_each_claim_is_paired_with_the_citation_its_marker_points_at() -> None:
    case = answered(
        "Flux threads the ring [1]. Charge moves along the wire [2].",
        [citation("a", "flux threads"), citation("b", "charge moves")],
    )

    pairs = claim_quote_pairs(case)

    assert [quote.quoted_text for _claim, quote in pairs] == ["flux threads", "charge moves"]
    assert pairs[0][0].startswith("Flux threads")


def test_a_judge_that_agrees_with_everything_is_reported_as_useless() -> None:
    """The failure mode this project measured seven times, caught by the control."""
    cases = [
        answered("Flux threads the ring [1].", [citation("a", "flux threads")]),
        answered("Charge moves along the wire [1].", [citation("b", "charge moves")]),
    ]

    report = judge_with_control(cases, lambda system, user: "STATES")

    assert report.real_pairs == 2 and report.control_pairs == 2
    assert report.real_supported == 2 and report.control_supported == 2
    assert report.discriminates is False
    assert "carry no information" in report.render()
    assert "should not be quoted" in report.render()


def test_a_judge_that_separates_them_is_reported_as_usable() -> None:
    cases = [
        answered("Flux threads the ring [1].", [citation("a", "flux threads the ring")]),
        answered("Charge moves along the wire [1].", [citation("b", "charge moves along")]),
    ]

    def discerning(system, user):
        # Supports only when the quote's words are in the claim, which is what
        # a real pair looks like and a rotated one does not.
        claim, _, quote = user.partition("QUOTE: ")
        return "STATES" if quote.strip().lower()[:12] in claim.lower() else "UNRELATED"

    report = judge_with_control(cases, discerning)

    assert report.real_supported == 2 and report.control_supported == 0
    assert report.discriminates is True
    assert "means something" in report.render()


def test_a_control_pair_never_borrows_from_its_own_answer() -> None:
    """Otherwise the control is a real pair wearing a control's label."""
    seen = []

    def record(system, user):
        seen.append(user)
        return "UNRELATED"

    cases = [
        answered("Alpha claim [1].", [citation("a", "ALPHAQUOTE")]),
        answered("Beta claim [1].", [citation("b", "BETAQUOTE")]),
    ]
    judge_with_control(cases, record)

    controls = seen[2:]
    assert "ALPHAQUOTE" not in controls[0]   # alpha's claim borrowed beta's quote
    assert "BETAQUOTE" not in controls[1]


def test_an_unreadable_or_unreachable_verdict_counts_as_unsupported() -> None:
    def broken(system, user):
        raise RuntimeError("Cannot reach Ollama")

    assert judge_support(broken, "claim", "quote") == "UNRELATED"
    assert judge_support(lambda s, u: "I think perhaps", "claim", "quote") == "UNRELATED"
    assert judge_support(lambda s, u: "  states  ", "claim", "quote") == "STATES"


def test_no_citations_anywhere_produces_an_empty_report() -> None:
    report = judge_with_control([answered("Nothing cited.", [])], lambda s, u: "STATES")

    assert report == JudgeReport(0, 0, 0, 0)
    assert report.discriminates is False


def test_the_hard_control_borrows_from_the_same_answer() -> None:
    """A judge separating physics from recommender systems has shown nothing.

    The adversary that matters is a quote on the same subject, from the same
    paper, that still does not support this particular claim.
    """
    seen = []

    def record(system, user):
        seen.append(user)
        return "UNRELATED"

    case = AnsweredCase(
        question="q",
        answer=Answer(
            question="q",
            text="Flux threads the ring [1]. Charge moves along the wire [2].",
            citations=[citation("a", "ALPHAQUOTE"), citation("b", "BETAQUOTE")],
            rejected=[],
        ),
        expected_papers=("2608.11111v1",),
    )
    other = answered("Elsewhere [1].", [citation("c", "OTHERQUOTE")])

    report = judge_with_control([case, other], record)

    assert report.hard_pairs == 2      # each claim against its sibling's quote
    # The hard control never reaches outside the answer it came from.
    hard = seen[-2:]
    assert "OTHERQUOTE" not in " ".join(hard)


def test_discrimination_is_judged_against_the_hard_control_when_there_is_one() -> None:
    lenient = JudgeReport(
        real_pairs=10, real_supported=9,
        control_pairs=10, control_supported=0,   # easy negatives all rejected
        hard_pairs=10, hard_supported=8,         # near-misses all accepted
    )

    # Separating the easy control by 90 points counts for nothing when the
    # judge cannot tell a claim's own quote from its neighbour's.
    assert lenient.discriminates is False
    assert "does not separate" in lenient.render()
