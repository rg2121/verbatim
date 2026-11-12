"""M2a: the control contract, the rules code holds every agent to.

Agents choose strategy; these are the invariants (I1, I2, I6, I10) and the legal
Lead actions, as pure functions over facts code computed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import chunked_corpus
from verbatim.agents.contract import (
    ABSTAINED,
    BUDGET_EXHAUSTED,
    COMPLETE,
    DROP,
    ERROR,
    FAILED,
    KEEP,
    KEEP_UNVERIFIED,
    NOT_CHECKED,
    PASSED,
    PROTOCOL_FAILURE,
    REFUTED,
    REMOVED,
    REPAIR,
    STANDS,
    TERMINAL_STATES,
    UNPROVEN,
    UNVERIFIED,
    VERIFIED,
    decision_refused,
    dominates,
    first_by_rank,
    lead_action_refused,
    legal_returns,
    legal_verdicts,
    marking,
    report_status,
    return_refused,
    revision_refused,
    tally,
    visible_status,
)
from verbatim.agents.state import Abstain, Attempt, Final, Inspect, Next, Tally
from verbatim.answers import answer_from_payload
from verbatim.corpus import ChunkMatch
from verbatim.reflection import passage_texts

PASSAGE = (
    "Transformers replace recurrence with attention. "
    "The model reaches 28.4 BLEU on the English-to-German translation task. "
    "Training took three and a half days on eight GPUs."
)
OTHER = "Dropout of 0.1 was applied to the output of each sub-layer."
ID1, ID2, ID3 = "a" * 16, "b" * 16, "c" * 16


# -- terminal states (I2) ---------------------------------------------------------------


def test_every_terminal_state_but_complete_is_marked_where_it_is_shown() -> None:
    assert marking(COMPLETE) is None
    marks = [marking(state) for state in TERMINAL_STATES if state != COMPLETE]
    assert all(marks) and len(set(marks)) == len(marks)
    assert marking(BUDGET_EXHAUSTED) == "incomplete (budget)"
    assert marking(PROTOCOL_FAILURE) == "incomplete (protocol)"


def test_a_report_is_complete_only_if_every_section_is_complete_or_abstained() -> None:
    assert report_status([COMPLETE, ABSTAINED, COMPLETE]) == COMPLETE
    assert report_status([COMPLETE, BUDGET_EXHAUSTED]) == BUDGET_EXHAUSTED
    # Otherwise it takes its worst section's state.
    assert report_status([BUDGET_EXHAUSTED, PROTOCOL_FAILURE, COMPLETE]) == PROTOCOL_FAILURE
    assert report_status([PROTOCOL_FAILURE, ERROR, ABSTAINED]) == ERROR
    with pytest.raises(ValueError, match="no sections"):
        report_status([])
    with pytest.raises(ValueError, match="not a terminal state"):
        report_status([COMPLETE, "finished"])


# -- citations (I1) ---------------------------------------------------------------------


@pytest.mark.parametrize("check", [PASSED, FAILED, NOT_CHECKED])
@pytest.mark.parametrize("decision", [None, KEEP, REPAIR, DROP, KEEP_UNVERIFIED])
def test_a_citation_shows_verified_only_if_its_check_passed_and_it_was_kept(check, decision) -> None:
    if decision_refused(check, decision):
        with pytest.raises(ValueError):
            visible_status(check, decision)
        return
    shown = visible_status(check, decision)
    assert (shown == VERIFIED) == (check == PASSED and decision in (None, KEEP))
    # A decision can lower a status, never raise it.
    order = [REMOVED, UNVERIFIED, VERIFIED]
    assert order.index(shown) <= order.index(visible_status(check))


def test_the_legal_decisions_follow_the_check() -> None:
    assert decision_refused(PASSED, KEEP) is None and decision_refused(FAILED, REPAIR) is None
    assert "failed" in decision_refused(FAILED, KEEP)
    assert decision_refused(PASSED, KEEP_UNVERIFIED) is not None
    assert "not been checked" in decision_refused(NOT_CHECKED, KEEP)
    assert decision_refused(FAILED, None) is None  # no decision keeps the default
    with pytest.raises(ValueError, match="not a check result"):
        decision_refused("maybe", KEEP)


# -- attempts (I10) ---------------------------------------------------------------------


FIRST, SECOND = "2608.11111v1", "2608.22222v1"
QUOTE = "The model reaches 28.4 BLEU on the English-to-German translation task"
FAILED_QUOTE = (1, SECOND, "Dropout of 0.2 was applied")  # not word for word: the check fails it
DROPOUT = "Dropout was applied to every sub-layer output in the model"
CLAIMS = ("Larger ensembles would generalise better across unseen languages",
          "Smaller vocabularies would train faster on every benchmark",
          "Deeper decoders would need far more memory than shallow ones")


@pytest.fixture
def tally_of(tmp_path: Path):
    """Check a Writer's answer and CITE lines against the two passages, and tally it."""
    corpus = chunked_corpus(tmp_path, {FIRST: [PASSAGE], SECOND: [OTHER]})
    matches = [ChunkMatch(chunk=c, score=1.0, snippet=c.text) for c in corpus.iter_chunks()]
    ids = {match.chunk.arxiv_id: match.chunk.chunk_id for match in matches}
    papers = corpus.papers_by_id(sorted(ids))

    def run(answer: str, *citations: tuple[int, str, str]) -> Tally:
        payload = {"answer": answer, "citations": [
            {"marker": marker, "chunk_id": ids[paper], "quoted_text": quote}
            for marker, paper, quote in citations]}
        checked = answer_from_payload("q", matches, payload, papers, quotations=True)
        return tally(checked, passage_texts(matches))

    run.first = ids[FIRST]
    return run


def test_an_attempt_is_tallied_from_the_check_alone(tally_of) -> None:
    found = tally_of(
        "The model reaches 28.4 BLEU on English-to-German translation [1]. "
        'Training took "three and a half weeks" on eight GPUs [1]. '
        f"{DROPOUT} [2]. "
        f"{CLAIMS[0]}. "
        "Convolutions outperform attention on every benchmark studied [1]. "
        "It also used label smoothing during all of its training [3].",
        (1, FIRST, QUOTE), (2, SECOND, "Dropout of 0.2 was applied"))
    # failed, as v1 counts problems: the quote behind [2], the [unverified] that [2]
    # and [3] leave in the text, and the quotation [1]'s passage does not hold.
    # unbacked: the convolutions sentence shares almost nothing with its passage.
    assert found == Tally(verified=1, failed=4, unbacked=1, uncited=1, passages=(tally_of.first,))


def test_pointing_claims_at_a_failed_quote_does_not_beat_leaving_them_unmarked(tally_of) -> None:
    # Each [unverified] the text shows is a failed check, so marking the claims with
    # the failed citation moves them from uncited to failed: no better than the draft.
    draft = tally_of(f"{DROPOUT} [1]. " + ". ".join(CLAIMS) + ".", FAILED_QUOTE)
    marked = tally_of(f"{DROPOUT} [1]. " + " ".join(f"{claim} [1]." for claim in CLAIMS), FAILED_QUOTE)
    assert (draft.failed, draft.uncited) == (2, 3) and (marked.failed, marked.uncited) == (5, 0)
    assert not dominates(marked, draft)
    assert first_by_rank([Attempt(1, draft), Attempt(2, marked, revises=1)]) == 1


def test_dropping_a_failed_citation_leaves_its_claims_counted(tally_of) -> None:
    # Citing nothing is how v1's Writer declines, but claims made without a citation
    # are still claims: deleting a failed citation never leaves an answer clean.
    draft = tally_of(f"{DROPOUT} [1]. " + ". ".join(CLAIMS) + ".", FAILED_QUOTE)
    bare = tally_of(f"{DROPOUT}. " + ". ".join(CLAIMS) + ".")
    assert bare == Tally(uncited=4)
    assert not dominates(bare, draft)


def test_a_decline_is_tallied_for_what_it_says(tally_of) -> None:
    assert tally_of("The passages do not say how the model was trained.") == Tally(uncited=1)
    # A quotation that names no citation, here the question's own words, fails no check.
    assert tally_of('The passages do not explain "what a tensor is" in any way.').failed == 0


def test_a_citation_no_marker_points_at_supports_nothing(tally_of) -> None:
    claim = "The model reaches 28.4 BLEU on English-to-German translation [1]."
    once = tally_of(claim, (1, FIRST, QUOTE))
    padded = tally_of(claim, (1, FIRST, QUOTE),
                   (2, SECOND, "Dropout of 0.1 was applied to the output of each sub-layer"))
    assert once == padded == Tally(verified=1, passages=(tally_of.first,))


def test_a_quotation_its_markers_passage_does_not_hold_is_a_failed_check(tally_of) -> None:
    # The quoted words are passage 2's, but the marker after them names [1].
    found = tally_of('In training, "Dropout of 0.1 was applied" to each layer [1]. '
                  "The output of each sub-layer had dropout applied to it [2].",
                  (1, FIRST, QUOTE),
                  (2, SECOND, "Dropout of 0.1 was applied to the output of each sub-layer"))
    assert found.failed == 1


def test_one_attempt_dominates_another_only_if_no_worse_on_every_count_and_better_on_one() -> None:
    draft = Tally(verified=2, failed=1, unbacked=0, uncited=1)
    assert dominates(Tally(verified=2, failed=0, unbacked=0, uncited=1), draft)
    assert dominates(Tally(verified=3, failed=1, unbacked=0, uncited=1), draft)
    assert not dominates(draft, draft)
    assert not dominates(Tally(verified=3, failed=2, unbacked=0, uncited=0), draft)


def test_a_revision_that_loses_a_verified_passage_is_refused_whatever_else_it_fixes() -> None:
    draft = Tally(verified=2, failed=1, unbacked=0, uncited=2, passages=(ID1, ID2))
    swapped = Tally(verified=2, failed=0, unbacked=0, uncited=0, passages=(ID1, ID3))
    assert dominates(swapped, draft)
    assert f"loses the verified citation to passage {ID2}" in revision_refused(draft, swapped)


def test_a_revision_with_more_unbacked_sentences_is_refused() -> None:
    draft = Tally(verified=1, failed=2, unbacked=0, uncited=0, passages=(ID1,))
    revision = Tally(verified=2, failed=0, unbacked=1, uncited=0, passages=(ID1, ID2))
    assert "1 unbacked sentence" in revision_refused(draft, revision)


def test_dropping_a_failed_citation_stays_legal() -> None:
    draft = Tally(verified=1, failed=1, unbacked=0, uncited=0, passages=(ID1,))
    assert revision_refused(draft, Tally(verified=1, passages=(ID1,))) is None


def test_only_an_attempt_no_other_dominates_may_be_returned() -> None:
    attempts = [Attempt(1, Tally(verified=1, failed=1, passages=(ID1,))),
                Attempt(2, Tally(verified=1, passages=(ID1,)), revises=1)]
    assert legal_returns(attempts) == (2,)
    assert "attempt 2 is at least as good on every count" in return_refused(1, attempts)
    assert "there is no attempt 3" in return_refused(3, attempts)
    # Equal attempts are both legal: the Lead chooses.
    assert legal_returns([Attempt(1, Tally(verified=1)), Attempt(2, Tally(verified=1))]) == (1, 2)


def test_two_attempts_cannot_share_a_number() -> None:
    with pytest.raises(ValueError, match="share a number"):
        legal_returns([Attempt(1, Tally()), Attempt(1, Tally(verified=1))])


def test_a_refused_revision_is_never_returned_and_does_not_count_against_its_draft() -> None:
    draft = Tally(verified=2, failed=1, unbacked=0, uncited=2, passages=(ID1, ID2))
    swapped = Tally(verified=2, failed=0, unbacked=0, uncited=0, passages=(ID1, ID3))
    attempts = [Attempt(1, draft), Attempt(2, swapped, revises=1)]
    assert legal_returns(attempts) == (1,)
    assert "attempt 1" in return_refused(2, attempts)


def test_a_revision_is_held_to_every_attempt_it_descends_from() -> None:
    one = Attempt(1, Tally(verified=2, passages=(ID1, ID2)))
    two = Attempt(2, Tally(verified=1, passages=(ID1,)), revises=1)  # refused: loses ID2
    three = Attempt(3, Tally(verified=2, passages=(ID1, ID3)), revises=2)
    assert legal_returns([one, two, three]) == (1,)
    assert "attempt 1" in return_refused(3, [one, two, three])


def test_an_incomplete_run_returns_the_first_legal_attempt_by_rank() -> None:
    # Most verified, then fewest failed, then fewest unbacked, then the earliest.
    many = Attempt(1, Tally(verified=3, failed=2, passages=(ID1, ID2, ID3)))
    clean = Attempt(2, Tally(verified=2, passages=(ID1, ID2)))
    assert first_by_rank([many, clean]) == 1
    # The rank ignores uncited claims, so on it alone the earlier attempt would win;
    # but the later one dominates it, so the earlier may not be returned.
    loose = Attempt(1, Tally(verified=2, uncited=3, passages=(ID1, ID2)))
    tight = Attempt(2, Tally(verified=2, uncited=1, passages=(ID1, ID2)), revises=1)
    assert first_by_rank([loose, tight]) == 2
    assert first_by_rank([]) is None


# -- gap verdicts (I6) ------------------------------------------------------------------


def test_a_gap_verdict_is_legal_only_on_the_evidence_for_it() -> None:
    # Refuted needs a citation whose check passed; stands needs the effort floor
    # (3 queries, 8 papers); unproven is always legal.
    assert legal_verdicts(refutation_verified=True, queries=1, papers=1) == (REFUTED, UNPROVEN)
    assert legal_verdicts(refutation_verified=False, queries=3, papers=8) == (STANDS, UNPROVEN)
    # A refutation the check passed is never reported as a gap that stands, as in v1.
    assert legal_verdicts(refutation_verified=True, queries=3, papers=8) == (REFUTED, UNPROVEN)
    assert legal_verdicts(refutation_verified=False, queries=3, papers=7) == (UNPROVEN,)


# -- the Lead's actions -----------------------------------------------------------------


BOARD = {ID1, ID2, ID3}


def test_the_lead_queues_steps_from_the_roster_with_what_each_needs() -> None:
    def refused(action: Next, board=BOARD, attempts=()) -> str | None:
        return lead_action_refused(action, board=board, attempts=list(attempts))

    assert refused(Next(("writer",))) is None
    assert refused(Next(("researcher", "writer"), brief="find the training time")) is None
    assert "'librarian' is not a step" in refused(Next(("librarian",)))
    assert "at most 4 steps" in refused(Next(("researcher",) * 5, brief="x"))
    assert "BRIEF" in refused(Next(("researcher", "writer")))
    assert "PASSAGES" in refused(Next(("researcher",), brief="x", passages=(ID1,)))
    assert f"passage {'d' * 16} is not on the board" in refused(Next(("writer",), passages=("d" * 16,)))
    assert "no draft" in refused(Next(("verifier", "writer")))
    assert refused(Next(("writer", "verifier"))) is None
    assert "at least one step" in refused(Next(()))
    # A writer step needs passages: on the board, or from research queued before it.
    assert "no passages" in refused(Next(("writer",)), board=set())
    assert refused(Next(("researcher", "writer"), brief="x"), board=set()) is None
    assert "no passages" in refused(Next(("writer", "researcher"), brief="x"), board=set())


def test_the_lead_inspects_at_most_four_passages_from_the_board() -> None:
    assert lead_action_refused(Inspect((ID1, ID2)), board=BOARD, attempts=[]) is None
    assert "at most 4" in lead_action_refused(Inspect((ID1,) * 5), board=BOARD, attempts=[])
    assert "not on the board" in lead_action_refused(Inspect(("d" * 16,)), board=BOARD, attempts=[])


def test_the_lead_returns_only_a_legal_attempt_and_may_always_abstain() -> None:
    attempts = [Attempt(1, Tally(verified=1, failed=1, passages=(ID1,))),
                Attempt(2, Tally(verified=1, passages=(ID1,)), revises=1)]
    assert lead_action_refused(Final(2), board=BOARD, attempts=attempts) is None
    assert "attempt 2 is at least as good" in lead_action_refused(
        Final(1), board=BOARD, attempts=attempts)
    assert "nothing has been written" in lead_action_refused(Final(1), board=BOARD, attempts=[])
    assert lead_action_refused(Abstain("off topic"), board=set(), attempts=[]) is None
