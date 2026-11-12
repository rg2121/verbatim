"""The control contract: what agents may decide, and the rules code holds them to.

Agents choose strategy within a typed action space; code enforces invariants and
runtime limits. These are the invariants and the legal actions, as pure
functions over facts code computed, never over a model's judgement:

- I1, a citation is shown as verified only if its check passed and it was kept;
- I2, every result ends in a terminal state, and any but complete is marked;
- I6, a gap verdict is legal only on the evidence for it;
- I10, an attempt another dominates is never returned, and v1's two refusals hold.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

from verbatim import gaps
from verbatim.agents.state import Abstain, Attempt, Final, Inspect, Next, Tally
from verbatim.answers import Answer
from verbatim.critique import unbacked_sentences, uncited_sentences
from verbatim.reflection import UNVERIFIED as UNVERIFIED_MARKER
from verbatim.reflection import dangling_markers

# -- terminal states (I2) ---------------------------------------------------------------

COMPLETE = "complete"
ABSTAINED = "abstained"
BUDGET_EXHAUSTED = "budget_exhausted"
PROTOCOL_FAILURE = "incomplete_protocol_failure"
ERROR = "error"
# From best to worst, so a report can take its worst section's state.
TERMINAL_STATES = (COMPLETE, ABSTAINED, BUDGET_EXHAUSTED, PROTOCOL_FAILURE, ERROR)
_MARKS = {
    COMPLETE: None,
    ABSTAINED: "declined: the corpus does not answer this",
    BUDGET_EXHAUSTED: "incomplete (budget)",
    PROTOCOL_FAILURE: "incomplete (protocol)",
    ERROR: "failed",
}


def marking(status: str) -> str | None:
    """How a result is marked in the CLI, the web page and MCP; None only for complete."""
    return _MARKS[_known(status)]


def report_status(sections: Iterable[str]) -> str:
    """A report is complete only if every section is complete or abstained;
    otherwise it takes its worst section's state."""
    statuses = [_known(status) for status in sections]
    if not statuses:
        raise ValueError("a report with no sections has no status")
    if all(status in (COMPLETE, ABSTAINED) for status in statuses):
        return COMPLETE
    return max(statuses, key=TERMINAL_STATES.index)


def _known(status: str) -> str:
    if status not in TERMINAL_STATES:
        raise ValueError(f"{status!r} is not a terminal state")
    return status


# -- citations (I1) ---------------------------------------------------------------------

PASSED, FAILED, NOT_CHECKED = "passed", "failed", "not_checked"  # the check, run by code
KEEP, REPAIR, DROP, KEEP_UNVERIFIED = "keep", "repair", "drop", "keep_unverified"  # the Verifier
VERIFIED, UNVERIFIED, REMOVED = "verified", "unverified", "removed"  # what is shown, by code
LEGAL_DECISIONS = {
    PASSED: (KEEP, REPAIR, DROP),
    FAILED: (REPAIR, DROP, KEEP_UNVERIFIED),
    NOT_CHECKED: (),
}


def decision_refused(check: str, decision: str | None) -> str | None:
    """Why `decision` is illegal for a citation whose check came out `check`, or
    None. No decision is always legal: a passed citation is kept, and a failed
    one shown as unverified."""
    if check not in LEGAL_DECISIONS:
        raise ValueError(f"{check!r} is not a check result")
    if decision is None or decision in LEGAL_DECISIONS[check]:
        return None
    if check == NOT_CHECKED:
        return "a citation that has not been checked takes no decision until it is"
    if check == FAILED:
        return (f"a citation whose check failed can only be {_either(LEGAL_DECISIONS[FAILED])}; "
                "it is never shown as verified")
    return f"a citation whose check passed can only be {_either(LEGAL_DECISIONS[PASSED])}"


def visible_status(check: str, decision: str | None = None) -> str:
    """How a citation is shown (I1): verified only if its check passed and it was
    kept. A decision can lower that, never raise it; a repaired citation is shown
    through its repair, which is checked again as a citation of its own."""
    if why := decision_refused(check, decision):
        raise ValueError(why)
    if decision in (REPAIR, DROP):
        return REMOVED
    return VERIFIED if check == PASSED else UNVERIFIED


def _either(options: Sequence[str]) -> str:
    return ", ".join(options[:-1]) + f" or {options[-1]}"


# -- attempts (I10) ---------------------------------------------------------------------


def tally(answer: Answer, payload: Mapping[str, Any], texts: Mapping[str, str]) -> Tally:
    """What the check found in an attempt. `payload` is the Writer's reply as
    parsed, and `texts` each passage's text by chunk id as the check reads it
    (reflection.passage_texts). Failed checks are quotes the check rejected,
    markers with no CITE line, and quotations in the prose no cited passage holds.
    An abstention, which cites nothing, makes no uncited claims, as in v1."""
    abstains = (not answer.citations and not answer.rejected
                and UNVERIFIED_MARKER not in answer.text)
    dangling = dangling_markers(dict(payload), list(texts.values()))
    return Tally(
        verified=len(answer.citations),
        failed=len(answer.rejected) + len(dangling) + len(answer.quotation_problems),
        unbacked=len(unbacked_sentences(answer, dict(texts))),
        uncited=0 if abstains else len(uncited_sentences(answer.text)),
        passages=tuple(sorted({citation.chunk_id for citation in answer.citations})),
    )


def dominates(a: Tally, b: Tally) -> bool:
    """Whether `a` is at least as good as `b` on every count and better on one:
    more verified citations, or fewer failed checks, unbacked sentences or
    uncited claims."""
    gains = (a.verified - b.verified, b.failed - a.failed, b.unbacked - a.unbacked,
             b.uncited - a.uncited)
    return min(gains) >= 0 and max(gains) > 0


def revision_refused(draft: Tally, revision: Tally, draft_name: str = "the draft") -> str | None:
    """v1's two refusals (reflection.better), which hold whatever else a revision
    fixes: it may not lose a passage the draft had a verified citation for, nor
    have more unbacked sentences. Dropping a failed citation loses nothing."""
    lost = sorted(set(draft.passages) - set(revision.passages))
    if lost:
        return f"it loses the verified citation to passage {lost[0]} that {draft_name} had"
    if revision.unbacked > draft.unbacked:
        return (f"it has {revision.unbacked} unbacked sentence{'s' * (revision.unbacked != 1)} "
                f"where {draft_name} had {draft.unbacked}")
    return None


def return_refused(number: int, attempts: Sequence[Attempt]) -> str | None:
    """Why attempt `number` may not be returned, or None (I10). A revision that
    breaks v1's refusals against any attempt it descends from may not; of the
    rest, none that another of the rest dominates."""
    by_number = {attempt.number: attempt for attempt in attempts}
    if number not in by_number:
        return f"there is no attempt {number}"
    if why := _refused(by_number[number], by_number):
        return why
    for other in attempts:
        if (other.number != number and _refused(other, by_number) is None
                and dominates(other.tally, by_number[number].tally)):
            return f"attempt {other.number} is at least as good on every count and better on one"
    return None


def legal_returns(attempts: Sequence[Attempt]) -> tuple[int, ...]:
    """The attempts that may be returned, in the order written. Never empty while
    any exist: a first draft revises nothing, so it is never refused, and of the
    rest some attempt is always undominated."""
    return tuple(attempt.number for attempt in attempts
                 if return_refused(attempt.number, attempts) is None)


def first_by_rank(attempts: Sequence[Attempt]) -> int | None:
    """What code returns when a run ends incomplete: of the legal attempts, the one
    with the most verified citations, then the fewest failed checks, then the
    fewest unbacked sentences, then the earliest."""
    legal = set(legal_returns(attempts))
    ranked = sorted((attempt for attempt in attempts if attempt.number in legal),
                    key=lambda a: (-a.tally.verified, a.tally.failed, a.tally.unbacked, a.number))
    return ranked[0].number if ranked else None


def _refused(attempt: Attempt, by_number: Mapping[int, Attempt]) -> str | None:
    """v1's refusals against each attempt `attempt` descends from, nearest first."""
    seen, parent = {attempt.number}, attempt.revises
    while parent is not None:
        if parent not in by_number:
            raise ValueError(f"attempt {attempt.number} descends from a missing attempt {parent}")
        if parent in seen:
            raise ValueError(f"attempt {attempt.number} descends from itself")
        seen.add(parent)
        if why := revision_refused(by_number[parent].tally, attempt.tally, f"attempt {parent}"):
            return why
        parent = by_number[parent].revises
    return None


# -- gap verdicts (I6) ------------------------------------------------------------------

REFUTED, STANDS, UNPROVEN = gaps.REFUTED, gaps.STANDS, gaps.UNPROVEN


def legal_verdicts(*, refutation_verified: bool, queries: int, papers: int) -> tuple[str, ...]:
    """The verdicts a Falsifier may give a gap: refuted needs a citation whose
    check passed, stands needs v1's effort floor (gaps.MIN_QUERIES queries over
    gaps.MIN_PAPERS papers), and unproven is always legal."""
    legal = [REFUTED] if refutation_verified else []
    if queries >= gaps.MIN_QUERIES and papers >= gaps.MIN_PAPERS:
        legal.append(STANDS)
    return (*legal, UNPROVEN)


# -- the Lead's actions -----------------------------------------------------------------

RESEARCH_STEPS = ("researcher", "skeptic")
LEAD_STEPS = (*RESEARCH_STEPS, "writer", "verifier")
MAX_QUEUE = 4
MAX_INSPECT = 4


def lead_action_refused(action: Next | Inspect | Final | Abstain, *, board: Collection[str],
                        attempts: Sequence[Attempt],
                        steps: Sequence[str] = LEAD_STEPS) -> str | None:
    """Why the Lead may not take `action` now, or None if it may. `board` holds the
    ids of the passages found so far, `attempts` what has been written, and
    `steps` the agents this run's Lead can send work to. Abstaining is always
    legal."""
    if isinstance(action, Next):
        return _queue_refused(action, board, attempts, steps)
    if isinstance(action, Inspect):
        if not 0 < len(action.ids) <= MAX_INSPECT:
            return f"INSPECT reads 1 to at most {MAX_INSPECT} passages at a time"
        return _off_board(action.ids, board)
    if isinstance(action, Final):
        if not attempts:
            return "nothing has been written yet, so there is no attempt to return"
        return return_refused(action.attempt, attempts)
    if isinstance(action, Abstain):
        return None
    raise TypeError(f"{type(action).__name__} is not a Lead action")


def _queue_refused(action: Next, board: Collection[str], attempts: Sequence[Attempt],
                   steps: Sequence[str]) -> str | None:
    unknown = [step for step in action.steps if step not in steps]
    if unknown:
        return f"{unknown[0]!r} is not a step; the steps are {', '.join(steps)}"
    if len(action.steps) > MAX_QUEUE:
        return f"a queue holds at most {MAX_QUEUE} steps"
    if not action.brief and any(step in RESEARCH_STEPS for step in action.steps):
        return "a researcher or skeptic step needs a BRIEF: line saying what to find"
    if action.passages and "writer" not in action.steps:
        return "PASSAGES: goes with a writer step"
    drafted = bool(attempts)
    for step in action.steps:
        if step == "writer" and not board:
            return "the board holds no passages to write from"
        if step == "verifier" and not drafted:
            return "the verifier has no draft to check yet"
        drafted = drafted or step == "writer"
    return _off_board(action.passages, board)


def _off_board(ids: Iterable[str], board: Collection[str]) -> str | None:
    off = [passage for passage in ids if passage not in board]
    return f"passage {off[0]} is not on the board" if off else None
