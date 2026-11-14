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

from verbatim import gaps, reflection
from verbatim.agents.state import Abstain, Attempt, Final, Inspect, Next, Tally, Verification
from verbatim.answers import Answer
from verbatim.critique import CITED, unbacked_sentences, uncited_sentences
from verbatim.quotations import mask_quoted_brackets

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


def verification_refused(verification: Verification,
                         checks: Mapping[int, Sequence[str]]) -> str | None:
    """Why the Verifier may not decide this, or None. `checks` holds, by marker,
    the check result of each CITE line under it: a decision must be legal for
    every one. A citation sent for repair needs the draft sent back."""
    for decided in verification.decisions:
        if decided.marker not in checks:
            return f"there is no citation [{decided.marker}] in this draft"
        for result in checks[decided.marker]:
            if why := decision_refused(result, decided.decision):
                return f"[{decided.marker}]: {why}"
    if not verification.repair and any(d.decision == REPAIR for d in verification.decisions):
        return "a citation marked repair needs NEXT: writer, which sends the draft back"
    return None


def _either(options: Sequence[str]) -> str:
    return ", ".join(options[:-1]) + f" or {options[-1]}"


# -- attempts (I10) ---------------------------------------------------------------------


def tally(answer: Answer, texts: Mapping[str, str]) -> Tally:
    """What the check found in an attempt; `texts` holds each passage's text by
    chunk id as the check reads it (reflection.passage_texts).

    - verified: the citations a marker in the text points at; one that no
      marker points at supports nothing.
    - failed, as v1's problem_count counts them: each quote the check rejected,
      and each [unverified] marker the text shows, every claim a failed or
      missing citation leaves unsupported; and each quotation in the prose that
      the passage its marker names does not hold.
    - unbacked and uncited: v1's sentence checks, on every attempt. One that
      cites nothing, as v1's Writer declines, still makes the claims it makes.
    """
    markers = {int(number) for number in CITED.findall(mask_quoted_brackets(answer.text))}
    used = [citation for number, citation in enumerate(answer.citations, start=1)
            if number in markers]
    misquoted = [problem for problem in answer.quotation_problems if problem.marker is not None]
    return Tally(
        verified=len(used),
        failed=len(answer.rejected) + answer.text.count(reflection.UNVERIFIED) + len(misquoted),
        unbacked=len(unbacked_sentences(answer, dict(texts))),
        uncited=len(uncited_sentences(answer.text)),
        passages=tuple(sorted({citation.chunk_id for citation in used})),
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
    """Why attempt `number` may not be returned, or None (I10)."""
    reasons = _reasons(attempts)
    return reasons[number] if number in reasons else f"there is no attempt {number}"


def legal_returns(attempts: Sequence[Attempt]) -> tuple[int, ...]:
    """The attempts that may be returned, in the order written. Never empty while
    any exist: a first draft revises nothing, so it is never refused, and of the
    rest some attempt is always undominated."""
    reasons = _reasons(attempts)
    return tuple(attempt.number for attempt in attempts if reasons[attempt.number] is None)


def _reasons(attempts: Sequence[Attempt]) -> dict[int, str | None]:
    """Each attempt's reason it may not be returned, or None. A revision that
    breaks v1's refusals against any attempt it descends from may not; of the
    rest, none that another of the rest dominates."""
    by_number = {attempt.number: attempt for attempt in attempts}
    if len(by_number) != len(attempts):
        raise ValueError("two attempts share a number")
    reasons = {attempt.number: _refused(attempt, by_number) for attempt in attempts}
    standing = [attempt for attempt in attempts if reasons[attempt.number] is None]
    for attempt in standing:
        better = next((other for other in standing if dominates(other.tally, attempt.tally)), None)
        if better is not None:
            reasons[attempt.number] = (f"attempt {better.number} is at least as good on every "
                                       "count and better on one")
    return reasons


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
    gaps.MIN_PAPERS papers) and no such citation, as in v1, and unproven is
    always legal."""
    if refutation_verified:
        return (REFUTED, UNPROVEN)
    if queries >= gaps.MIN_QUERIES and papers >= gaps.MIN_PAPERS:
        return (STANDS, UNPROVEN)
    return (UNPROVEN,)


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
    if not action.steps:
        return "a queue needs at least one step"
    if len(action.steps) > MAX_QUEUE:
        return f"a queue holds at most {MAX_QUEUE} steps"
    if not action.brief and any(step in RESEARCH_STEPS for step in action.steps):
        return "a researcher or skeptic step needs a BRIEF: line saying what to find"
    if action.passages and "writer" not in action.steps:
        return "PASSAGES: goes with a writer step"
    found, drafted = bool(board), bool(attempts)
    for step in action.steps:
        if step == "writer" and not found:
            return ("a writer step has no passages to write from: none are on the board, "
                    "and no research step comes before it")
        if step == "verifier" and not drafted:
            return "the verifier has no draft to check yet"
        found = found or step in RESEARCH_STEPS
        drafted = drafted or step == "writer"
    return _off_board(action.passages, board)


def _off_board(ids: Iterable[str], board: Collection[str]) -> str | None:
    off = [passage for passage in ids if passage not in board]
    return f"passage {off[0]} is not on the board" if off else None
