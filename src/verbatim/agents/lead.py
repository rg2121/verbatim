"""The Lead: what it reads, and who decides for it.

Its view is split as the control contract says. FACTS are rendered by code: what
the opening passages are about, the passages on the board and how each was found
and cited, each attempt's counts and whether it may be returned (I10), the
passages it inspected, and the limits left. NOTES are what other agents wrote:
their opinion, under their name. Paper text is marked as data (I7).

A LeadPolicy makes the decision. AgentPolicy asks a model in the line protocol and
sends an unusable or illegal reply back once; the RulePolicy fixture (M3) will
apply v1's rules under the same contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from verbatim.agents.contract import (
    FAILED,
    PASSED,
    return_refused,
)
from verbatim.agents.protocol import ProtocolError, ask_twice, parse_lead
from verbatim.agents.roster import LEAD_SYSTEM
from verbatim.agents.state import Abstain, CitationRecord, Draft, Final, Inspect, Next, Note

Action = Next | Inspect | Final | Abstain



class LeadPolicy(Protocol):
    """Who decides the Lead's next action. `refuse` says why an action is not
    legal, or None; `ask` calls the Lead's model, checked against the budget."""

    def decide(self, view: str, refuse: Callable[[Action], str | None],
               ask: Callable[[str, str], str]) -> Action | None: ...


class AgentPolicy:
    """The Lead as a model replying in the line protocol. A reply it cannot use,
    or an illegal decision, goes back once with the reason; None after two."""

    def decide(self, view: str, refuse: Callable[[Action], str | None],
               ask: Callable[[str, str], str]) -> Action | None:
        def read(reply: str) -> Action:
            action = parse_lead(reply)
            if why := refuse(action):
                raise ProtocolError(why)
            return action

        return ask_twice(ask, LEAD_SYSTEM, view, read)[0]


def lead_view(question: str, *, subject: str | None, closeness: float | None,
              board: Sequence[Mapping[str, Any]], drafts: Sequence[Draft],
              records: Sequence[CitationRecord], inspected: Sequence[Mapping[str, Any]],
              notes: Sequence[Note], limits: str, steps: Sequence[str]) -> str:
    """What the Lead reads before each decision. `board` and `inspected` are
    passage rows (tools.passage_rows), each board row with how it was found."""
    shown = [f"QUESTION: {question}", "", "FACTS (computed by code):"]
    if subject:
        close = f"; closeness to the question {closeness:.2f}" if closeness is not None else ""
        shown.append(f'The opening passages are about: "{subject}"{close}. '
                     "Below 0.65 they are probably about another subject.")
    shown += ["", f"PASSAGES ON THE BOARD ({len(board)}):"]
    cited: dict[str, list[str]] = {}
    for record in records:
        result = "passed" if record.check == PASSED else "failed" if record.check == FAILED else "not checked"
        cited.setdefault(record.chunk_id, []).append(
            f"attempt {record.attempt} cites it as [{record.marker}]: {result}")
    for row in board:
        uses = "; ".join(cited.get(row["chunk_id"], ())) or "not cited yet"
        shown.append(f"- {row['chunk_id']}  {row['arxiv_id']}, {row['locator']}  "
                     f"found by {row['found']}  ({uses})")
    attempts = [draft.attempt for draft in drafts]
    shown += ["", "ATTEMPTS:" if drafts else "ATTEMPTS: none yet"]
    for draft in drafts:
        tally = draft.tally
        made = f"repair of attempt {draft.revises}" if draft.revises is not None else "draft"
        why = return_refused(draft.number, attempts)
        shown.append(f"- attempt {draft.number} ({made}): {_count(tally.verified, 'verified citation')}, "
                     f"{_count(tally.failed, 'failed check')}, {_count(tally.unbacked, 'unbacked sentence')}, "
                     f"{_count(tally.uncited, 'uncited claim')}; "
                     + ("it may be returned" if why is None else f"it may not be returned: {why}"))
    if inspected:
        shown += ["", "PASSAGES YOU INSPECTED:"]
        shown += [f'<passage id="{row["chunk_id"]}">{row["text"]}</passage>' for row in inspected]
    shown += ["", f"LIMITS: {limits}", "", "NOTES (written by other agents):" if notes else "NOTES: none"]
    shown += [f"- {note.agent}: {note.text}" for note in notes]
    shown += ["", f"STEPS: {', '.join(steps)}",
              "Reply with exactly one line: NEXT:, INSPECT:, FINAL: or ABSTAIN:."]
    return "\n".join(shown)


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'s' * (number != 1)}"
