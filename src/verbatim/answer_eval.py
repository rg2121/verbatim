"""Answer-level evaluation: is the answer supported, not just the right paper.

Most of it is counted without a model: declines, verified citations, uncited
sentences, rejected quotes. Whether a quote supports its claim needs a judge,
so each real (claim, quote) pair is scored beside control pairs whose quote
cannot support the claim. A judge that accepts both tells us nothing, and the
report says so.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import random
import re

from verbatim.answers import Answer
from verbatim.citations import Citation
from verbatim.critique import MARKER, SENTENCE, uncited_sentences

# Forced choice rather than yes/no, so "unrelated" is an answer, not a refusal.
SUPPORT_SYSTEM = """You are given a CLAIM and a QUOTE taken from a research paper.

Say how the quote relates to the claim. Reply with exactly one word:

STATES     - the quote says the claim, in its own words
IMPLIES    - the quote does not say it but makes it follow
UNRELATED  - the quote is about something else, or does not bear on the claim
CONTRADICTS - the quote says the opposite

Many pairs are UNRELATED. Do not try to find a connection that is not there.
Reply with one word and nothing else."""
SUPPORTING = {"STATES", "IMPLIES"}
VERDICT = re.compile(r"\b(STATES|IMPLIES|UNRELATED|CONTRADICTS)\b", re.IGNORECASE)


@dataclass(frozen=True)
class AnsweredCase:
    """One question put to the system, and what came back."""

    question: str
    answer: Answer
    expected_papers: tuple[str, ...] = ()

    @property
    def cited_papers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c.arxiv_id for c in self.answer.citations))

    @property
    def cited_an_expected_paper(self) -> bool:
        return bool(set(self.cited_papers) & set(self.expected_papers))


@dataclass(frozen=True)
class AnswerScore:
    """What the answers were like, counted."""

    cases: int
    declined: int
    with_verified_citation: int
    fully_cited: int
    cited_expected_paper: int
    rejected_citations: int

    def render(self) -> str:
        def share(n: int) -> str:
            return f"{n}/{self.cases} ({n / max(self.cases, 1):.0%})"
        return "\n".join([
            f"  {self.cases} questions answered",
            f"    declined outright              {share(self.declined)}",
            f"    at least one verified citation {share(self.with_verified_citation)}",
            f"    every claim carries a marker   {share(self.fully_cited)}",
            f"    cited the expected paper       {share(self.cited_expected_paper)}",
            f"    citations rejected in total    {self.rejected_citations}",
        ])


@dataclass(frozen=True)
class JudgeReport:
    """How the judge scored real pairs against two controls.

    The easy control borrows a quote from another question's answer. The hard
    control borrows one from the same answer, so the subject matches but the
    quote still does not support the claim.
    """

    real_pairs: int
    real_supported: int
    control_pairs: int
    control_supported: int
    hard_pairs: int = 0
    hard_supported: int = 0

    @property
    def discriminates(self) -> bool:
        """Whether real pairs score clearly above the hard control, if there is one."""
        against = self.hard_rate if self.hard_pairs else self.control_rate
        return (self.real_rate - against) > 0.15

    @property
    def real_rate(self) -> float:
        return self.real_supported / max(self.real_pairs, 1)

    @property
    def control_rate(self) -> float:
        return self.control_supported / max(self.control_pairs, 1)

    @property
    def hard_rate(self) -> float:
        return self.hard_supported / max(self.hard_pairs, 1)

    def render(self) -> str:
        lines = [
            f"  real pairs                       {self.real_supported}/{self.real_pairs} "
            f"supported ({self.real_rate:.0%})",
            f"  hard control (same answer)       {self.hard_supported}/{self.hard_pairs} "
            f"supported ({self.hard_rate:.0%})",
            f"  easy control (another question)  {self.control_supported}/{self.control_pairs} "
            f"supported ({self.control_rate:.0%})",
        ]
        against = "hard control" if self.hard_pairs else "control"
        margin = self.real_rate - (self.hard_rate if self.hard_pairs else self.control_rate)
        if self.discriminates:
            lines.append(
                f"  the judge separates real pairs from the {against} by "
                f"{margin:.0%}, so the real rate above means something"
            )
        else:
            lines.append(
                f"  the judge does not separate a real pair from the {against}, "
                "so its verdicts carry no information and the real rate above "
                "should not be quoted"
            )
        return "\n".join(lines)


def claim_quote_pairs(case: AnsweredCase) -> list[tuple[str, Citation]]:
    """Each cited sentence paired with the citation its marker points at."""
    pairs: list[tuple[str, Citation]] = []
    for sentence in SENTENCE.split(case.answer.text.strip()):
        for marker in MARKER.findall(sentence):
            if not marker.strip("[]").isdigit():
                continue
            position = int(marker.strip("[]"))
            if 1 <= position <= len(case.answer.citations):
                pairs.append((sentence.strip(), case.answer.citations[position - 1]))
    return pairs


def score_answers(cases: Sequence[AnsweredCase]) -> AnswerScore:
    """The countable half. No model is asked anything here."""
    return AnswerScore(
        cases=len(cases),
        declined=sum(1 for c in cases if c.answer.declined),
        with_verified_citation=sum(1 for c in cases if c.answer.citations),
        fully_cited=sum(
            1 for c in cases
            if c.answer.citations and not uncited_sentences(c.answer.text)
        ),
        cited_expected_paper=sum(1 for c in cases if c.cited_an_expected_paper),
        rejected_citations=sum(len(c.answer.rejected) for c in cases),
    )


def judge_support(judge: Callable[[str, str], str], claim: str, quote: str) -> str:
    """One forced-choice verdict, defaulting to UNRELATED when unreadable."""
    try:
        reply = judge(SUPPORT_SYSTEM, f"CLAIM: {claim}\n\nQUOTE: {quote}")
    except (RuntimeError, OSError):
        return "UNRELATED"
    found = VERDICT.search(reply or "")
    return found.group(1).upper() if found else "UNRELATED"


def judge_with_control(
    cases: Sequence[AnsweredCase],
    judge: Callable[[str, str], str],
    *,
    seed: int = 0,
) -> JudgeReport:
    """Score the real (claim, quote) pairs and their mismatched controls."""
    # Track each pair's answer so a control never borrows from its own answer.
    real: list[tuple[int, str, str]] = []
    quotes_by_case: dict[int, list[str]] = {}
    for index, case in enumerate(cases):
        for claim, citation in claim_quote_pairs(case):
            real.append((index, claim, citation.quoted_text))
        if case.answer.citations:
            quotes_by_case[index] = [c.quoted_text for c in case.answer.citations]
    if not real:
        return JudgeReport(0, 0, 0, 0)

    rng = random.Random(seed)
    control: list[tuple[str, str]] = []
    hard: list[tuple[str, str]] = []
    for owner, claim, quote in real:
        elsewhere = [index for index in quotes_by_case if index != owner]
        if elsewhere:
            borrowed = quotes_by_case[rng.choice(elsewhere)]
            control.append((claim, rng.choice(borrowed)))
        # Hard control: another quote from the same answer.
        siblings = [q for q in quotes_by_case.get(owner, []) if q != quote]
        if siblings:
            hard.append((claim, rng.choice(siblings)))

    real_supported = sum(
        1 for _owner, claim, quote in real
        if judge_support(judge, claim, quote) in SUPPORTING
    )
    control_supported = sum(
        1 for claim, quote in control if judge_support(judge, claim, quote) in SUPPORTING
    )
    hard_supported = sum(
        1 for claim, quote in hard if judge_support(judge, claim, quote) in SUPPORTING
    )
    return JudgeReport(
        len(real), real_supported, len(control), control_supported,
        len(hard), hard_supported,
    )
