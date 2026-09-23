"""Authoring held-out evaluation cases without letting the system see them.

The development key in `evals/retrieval.jsonl` was written alongside the
retrieval it scores, so a question that retrieved badly could be reworded
until it did not. Its numbers cannot rule that out, and the gap is measured:
0.869 on the development key against 0.758 held out, on the same system.

So a held-out case is authored under one rule - **from the paper's title and
abstract only, with no retrieval run against the question until the file is
committed.** This module exists to keep that rule mechanical rather than
remembered. It imports nothing from `retrieval` or `vectors`, and a test
asserts that it never will: a module that cannot search cannot be tempted to
check whether a question retrieves well before keeping it.

Generating questions with a model adds one hazard hand-authoring did not
have. A person paraphrases; a model reaches for the abstract's own words, and
abstracts are passages in this corpus, so a question sharing a distinctive
span with one is found by keyword search for that reason alone. Measured
against the 88 hand-authored cases, a question shares a median run of 2 words
with its paper's abstract and at most 6; only 4 of the 88 share 5 or more.
Cases sharing 5 or more are therefore rejected here, which makes a generated
set no leakier than the hand-authored one, and slightly stricter.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
import json
import re

from verbatim.conversation import refers_back
from verbatim.corpus import CorpusStore, split_arxiv_version
from verbatim.distill import bibliographic, standalone

# Set from the hand-authored key rather than chosen: see the module docstring.
MAX_SHARED_RUN = 5

# The band the original held-out papers were drawn from. A paper with a
# handful of passages is a note, and one with hundreds is a survey whose
# abstract promises more than any single passage delivers.
MIN_PASSAGES = 25
MAX_PASSAGES = 90

QUESTION_SYSTEM = """You are given the title and abstract of a research paper. Write ONE question that this paper answers.

Rules:
- The question must be answerable from what the abstract claims.
- Use your own words. Do not reuse phrases from the abstract; someone must be
  able to ask this without having read it.
- Name the subject concretely, so the question stands alone. Write "what limits
  the coherence time of transmon qubits", never "what does the method improve".
- Do not refer to the paper, the authors, the abstract or the figures.
- Ask about the research, never about who wrote it, where they work or who
  funded it.

Reply with exactly one line, and nothing else:

Q: <question>"""

QUESTION_LINE = re.compile(r"^\s*Q:\s*(.+?)\s*$", re.MULTILINE)
WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Case:
    """One held-out question and the paper it was written from."""

    question: str
    arxiv_id: str
    shared_run: int

    def to_row(self) -> str:
        return json.dumps({
            "question": self.question,
            "expect_papers": [self.arxiv_id],
            "note": f"source=heldout-generated; shape=plain; shared_run={self.shared_run}",
        })


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def longest_shared_run(question: str, abstract: str) -> int:
    """The longest run of consecutive words the two have in common.

    Runs rather than bag-of-words overlap: a question and an abstract about
    the same subject necessarily share vocabulary, and penalising that would
    reject every real question. What matters is borrowed *phrasing*, which is
    what a keyword index rewards.
    """
    asked, given = words(question), words(abstract)
    if not asked or not given:
        return 0
    grams = lambda source, n: {
        tuple(source[i : i + n]) for i in range(len(source) - n + 1)
    }
    longest = 0
    for size in range(1, min(len(asked), len(given)) + 1):
        if grams(asked, size) & grams(given, size):
            longest = size
        else:
            break
    return longest


def usable_case(question: str, abstract: str) -> bool:
    """Whether a generated question may join the frozen set."""
    if not standalone(question) or bibliographic(question):
        return False
    return longest_shared_run(question, abstract) < MAX_SHARED_RUN


def excluded_papers(paths: Iterable[str]) -> set[str]:
    """Every paper already named by an existing key, so sets never overlap.

    A paper scored in the development key cannot also be held out: the system
    was tuned while looking at it.
    """
    seen: set[str] = set()
    for path in paths:
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Both shapes of key name their papers: `expect_papers` is "any of
            # these", `require_papers` "all of these". A paper in either has
            # been scored already and cannot also be held out.
            for arxiv_id in (row.get("expect_papers") or []) + (row.get("require_papers") or []):
                seen.add(str(arxiv_id))
                seen.add(str(arxiv_id).split("v")[0])
    return seen


def candidate_papers(
    corpus: CorpusStore,
    *,
    exclude: set[str],
    limit: int,
    min_passages: int = MIN_PASSAGES,
    max_passages: int = MAX_PASSAGES,
    stride: int = 7,
) -> list[tuple[str, str, str]]:
    """Papers to author from, chosen before any question exists.

    Returned as (arxiv_id, title, abstract). The selection is a fixed stride
    through papers grouped by primary category, so the set spans the corpus
    rather than its largest category, and so the same call returns the same
    papers - a frozen key that changes when it is regenerated is not frozen.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    rows = corpus.connection.execute(
        """
        SELECT p.arxiv_id, p.title, p.summary, p.categories_json,
               COUNT(c.chunk_id) AS passages
        FROM papers AS p
        JOIN chunks AS c ON c.arxiv_id = p.arxiv_id AND c.kind = 'body'
        WHERE p.is_latest = 1 AND p.summary IS NOT NULL AND p.summary != ''
        GROUP BY p.arxiv_id
        HAVING passages BETWEEN ? AND ?
        ORDER BY p.arxiv_id
        """,
        (min_passages, max_passages),
    ).fetchall()

    by_category: dict[str, list[tuple[str, str, str]]] = {}
    for row in rows:
        arxiv_id = row["arxiv_id"]
        if arxiv_id in exclude or arxiv_id.split("v")[0] in exclude:
            continue
        try:
            primary = (json.loads(row["categories_json"]) or ["unknown"])[0]
        except (json.JSONDecodeError, TypeError):
            primary = "unknown"
        by_category.setdefault(primary, []).append(
            (arxiv_id, row["title"], row["summary"])
        )

    # Round-robin across categories, taking every `stride`-th paper within
    # each, so no category can fill the set by being large.
    picked: list[tuple[str, str, str]] = []
    strided = {
        category: papers[::stride] for category, papers in sorted(by_category.items())
    }
    position = 0
    while len(picked) < limit and any(
        position < len(papers) for papers in strided.values()
    ):
        for _category, papers in strided.items():
            if position < len(papers):
                picked.append(papers[position])
                if len(picked) == limit:
                    break
        position += 1
    return picked


def question_for(
    complete: Callable[[str, str], str], title: str, abstract: str
) -> str | None:
    """One question written from the title and abstract, or None."""
    reply = complete(
        QUESTION_SYSTEM, f"TITLE: {title}\n\nABSTRACT: {abstract}"
    )
    found = QUESTION_LINE.search(reply or "")
    return found.group(1).strip() if found else None


def author(
    corpus: CorpusStore,
    complete: Callable[[str, str], str],
    *,
    cases: int,
    exclude: set[str],
    rejected: dict[str, int] | None = None,
    stride: int = 7,
) -> Iterator[Case]:
    """Yield held-out cases until `cases` are written or candidates run out."""
    if cases < 1:
        raise ValueError("cases must be at least 1")
    counts = rejected if rejected is not None else {}
    bump = lambda reason: counts.__setitem__(reason, counts.get(reason, 0) + 1)
    written = 0
    # Over-draw: most candidates survive, but a run that quietly returns fewer
    # cases than asked for is the failure this project already made once.
    for arxiv_id, title, abstract in candidate_papers(
        corpus, exclude=exclude, limit=cases * 3, stride=stride
    ):
        question = question_for(complete, title, abstract)
        if question is None:
            bump("no question written")
            continue
        if not standalone(question):
            bump("does not stand alone")
            continue
        if bibliographic(question):
            bump("about the paper's cover")
            continue
        run = longest_shared_run(question, abstract)
        if run >= MAX_SHARED_RUN:
            bump(f"borrowed a {run}-word phrase")
            continue
        written += 1
        yield Case(question=question, arxiv_id=arxiv_id, shared_run=run)
        if written >= cases:
            return


def render_rejections(counts: dict[str, int]) -> str:
    if not counts:
        return "  nothing rejected"
    return "\n".join(
        f"    rejected, {reason:<28} {count}"
        for reason, count in sorted(counts.items(), key=lambda kv: -kv[1])
    )


# --- two-paper questions, for the breadth evaluation ----------------------
#
# Held-out cases are written from one abstract, so they reward finding one
# paper. The team exists for questions that need several, and the only honest
# way to test it on those is to write them under the same blind rule: titles
# and abstracts only, no retrieval, frozen before anything runs. See
# evals/breadth/PROTOCOL.md, which fixed all of this before any question existed.

PAIR_KINDS: tuple[str, ...] = ("comparison", "limitation", "synthesis")

KIND_RULES = {
    "comparison": (
        "Ask how the two approaches differ, and on what: method, assumptions, "
        "results or cost."
    ),
    "limitation": (
        "Ask what limitation of one approach the other exposes or addresses, "
        "and at what cost."
    ),
    "synthesis": (
        "Ask what approaches exist to the problem the two share, in a way that "
        "needs both to answer."
    ),
}

PAIR_SYSTEM = """You are given the titles and abstracts of two related research papers.

Write ONE question that can only be answered well by drawing on BOTH papers.

{kind_rule}

Rules:
- Use your own words. Do not reuse phrases from either abstract.
- Name the subjects concretely, so the question stands alone.
- Do not mention the papers, the authors, the abstracts, or which paper said what.
- Ask about the research, never about who wrote it, where they work or who
  funded it.

Reply with exactly one line, and nothing else:

Q: <question>"""


@dataclass(frozen=True)
class PairCase:
    """One breadth question, and the two papers it needs."""

    question: str
    kind: str
    arxiv_ids: tuple[str, str]
    shared_run: int

    def to_row(self) -> str:
        # require_papers alone: the key format reads it as "all of these", and
        # refuses a case that also lists expect_papers ("any of these"), since
        # the two contradict each other.
        return json.dumps({
            "question": self.question,
            "require_papers": list(self.arxiv_ids),
            "note": (
                f"source=breadth; shape={self.kind}; "
                f"shared_run={self.shared_run}"
            ),
        })


def candidate_pairs(
    corpus: CorpusStore,
    *,
    exclude: set[str],
    limit: int,
    min_passages: int = MIN_PASSAGES,
    max_passages: int = MAX_PASSAGES,
    stride: int = 3,
) -> list[tuple[tuple[str, str, str], tuple[str, str, str]]]:
    """Pairs of related papers, chosen before any question exists.

    Related by the corpus's own record: one holds the other in its
    bibliography. That keeps "these two bear on each other" out of anyone's
    judgement, a model's included. Each pair is ((id, title, abstract) of the
    cited paper, the same for the citing one), taken in a fixed order at a
    fixed stride, with no paper in two pairs - a paper reused across questions
    would weight its subject twice.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    rows = corpus.connection.execute(
        """
        WITH band AS MATERIALIZED (
            SELECT p.arxiv_id, p.base_arxiv_id, p.title, p.summary
            FROM papers AS p
            JOIN chunks AS c ON c.arxiv_id = p.arxiv_id AND c.kind = 'body'
            WHERE p.is_latest = 1 AND p.summary IS NOT NULL AND p.summary != ''
            GROUP BY p.arxiv_id
            HAVING COUNT(c.chunk_id) BETWEEN ? AND ?
        )
        SELECT DISTINCT
            cited.arxiv_id AS cited_id, cited.title AS cited_title,
            cited.summary AS cited_summary,
            citing.arxiv_id AS citing_id, citing.title AS citing_title,
            citing.summary AS citing_summary
        FROM paper_references AS r
        JOIN band AS citing ON citing.arxiv_id = r.citing_arxiv_id
        JOIN band AS cited ON cited.base_arxiv_id = r.cited_arxiv_id
        WHERE citing.base_arxiv_id != cited.base_arxiv_id
        ORDER BY citing.arxiv_id, cited.arxiv_id
        """,
        (min_passages, max_passages),
    ).fetchall()

    def excluded(arxiv_id: str) -> bool:
        return arxiv_id in exclude or split_arxiv_version(arxiv_id)[0] in exclude

    usable = [
        row for row in rows
        if not (excluded(row["cited_id"]) or excluded(row["citing_id"]))
    ]
    picked: list[tuple[tuple[str, str, str], tuple[str, str, str]]] = []
    used: set[str] = set()
    for row in usable[::stride]:
        first, second = row["cited_id"], row["citing_id"]
        if first in used or second in used:
            continue
        used.update((first, second))
        picked.append((
            (first, row["cited_title"], row["cited_summary"]),
            (second, row["citing_title"], row["citing_summary"]),
        ))
        if len(picked) == limit:
            break
    return picked


def question_for_pair(
    complete: Callable[[str, str], str],
    kind: str,
    first: tuple[str, str, str],
    second: tuple[str, str, str],
) -> str | None:
    """One question needing both papers, written from their abstracts only."""
    if kind not in KIND_RULES:
        raise ValueError(f"unknown kind {kind!r}; known: {', '.join(PAIR_KINDS)}")
    reply = complete(
        PAIR_SYSTEM.format(kind_rule=KIND_RULES[kind]),
        f"PAPER ONE\nTITLE: {first[1]}\nABSTRACT: {first[2]}\n\n"
        f"PAPER TWO\nTITLE: {second[1]}\nABSTRACT: {second[2]}",
    )
    found = QUESTION_LINE.search(reply or "")
    return found.group(1).strip() if found else None


def author_pairs(
    corpus: CorpusStore,
    complete: Callable[[str, str], str],
    *,
    per_kind: int,
    exclude: set[str],
    rejected: dict[str, int] | None = None,
    stride: int = 3,
) -> Iterator[PairCase]:
    """Yield breadth questions until every kind has `per_kind`, or pairs run out."""
    if per_kind < 1:
        raise ValueError("per_kind must be at least 1")
    counts = rejected if rejected is not None else {}
    bump = lambda reason: counts.__setitem__(reason, counts.get(reason, 0) + 1)
    wanted = {kind: per_kind for kind in PAIR_KINDS}
    for first, second in candidate_pairs(
        corpus, exclude=exclude, limit=per_kind * len(PAIR_KINDS) * 4, stride=stride,
    ):
        open_kinds = [kind for kind in PAIR_KINDS if wanted[kind]]
        if not open_kinds:
            return
        # The kind still needing the most goes next, so every kind fills at the
        # same pace and no kind is starved by rejections in the others.
        kind = max(open_kinds, key=lambda k: wanted[k])
        question = question_for_pair(complete, kind, first, second)
        if question is None:
            bump("no question written")
            continue
        if not standalone(question):
            bump("does not stand alone")
            continue
        if bibliographic(question):
            bump("about the paper's cover")
            continue
        run = max(
            longest_shared_run(question, first[2]),
            longest_shared_run(question, second[2]),
        )
        if run >= MAX_SHARED_RUN:
            bump(f"borrowed a {run}-word phrase")
            continue
        wanted[kind] -= 1
        yield PairCase(
            question=question, kind=kind,
            arxiv_ids=(first[0], second[0]), shared_run=run,
        )


# --- two-turn questions, for measuring follow-ups -------------------------
#
# Every key here is single-turn, so "did the follow-up still find the right
# paper" has no measurement. These cases are pairs: a question that stands on
# its own, and a follow-up that only makes sense after it. Both are answered
# by the same paper, so retrieval for the second turn can be scored exactly as
# the first - which is the point, since a follow-up is where retrieval loses
# the subject. Written under the same blind rule: titles and abstracts only,
# no retrieval, frozen before anything is scored.

CONVERSATION_SYSTEM = """You are given the title and abstract of a research paper.

Write two questions a reader would ask about it, in order, as a short conversation.

Rules:
- The FIRST question must stand on its own: name the subject concretely, so
  someone could search for it having read nothing else.
- The SECOND must be a follow-up that only makes sense after the first. Refer
  back to the subject with "it", "they", "that approach" and so on, and do NOT
  name the subject again.
- Both must be answerable from what the abstract claims.
- Use your own words. Do not reuse phrases from the abstract.
- Ask about the research, never about who wrote it, where they work or who
  funded it.

Reply with exactly two lines, and nothing else:

Q1: <question>
Q2: <follow-up>"""

FIRST_LINE = re.compile(r"^\s*Q1:\s*(.+?)\s*$", re.MULTILINE)
FOLLOW_UP_LINE = re.compile(r"^\s*Q2:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ConversationCase:
    """One opening question, one follow-up, and the paper that answers both."""

    question: str
    follow_up: str
    arxiv_id: str
    shared_run: int

    def to_row(self) -> str:
        return json.dumps({
            "question": self.question,
            "follow_up": self.follow_up,
            "expect_papers": [self.arxiv_id],
            "note": f"source=conversational; shape=follow-up; shared_run={self.shared_run}",
        })


def usable_follow_up(follow_up: str) -> bool:
    """Whether a second turn genuinely depends on the first.

    A follow-up that names its own subject measures nothing: retrieval would
    find the paper from that turn alone and the conversation would never be
    consulted. So it has to point backwards - which is the same test the web
    page uses to decide whether to resolve a question at all, and a question
    naming its subject fails it.

    Not `standalone()`: that rejects a question pointing at *its source* ("what
    does this paper show"), which is a different fault and says nothing about
    whether a turn depends on the one before it.
    """
    return refers_back(follow_up) and not bibliographic(follow_up)


def author_conversations(
    corpus: CorpusStore,
    complete: Callable[[str, str], str],
    *,
    cases: int,
    exclude: set[str],
    rejected: dict[str, int] | None = None,
    stride: int = 5,
) -> Iterator[ConversationCase]:
    """Yield two-turn cases until `cases` are written or candidates run out."""
    if cases < 1:
        raise ValueError("cases must be at least 1")
    counts = rejected if rejected is not None else {}
    bump = lambda reason: counts.__setitem__(reason, counts.get(reason, 0) + 1)
    written = 0
    for arxiv_id, title, abstract in candidate_papers(
        corpus, exclude=exclude, limit=cases * 3, stride=stride
    ):
        reply = complete(CONVERSATION_SYSTEM, f"TITLE: {title}\n\nABSTRACT: {abstract}")
        opening = FIRST_LINE.search(reply or "")
        follow_up = FOLLOW_UP_LINE.search(reply or "")
        if opening is None or follow_up is None:
            bump("no exchange written")
            continue
        question, second = opening.group(1), follow_up.group(1)
        if not standalone(question) or bibliographic(question):
            bump("opening does not stand alone")
            continue
        if not usable_follow_up(second):
            bump("follow-up does not depend on the opening")
            continue
        run = max(
            longest_shared_run(question, abstract),
            longest_shared_run(second, abstract),
        )
        if run >= MAX_SHARED_RUN:
            bump(f"borrowed a {run}-word phrase")
            continue
        written += 1
        yield ConversationCase(
            question=question, follow_up=second, arxiv_id=arxiv_id, shared_run=run,
        )
        if written >= cases:
            return

