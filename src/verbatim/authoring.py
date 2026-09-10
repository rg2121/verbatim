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

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
import json
import re

from verbatim.corpus import CorpusStore
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
            for arxiv_id in row.get("expect_papers") or []:
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
