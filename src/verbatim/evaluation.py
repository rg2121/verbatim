"""Measuring whether a retrieval change actually helps.

Every retrieval decision so far - hybrid fusion, the diversity cap, excluding
bibliographies - is correct by construction and unmeasured. This turns them
into numbers.

Relevance is recorded per *paper*, not per chunk: chunk ids change whenever the
chunker version does, so a chunk-level answer key would expire on every
re-chunk, while "this question is answered by that paper" stays true.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from verbatim.corpus import ChunkMatch, CorpusStore
from verbatim.embedding import Embedder
from verbatim.retrieval import hybrid_search, vector_search

CUTOFFS = (1, 5, 10)


@dataclass(frozen=True)
class EvalCase:
    """One question, and either the papers that answer it or nothing at all.

    A case with `expect_none` is a question the corpus genuinely cannot answer.
    Those matter as much as the rest: retrieval always returns something, and
    its scores do not separate a real hit from a coincidental one, so the only
    defence is the answer stage declining. Measuring that is measuring whether
    the tool can be trusted on the edge of its own coverage.
    """

    question: str
    expect_papers: tuple[str, ...] = ()
    expect_none: bool = False
    note: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> EvalCase:
        expect_none = bool(payload.get("expect_none", False))
        papers = payload.get("expect_papers") or []
        if not isinstance(papers, list):
            raise ValueError(f"case {payload.get('question')!r} has a malformed expect_papers")
        if not papers and not expect_none:
            raise ValueError(
                f"case {payload.get('question')!r} lists no expected papers; "
                "set expect_none if the corpus cannot answer it"
            )
        if papers and expect_none:
            raise ValueError(
                f"case {payload.get('question')!r} cannot both expect papers and expect none"
            )
        return cls(
            question=str(payload["question"]),
            expect_papers=tuple(str(p) for p in papers),
            expect_none=expect_none,
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True)
class ModeScore:
    """How one retrieval mode did across the whole eval set."""

    mode: str
    cases: int
    recall_at: dict[int, float]
    mrr: float
    misses: tuple[str, ...]

    def render(self) -> str:
        recalls = "  ".join(f"r@{k} {self.recall_at[k]:.2f}" for k in sorted(self.recall_at))
        return f"  {self.mode:<16} {recalls}  mrr {self.mrr:.3f}"


def load_cases(path: Path) -> list[EvalCase]:
    """Read an eval set from JSON Lines, skipping blank lines and comments."""
    cases: list[EvalCase] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            cases.append(EvalCase.from_dict(json.loads(stripped)))
        except (json.JSONDecodeError, KeyError, ValueError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    if not cases:
        raise ValueError(f"{path} contains no eval cases")
    return cases


def rank_of_first_hit(matches: Sequence[ChunkMatch], expected: Iterable[str]) -> int | None:
    """1-based rank of the first result from an expected paper, or None."""
    wanted = set(expected)
    for rank, match in enumerate(matches, start=1):
        if match.chunk.arxiv_id in wanted:
            return rank
    return None


def score_mode(
    mode: str,
    cases: Sequence[EvalCase],
    retrieve: Callable[[str], Sequence[ChunkMatch]],
    *,
    cutoffs: Sequence[int] = CUTOFFS,
) -> ModeScore:
    """Run every case through one retrieval mode and summarise it.

    Recall@k here means "at least one result in the top k came from a paper
    that answers the question", which is what matters when the answer is then
    written from those passages.
    """
    answerable = [case for case in cases if not case.expect_none]
    hits = {k: 0 for k in cutoffs}
    reciprocal = 0.0
    misses: list[str] = []
    for case in answerable:
        rank = rank_of_first_hit(retrieve(case.question), case.expect_papers)
        if rank is None:
            misses.append(case.question)
            continue
        reciprocal += 1.0 / rank
        for k in cutoffs:
            if rank <= k:
                hits[k] += 1
    total = len(answerable) or 1
    return ModeScore(
        mode=mode,
        cases=len(answerable),
        recall_at={k: hits[k] / total for k in cutoffs},
        mrr=reciprocal / total,
        misses=tuple(misses),
    )


def evaluate(
    corpus: CorpusStore,
    embedder: Embedder,
    cases: Sequence[EvalCase],
    *,
    limit: int = 10,
    max_per_paper: int | None = None,
) -> list[ModeScore]:
    """Score every retrieval mode on the same cases, so they are comparable."""
    modes: dict[str, Callable[[str], Sequence[ChunkMatch]]] = {
        "keyword-all": lambda q: corpus.search_chunks(q, limit=limit, match="all"),
        "keyword-any": lambda q: corpus.search_chunks(q, limit=limit, match="any"),
        "vector": lambda q: vector_search(corpus, embedder, q, limit=limit),
        "hybrid": lambda q: hybrid_search(
            corpus, embedder, q, limit=limit, max_per_paper=max_per_paper
        ),
    }
    return [score_mode(name, cases, retrieve) for name, retrieve in modes.items()]


def render_report(scores: Sequence[ModeScore]) -> str:
    """Lay the modes out together, best mrr last so the winner is obvious."""
    lines = [f"  {'mode':<16} recall@1        recall@5        recall@10   mrr"]
    for score in sorted(scores, key=lambda s: s.mrr):
        lines.append(score.render())
    best = max(scores, key=lambda s: s.mrr)
    lines.append(f"\n  best by mrr: {best.mode}  ({best.cases} cases)")
    if best.misses:
        lines.append(f"  {len(best.misses)} question(s) it found nothing for:")
        lines.extend(f"    - {question}" for question in best.misses[:5])
    return "\n".join(lines)
