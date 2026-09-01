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

# The two reasons a question has no answer here. They are different problems
# and only the first is visible to retrieval, so scoring them together scores
# nothing: measured over this corpus, "what is the farthest star from the
# earth" matches no abstract at all, while "what is a tensor" matches 177 and
# is unanswerable only because none of them defines one.
NO_COVERAGE = "no_coverage"
NO_ANSWER = "no_answer"
UNANSWERABLE_KINDS = (NO_COVERAGE, NO_ANSWER)


@dataclass(frozen=True)
class EvalCase:
    """One question, and either the papers that answer it or nothing at all.

    A case with `expect_none` is a question the corpus cannot answer, and it
    must say which of the two kinds it is. `no_coverage` means no paper is
    about the subject, which retrieval can in principle see. `no_answer` means
    the subject is well covered but no paper answers the question - the corpus
    holds 177 papers whose abstracts mention tensors and defines none of them -
    which no retrieval signal distinguishes from a question that is answered.

    They were one label until measurement showed that every detector tried
    against them failed, because half the cases were undetectable by
    construction.
    """

    question: str
    expect_papers: tuple[str, ...] = ()
    expect_none: bool = False
    unanswerable_kind: str = ""
    note: str = ""

    @property
    def is_coverage_gap(self) -> bool:
        """Whether the corpus holds nothing on this subject at all."""
        return self.expect_none and self.unanswerable_kind == NO_COVERAGE

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
        kind = str(payload.get("unanswerable_kind", ""))
        if expect_none and kind not in UNANSWERABLE_KINDS:
            raise ValueError(
                f"case {payload.get('question')!r} must say why it is unanswerable: "
                f"unanswerable_kind is one of {', '.join(UNANSWERABLE_KINDS)}"
            )
        if kind and not expect_none:
            raise ValueError(
                f"case {payload.get('question')!r} sets unanswerable_kind but expects papers"
            )
        return cls(
            question=str(payload["question"]),
            expect_papers=tuple(str(p) for p in papers),
            expect_none=expect_none,
            unanswerable_kind=kind,
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
    reranker: object | None = None,
    complete: Callable[[str, str], str] | None = None,
    rewrites: int = 0,
    hyde: bool = False,
    decompose: bool = False,
) -> list[ModeScore]:
    """Score every retrieval mode on the same cases, so they are comparable.

    With a reranker, hybrid retrieves a wider shortlist and the reranker trims
    it, which is the only arrangement in which reranking could help.
    """
    def expanded(query: str) -> Sequence[ChunkMatch]:
        from verbatim.retrieval import expanded_search

        return expanded_search(
            corpus, embedder, query, complete=complete, rewrites=rewrites,
            hyde=hyde, limit=limit, max_per_paper=max_per_paper,
        )

    def hybrid(query: str) -> Sequence[ChunkMatch]:
        if reranker is None:
            return hybrid_search(corpus, embedder, query, limit=limit, max_per_paper=max_per_paper)
        from verbatim.reranking import rerank as _rerank

        wide = hybrid_search(corpus, embedder, query, limit=limit * 3, max_per_paper=max_per_paper)
        return _rerank(reranker, query, wide, limit=limit)

    modes: dict[str, Callable[[str], Sequence[ChunkMatch]]] = {
        "keyword-all": lambda q: corpus.search_chunks(q, limit=limit, match="all"),
        "keyword-any": lambda q: corpus.search_chunks(q, limit=limit, match="any"),
        "vector": lambda q: vector_search(corpus, embedder, q, limit=limit),
        "hybrid": hybrid,
        "hybrid+titles": lambda q: hybrid_search(
            corpus, embedder, q, limit=limit, max_per_paper=max_per_paper,
            use_titles=True,
        ),
        "hybrid+graph": lambda q: hybrid_search(
            corpus, embedder, q, limit=limit, max_per_paper=max_per_paper,
            use_graph=True,
        ),
    }
    # Only offered when a model is available to do the rewriting, so the
    # comparison is against the same corpus rather than a silently skipped leg.
    if complete is not None and (rewrites or hyde):
        modes["hybrid+expansion"] = expanded
    if complete is not None and decompose:
        from verbatim.research import gather_for_parts, plan_subquestions

        def decomposed(query: str) -> Sequence[ChunkMatch]:
            parts = plan_subquestions(complete, query)
            return gather_for_parts(
                corpus, embedder, parts,
                per_part=max(limit // max(len(parts), 1), 2),
                max_per_paper=max_per_paper,
            )[:limit]

        modes["hybrid+plan"] = decomposed
    return [score_mode(name, cases, retrieve) for name, retrieve in modes.items()]


def describe_case_mix(cases: Sequence[EvalCase]) -> str:
    """State what the answer key contains, split by what it is measuring.

    The two unanswerable kinds are reported apart because they are different
    problems: a coverage gap is visible to retrieval, and a covered-but-
    unanswered question is not. A single count hides which one a change moved.
    """
    coverage = sum(1 for case in cases if case.unanswerable_kind == NO_COVERAGE)
    unanswered = sum(1 for case in cases if case.unanswerable_kind == NO_ANSWER)
    answerable = sum(1 for case in cases if not case.expect_none)
    return (
        f"  {len(cases)} cases: {answerable} answerable, "
        f"{coverage} with no coverage, {unanswered} covered but unanswered"
    )


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
