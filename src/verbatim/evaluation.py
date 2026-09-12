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
import random
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
    require_papers: tuple[str, ...] = ()

    @property
    def is_coverage_gap(self) -> bool:
        """Whether the corpus holds nothing on this subject at all."""
        return self.expect_none and self.unanswerable_kind == NO_COVERAGE

    @property
    def is_compound(self) -> bool:
        """Whether this question needs every listed paper, not any one of them.

        `expect_papers` lists alternatives and is scored by first hit;
        `require_papers` lists parts of one answer - "compare A and B" is not
        answered by ten excellent passages about A - and is scored by
        coverage. Each entry may be an any-of group, written "idA|idB", for a
        part that more than one held paper can supply.
        """
        return bool(self.require_papers)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> EvalCase:
        expect_none = bool(payload.get("expect_none", False))
        papers = payload.get("expect_papers") or []
        required = payload.get("require_papers") or []
        if not isinstance(papers, list):
            raise ValueError(f"case {payload.get('question')!r} has a malformed expect_papers")
        if not isinstance(required, list):
            raise ValueError(f"case {payload.get('question')!r} has a malformed require_papers")
        if required and papers:
            raise ValueError(
                f"case {payload.get('question')!r} cannot both require papers and "
                "list alternatives; require_papers means all, expect_papers means any"
            )
        if required and expect_none:
            raise ValueError(
                f"case {payload.get('question')!r} cannot both require papers and expect none"
            )
        if required and len(required) < 2:
            raise ValueError(
                f"case {payload.get('question')!r} requires a single paper, "
                "which is what expect_papers already measures"
            )
        if not papers and not required and not expect_none:
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
            require_papers=tuple(str(p) for p in required),
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


def load_key(paths: Sequence[Path]) -> list[EvalCase]:
    """Read one evaluation key spread across several files.

    The hand-authored held-out cases and the generated ones live apart because
    they were authored differently, and a set is only as trustworthy as its
    authoring. Kept apart, `source_of` can split any result back into the two
    and show whether the generated half is systematically easier - which would
    mean its questions leak, not that the system improved.

    A question appearing in two files is refused rather than merged: scoring
    it twice would weight it double and silently distort every mean.
    """
    cases: list[EvalCase] = []
    seen: dict[str, Path] = {}
    for path in paths:
        for case in load_cases(path):
            first = seen.get(case.question)
            if first is not None:
                raise ValueError(
                    f"{path}: {case.question!r} is already in {first}; a case "
                    "scored twice is weighted twice"
                )
            seen[case.question] = path
            cases.append(case)
    return cases


def stratified_sample(
    cases: Sequence[EvalCase], *, size: int, seed: int = 0
) -> list[EvalCase]:
    """Draw evenly from each authoring batch, not from the top of the file.

    The batches are wildly different sizes - 36, 52 and 400 here - and score
    differently enough that a sample's composition decides its result. Taking
    the first N of a concatenated key would have drawn 88 hand-authored cases
    and 12 generated ones, and reported their average as though it described
    the corpus.

    Round-robin across batches, so a small batch is fully represented before a
    large one dominates, and seeded so the same sample can be scored twice.
    """
    if size < 1:
        raise ValueError("size must be at least 1")
    batches: dict[str, list[EvalCase]] = {}
    for case in cases:
        batches.setdefault(source_of(case), []).append(case)
    rng = random.Random(seed)
    for batch in batches.values():
        rng.shuffle(batch)

    drawn: list[EvalCase] = []
    position = 0
    while len(drawn) < size and any(
        position < len(batch) for batch in batches.values()
    ):
        for _source, batch in sorted(batches.items()):
            if position < len(batch):
                drawn.append(batch[position])
                if len(drawn) == size:
                    break
        position += 1
    return drawn


def source_of(case: EvalCase) -> str:
    """Which key a case came from, from its own note. 'unknown' if unlabelled."""
    for field in (case.note or "").split(";"):
        name, _, value = field.strip().partition("=")
        if name == "source":
            return value or "unknown"
    return "unknown"


def rank_of_first_hit(matches: Sequence[ChunkMatch], expected: Iterable[str]) -> int | None:
    """1-based rank of the first result from an expected paper, or None."""
    wanted = set(expected)
    for rank, match in enumerate(matches, start=1):
        if match.chunk.arxiv_id in wanted:
            return rank
    return None


def group_rank(matches: Sequence[ChunkMatch], group: str) -> int | None:
    """1-based rank at which one required part is first covered, or None.

    A group is a single paper id, or several joined by '|' when more than one
    held paper can supply that part of the answer.
    """
    return rank_of_first_hit(matches, group.split("|"))


@dataclass(frozen=True)
class CompoundScore:
    """How one retrieval mode covered the questions that need several papers.

    First-hit metrics cannot see the failure that matters here: ten excellent
    passages about PPO score perfectly on "how does GRPO differ from PPO" while
    answering half of it. Coverage@k is the mean fraction of a question's
    required parts found in the top k; complete@k is the fraction of questions
    with every part found.
    """

    mode: str
    cases: int
    coverage_at: dict[int, float]
    complete_at: dict[int, float]
    incomplete: tuple[str, ...]

    def render(self) -> str:
        columns = "  ".join(
            f"cov@{k} {self.coverage_at[k]:.2f}  all@{k} {self.complete_at[k]:.2f}"
            for k in sorted(self.coverage_at)
        )
        return f"  {self.mode:<16} {columns}"


def score_compound_mode(
    mode: str,
    cases: Sequence[EvalCase],
    retrieve: Callable[[str], Sequence[ChunkMatch]],
    *,
    cutoffs: Sequence[int] = (5, 10),
) -> CompoundScore:
    """Run every compound case through one retrieval mode and summarise it."""
    compound = [case for case in cases if case.is_compound]
    coverage = {k: 0.0 for k in cutoffs}
    complete = {k: 0 for k in cutoffs}
    incomplete: list[str] = []
    deepest = max(cutoffs, default=0)
    for case in compound:
        matches = retrieve(case.question)
        ranks = [group_rank(matches, group) for group in case.require_papers]
        for k in cutoffs:
            found = sum(1 for rank in ranks if rank is not None and rank <= k)
            coverage[k] += found / len(case.require_papers)
            if found == len(case.require_papers):
                complete[k] += 1
        if any(rank is None or rank > deepest for rank in ranks):
            incomplete.append(case.question)
    total = len(compound) or 1
    return CompoundScore(
        mode=mode,
        cases=len(compound),
        coverage_at={k: coverage[k] / total for k in cutoffs},
        complete_at={k: complete[k] / total for k in cutoffs},
        incomplete=tuple(incomplete),
    )


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

    Compound cases are excluded: any-of scoring would credit half an answer as
    a hit, and including them would also quietly move the numbers this table
    has carried across every measurement so far. They get their own scoring in
    score_compound_mode.
    """
    answerable = [case for case in cases if not case.expect_none and not case.is_compound]
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


def retrieval_modes(
    corpus: CorpusStore,
    embedder: Embedder,
    *,
    limit: int = 10,
    max_per_paper: int | None = None,
    reranker: object | None = None,
    complete: Callable[[str, str], str] | None = None,
    rewrites: int = 0,
    hyde: bool = False,
    decompose: bool = False,
) -> dict[str, Callable[[str], Sequence[ChunkMatch]]]:
    """Every retrieval mode under its reporting name, built once.

    Shared by first-hit and compound scoring, so both tables measure exactly
    the same retrieval rather than two drifting copies of it.

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
        "hybrid+compare": lambda q: hybrid_search(
            corpus, embedder, q, limit=limit, max_per_paper=max_per_paper,
            use_compare=True,
        ),
        # Both shape-gated legs together, which is what the orchestrator's
        # router runs. Each leg gates itself, so this should equal
        # hybrid+graph on single-subject questions and hybrid+compare on
        # comparisons - measured here rather than assumed.
        "hybrid+routed": lambda q: hybrid_search(
            corpus, embedder, q, limit=limit, max_per_paper=max_per_paper,
            use_graph=True, use_compare=True,
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
    return modes


def evaluate(
    corpus: CorpusStore,
    embedder: Embedder,
    cases: Sequence[EvalCase],
    **options: object,
) -> list[ModeScore]:
    """Score every retrieval mode on the same cases, so they are comparable."""
    modes = retrieval_modes(corpus, embedder, **options)  # type: ignore[arg-type]
    return [score_mode(name, cases, retrieve) for name, retrieve in modes.items()]


def evaluate_compound(
    corpus: CorpusStore,
    embedder: Embedder,
    cases: Sequence[EvalCase],
    **options: object,
) -> list[CompoundScore]:
    """Score every retrieval mode on the cases that need several papers."""
    modes = retrieval_modes(corpus, embedder, **options)  # type: ignore[arg-type]
    return [score_compound_mode(name, cases, retrieve) for name, retrieve in modes.items()]


def describe_case_mix(cases: Sequence[EvalCase]) -> str:
    """State what the answer key contains, split by what it is measuring.

    The two unanswerable kinds are reported apart because they are different
    problems: a coverage gap is visible to retrieval, and a covered-but-
    unanswered question is not. A single count hides which one a change moved.
    """
    coverage = sum(1 for case in cases if case.unanswerable_kind == NO_COVERAGE)
    unanswered = sum(1 for case in cases if case.unanswerable_kind == NO_ANSWER)
    compound = sum(1 for case in cases if case.is_compound)
    answerable = sum(1 for case in cases if not case.expect_none and not case.is_compound)
    described = (
        f"  {len(cases)} cases: {answerable} answerable, "
        f"{coverage} with no coverage, {unanswered} covered but unanswered"
    )
    if compound:
        described += f", {compound} compound (every listed paper required)"
    return described


def render_compound_report(scores: Sequence[CompoundScore]) -> str:
    """Lay out coverage of the compound cases, best complete@10 last."""
    deepest = max(scores[0].coverage_at) if scores else 10
    lines = [f"  {'mode':<16} coverage and completion of multi-paper questions"]
    ordered = sorted(scores, key=lambda s: (s.complete_at[deepest], s.coverage_at[deepest]))
    for score in ordered:
        lines.append(score.render())
    best = ordered[-1]
    lines.append(f"\n  best by all@{deepest}: {best.mode}  ({best.cases} cases)")
    if best.incomplete:
        lines.append(f"  {len(best.incomplete)} question(s) it did not fully cover:")
        lines.extend(f"    - {question}" for question in best.incomplete[:5])
    return "\n".join(lines)


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
