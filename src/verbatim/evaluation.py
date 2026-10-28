"""Retrieval evaluation against answer keys.

Relevance is recorded per paper, not per chunk: chunk ids change with the
chunker, while "this paper answers the question" stays true.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import random
import subprocess

from verbatim.corpus import ChunkMatch, CorpusStore
from verbatim.embedding import Embedder
from verbatim.research import gather_within_budget, plan_subquestions
from verbatim.retrieval import (
    default_search_options,
    expanded_search,
    hybrid_search,
    vector_search,
)

CUTOFFS = (1, 5, 10)
# Wide enough for the longest mode name, hybrid+graph+rerank.
MODE_WIDTH = 20

# Why a question has no answer here. Only a coverage gap is visible to
# retrieval, so the two kinds are scored apart.
NO_COVERAGE = "no_coverage"
NO_ANSWER = "no_answer"
UNANSWERABLE_KINDS = (NO_COVERAGE, NO_ANSWER)


@dataclass(frozen=True)
class EvalCase:
    """One question, and either the papers that answer it or nothing.

    An `expect_none` case says which kind it is: `no_coverage` (no paper is
    about the subject) or `no_answer` (covered, but no paper answers it).
    """

    question: str
    expect_papers: tuple[str, ...] = ()
    expect_none: bool = False
    unanswerable_kind: str = ""
    note: str = ""
    require_papers: tuple[str, ...] = ()

    @property
    def is_compound(self) -> bool:
        """Whether this question needs every listed paper, not any one of them.

        `require_papers` entries may be any-of groups written "idA|idB".
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
        return f"  {self.mode:<{MODE_WIDTH}} {recalls}  mrr {self.mrr:.3f}"


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
    """Read one key spread across several files, refusing a question listed twice."""
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
    """Draw round-robin from each authoring batch, seeded, so no batch dominates."""
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
    """Coverage of questions that need several papers.

    coverage@k is the mean share of required parts found in the top k;
    complete@k is the share of questions with every part found.
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
        return f"  {self.mode:<{MODE_WIDTH}} {columns}"


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
    """Run the answerable single-paper cases through one mode and summarise them.

    Recall@k means at least one top-k result came from an expected paper.
    Compound cases are scored by score_compound_mode.
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
    """Every retrieval mode under its reporting name, shared by both scorings.

    With a reranker, the shipped default is scored too: hybrid+graph+rerank,
    searched exactly as every front end searches.
    """
    def expanded(query: str) -> Sequence[ChunkMatch]:
        return expanded_search(
            corpus, embedder, query, complete=complete, rewrites=rewrites,
            hyde=hyde, limit=limit, max_per_paper=max_per_paper,
        )

    modes: dict[str, Callable[[str], Sequence[ChunkMatch]]] = {
        "keyword-all": lambda q: corpus.search_chunks(q, limit=limit, match="all"),
        "keyword-any": lambda q: corpus.search_chunks(q, limit=limit, match="any"),
        "vector": lambda q: vector_search(corpus, embedder, q, limit=limit),
        "hybrid": lambda q: hybrid_search(
            corpus, embedder, q, limit=limit, max_per_paper=max_per_paper,
        ),
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
        # Both shape-gated legs, as the orchestrator's router runs them.
        "hybrid+routed": lambda q: hybrid_search(
            corpus, embedder, q, limit=limit, max_per_paper=max_per_paper,
            use_graph=True, use_compare=True,
        ),
    }
    if reranker is not None:
        modes["hybrid+graph+rerank"] = lambda q: hybrid_search(
            corpus, embedder, q, limit=limit, max_per_paper=max_per_paper,
            **{**default_search_options(), "reranker": reranker},
        )
    # Offered only when a model is available to rewrite queries.
    if complete is not None and (rewrites or hyde):
        modes["hybrid+expansion"] = expanded
    if complete is not None and decompose:
        def decomposed(query: str) -> Sequence[ChunkMatch]:
            return gather_within_budget(
                corpus, embedder, plan_subquestions(complete, query),
                passages=limit, max_per_paper=max_per_paper,
            )

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
    """Summarise the key, counting the two unanswerable kinds apart."""
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
    lines = [f"  {'mode':<{MODE_WIDTH}} coverage and completion of multi-paper questions"]
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
    lines = [f"  {'mode':<{MODE_WIDTH}} recall@1        recall@5        recall@10   mrr"]
    for score in sorted(scores, key=lambda s: s.mrr):
        lines.append(score.render())
    best = max(scores, key=lambda s: s.mrr)
    lines.append(f"\n  best by mrr: {best.mode}  ({best.cases} cases)")
    if best.misses:
        lines.append(f"  {len(best.misses)} question(s) it found nothing for:")
        lines.extend(f"    - {question}" for question in best.misses[:5])
    return "\n".join(lines)


def current_commit() -> str | None:
    """The checked-out commit, with "+dirty" when tracked files have changed.

    None outside a repository, so a number is never credited to a commit it
    was not measured on.
    """
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], capture_output=True, text=True)

    try:
        head = git("rev-parse", "--short", "HEAD")
        changes = git("status", "--porcelain", "--untracked-files=no")
    except OSError:
        return None
    if head.returncode != 0:
        return None
    commit = head.stdout.strip()
    return f"{commit}+dirty" if changes.stdout.strip() else commit


def record_history(
    path: Path,
    scores: Sequence[ModeScore],
    *,
    keys: Sequence[str],
    state: str,
    passages: int,
    commit: str | None,
    when: str | None = None,
) -> int:
    """Append one line per mode to the eval history; returns the lines written.

    Each line names the commit and the corpus it was measured on, so a
    regression shows next to the change that caused it.
    """
    when = when or datetime.now(UTC).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as history:
        for score in scores:
            history.write(json.dumps({
                "when": when, "commit": commit, "keys": list(keys),
                "state": state, "passages": passages,
                "mode": score.mode, "cases": score.cases,
                **{f"r@{k}": round(score.recall_at[k], 4) for k in sorted(score.recall_at)},
                "mrr": round(score.mrr, 4),
            }) + "\n")
    return len(scores)
