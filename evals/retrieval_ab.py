"""Retrieval A/B: two named configurations, compared case by case on the held-out key.

    uv run python evals/retrieval_ab.py --list
    uv run python evals/retrieval_ab.py --arm hybrid+graph
    uv run python evals/retrieval_ab.py --compare hybrid+graph <another arm>
    uv run python evals/retrieval_ab.py --arm hybrid+graph --keys evals/retrieval.jsonl

An experiment's arms and its decision rule are added to ARMS together, and
committed, before either arm runs. Each arm's per-case ranks go to
evals/results/ab-<arm>.json, which git ignores, with the commit and corpus they
were measured on; ranks on another key, such as the development key, go to
ab-<arm>@<key>.json. --compare refuses arms measured on different cases or
corpora, and claims a difference only when the sign test and the bootstrap's
share at or below zero both support it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

from verbatim import config
from verbatim.corpus import CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.evaluation import (
    EvalCase,
    current_commit,
    load_key,
    rank_of_first_hit,
    source_of,
)
from verbatim.reranking import make_reranker
from verbatim.retrieval import hybrid_search, legs_disagree, vector_search
from significance import bootstrap_ci, paired_sign_test

KEYS = [Path("evals/heldout.jsonl"), Path("evals/heldout-generated.jsonl")]
RESULTS = Path("evals/results")
CUTOFFS = (1, 5, 10)

# Each arm is hybrid_search's options; a reranker is named and built once per run.
# The shipped default is hybrid+graph.
ARMS: dict[str, dict[str, Any]] = {
    "hybrid": {},
    "hybrid+graph": {"use_graph": True},
    # Experiment 1, ColBERT reranking. Rule fixed 2026-10-27, before either arm
    # ran: both rerank the top 50 fused passages with answerai-colbert-small-v1,
    # the gated arm only when the keyword and vector legs disagree on the top
    # paper. Each is compared with hybrid+graph on per-case reciprocal rank over
    # the 488 held-out cases. Two arms are tested, so each must clear p < 0.025
    # by the two-sided sign test, with under 2.5% of bootstrap resamples at or
    # below zero, to become the default; otherwise it stays behind
    # `--rerank colbert`. Either way the result and its added seconds go in
    # docs/findings.md.
    "hybrid+graph+colbert": {"use_graph": True, "reranker": "colbert"},
    "hybrid+graph+colbert-gated": {
        "use_graph": True, "reranker": "colbert", "rerank_when_legs_disagree": True,
    },
    # Exploratory, added 2026-10-28 after Experiment 1 ran, to ask whether the
    # gate or ColBERT did the work. No default rides on them.
    "hybrid+graph+cross": {"use_graph": True, "reranker": "cross"},
    "hybrid+graph+cross-gated": {
        "use_graph": True, "reranker": "cross", "rerank_when_legs_disagree": True,
    },
    # Experiment 2, the gated arm sparing definitions. Rule fixed 2026-10-28,
    # after the development key showed Experiment 1's arms burying definitional
    # questions. Experiment 1's saved ranks predict this arm exactly (held-out
    # mrr 0.939, 46-6; development 0.878, 1-1), so this run checks that the code
    # reproduces the prediction; it is not a fresh test. It replaces
    # hybrid+graph as the default only if, against it, it clears p < 0.025 by
    # the two-sided sign test with under 2.5% of bootstrap resamples at or below
    # zero on the 488 held-out cases, and is worse on no more development-key
    # cases (--keys evals/retrieval.jsonl) than it is better on.
    "hybrid+graph+colbert-gated-skipdefs": {
        "use_graph": True, "reranker": "colbert", "rerank_when_legs_disagree": True,
        "rerank_skips_definitions": True,
    },
}


def options_for(arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise SystemExit(f"error: unknown arm {arm!r}; known: {', '.join(sorted(ARMS))}")
    return ARMS[arm]


def rank_cases(
    corpus: CorpusStore, embedder, cases: list[EvalCase], options: dict[str, Any],
    *, limit: int = 10,
) -> list[dict]:
    """Where each case's expected paper ranks under one configuration, and how long it took."""
    if isinstance(options.get("reranker"), str):
        options = {**options, "reranker": make_reranker(options["reranker"])}
    rows = []
    for case in cases:
        started = time.perf_counter()
        matches = hybrid_search(corpus, embedder, case.question, limit=limit, **options)
        seconds = time.perf_counter() - started
        # Outside the timer: what the reranker's gate would see on this case.
        disagree = legs_disagree(
            corpus.search_chunks(case.question, limit=1, match="any"),
            vector_search(corpus, embedder, case.question, limit=1),
        )
        rows.append({
            "question": case.question,
            "source": source_of(case),
            "rank": rank_of_first_hit(matches, case.expect_papers),
            "seconds": round(seconds, 4),
            "legs_disagree": disagree,
        })
    return rows


def reciprocal(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def summarise(rows: list[dict]) -> dict[str, float]:
    n = len(rows) or 1
    summary: dict[str, float] = {"cases": len(rows)}
    for k in CUTOFFS:
        summary[f"r@{k}"] = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k) / n
    summary["mrr"] = sum(reciprocal(r["rank"]) for r in rows) / n
    summary["median_seconds"] = statistics.median(r["seconds"] for r in rows) if rows else 0.0
    return summary


def compare(before: list[dict], after: list[dict]) -> dict[str, Any]:
    """The second arm's per-case reciprocal rank minus the first's, tested."""
    first = {r["question"]: r for r in before}
    second = {r["question"]: r for r in after}
    if first.keys() != second.keys():
        raise SystemExit(
            f"error: the arms were run on different cases ({len(first)} and "
            f"{len(second)}); rerun one so both cover the same key"
        )
    diffs = [reciprocal(second[q]["rank"]) - reciprocal(first[q]["rank"]) for q in first]
    wins, losses, p = paired_sign_test(diffs)
    low, high, at_or_below = bootstrap_ci(diffs)
    return {
        "cases": len(diffs), "wins": wins, "losses": losses, "p": p,
        "mean_difference": sum(diffs) / len(diffs) if diffs else 0.0,
        "ci95": (low, high), "at_or_below_zero": at_or_below,
    }


def results_path(arm: str, keys: list[Path]) -> Path:
    """An arm's ranks on the held-out key, or on any other key, in a file of their own."""
    other = "" if keys == KEYS else "@" + "+".join(path.stem for path in keys)
    return RESULTS / f"ab-{arm}{other}.json"


def run_arm(arm: str, state: Path, keys: list[Path], limit: int) -> Path:
    options = options_for(arm)
    cases = [c for c in load_key(keys) if not c.expect_none and not c.is_compound]
    embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
    with CorpusStore(state) as corpus:
        passages = corpus.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        rows = rank_cases(corpus, embedder, cases, options, limit=limit)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = results_path(arm, keys)
    out.write_text(json.dumps({
        "arm": arm, "options": options, "commit": current_commit(),
        "state": str(state), "passages": passages,
        "keys": [str(k) for k in keys], "limit": limit, "rows": rows,
    }, indent=1))
    return out


def load(arm: str, keys: list[Path]) -> dict[str, Any]:
    path = results_path(arm, keys)
    if not path.exists():
        raise SystemExit(f"error: no results for {arm!r} on these keys; run --arm {arm} first")
    return json.loads(path.read_text())


def report(first: dict[str, Any], second: dict[str, Any]) -> str:
    if (first["state"], first["passages"]) != (second["state"], second["passages"]):
        raise SystemExit(
            "error: the arms were measured on different corpora "
            f"({first['state']}, {first['passages']} passages; "
            f"{second['state']}, {second['passages']} passages)"
        )
    lines = [f"  {'arm':<28} cases   r@1   r@5  r@10    mrr  median s"]
    for result in (first, second):
        s = summarise(result["rows"])
        lines.append(
            f"  {result['arm']:<28} {s['cases']:>5}  {s['r@1']:.2f}  {s['r@5']:.2f}"
            f"  {s['r@10']:.2f}  {s['mrr']:.3f}  {s['median_seconds']:.2f}"
        )
    sources = sorted({r["source"] for r in first["rows"]})
    if len(sources) > 1:
        lines.append("\n  mrr by batch:")
        for source in sources:
            pair = [summarise([r for r in result["rows"] if r["source"] == source])
                    for result in (first, second)]
            lines.append(f"    {source:<24} {pair[0]['mrr']:.3f} -> {pair[1]['mrr']:.3f}"
                         f"  ({pair[0]['cases']} cases)")
    # The gate's signal does not depend on the arm; read it from whichever recorded it.
    source = next((a for a in (first, second) if all("legs_disagree" in r for r in a["rows"])), None)
    flagged = source["rows"] if source else []
    if flagged:
        apart = [r for r in flagged if r["legs_disagree"]]
        together = [r for r in flagged if not r["legs_disagree"]]
        missed = lambda rows: sum(1 for r in rows if r["rank"] != 1)  # noqa: E731
        lines.append(
            f"\n  gate signal: the keyword and vector legs disagree on {len(apart)} of "
            f"{len(flagged)} cases; {source['arm']} misses rank 1 on {missed(apart)} "
            f"of those, and on {missed(together)} of the {len(together)} where they agree"
        )
    verdict = compare(first["rows"], second["rows"])
    low, high = verdict["ci95"]
    lines += [
        f"\n  {second['arm']} minus {first['arm']}:",
        f"    mrr {verdict['mean_difference']:+.3f}, better on {verdict['wins']}, "
        f"worse on {verdict['losses']}, sign test p = {verdict['p']:.4f}",
        f"    bootstrap 95% [{low:+.3f}, {high:+.3f}], "
        f"{verdict['at_or_below_zero']:.1%} of resamples at or below zero",
        f"\n  measured on {first['commit']} and {second['commit']}, "
        f"{first['state']} ({first['passages']} passages)",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--arm", help="Rank every held-out case under this arm")
    action.add_argument("--compare", nargs=2, metavar=("FIRST", "SECOND"))
    action.add_argument("--list", action="store_true", help="Name the arms and their options")
    parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    parser.add_argument("--keys", type=Path, nargs="+", default=KEYS)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if args.list:
        for name, options in ARMS.items():
            print(f"  {name:<28} {options}")
    elif args.arm:
        out = run_arm(args.arm, args.state, args.keys, args.limit)
        print(f"wrote {out}")
    else:
        print(report(load(args.compare[0], args.keys), load(args.compare[1], args.keys)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
