"""Encoder A/B: the same passages embedded by three models, each searched as the shipped default searches.

    uv run verbatim embed --state data/evalcorpus.db --model BAAI/bge-base-en-v1.5
    (and the same for each arm)
    uv run python evals/encoder_ab.py --arm bge-small
    uv run python evals/encoder_ab.py --compare bge-small bge-base
    uv run python evals/encoder_ab.py --compare bge-small bge-base --keys evals/retrieval.jsonl

Re-embedding the full corpus takes half a day or more per model, so the arms run
on the eval corpus of the chunking experiment (data/evalcorpus.db, windows), whose
vectors are stored per model, side by side. Ranks go to
evals/results/encoder-<arm>.json; git ignores them.

Experiment 4, encoders. Rule fixed 2026-10-29, before any arm was scored. Two
arms are tested against bge-small-en-v1.5 (384 numbers, today's):
bge-base-en-v1.5 (768) and snowflake-arctic-embed-s (384), each given the query
and passage prefixes it was trained with. Each is compared with bge-small on
per-case reciprocal rank over the 488 held-out cases, searched with the shipped
default. With two arms, each must clear p < 0.025 by the two-sided sign test
with under 2.5% of bootstrap resamples at or below zero, and be worse on no
more development-key cases than it is better on. (First drafted with
arctic-embed-m and nomic-embed-text-v1.5 in place of arctic-embed-s; timed
before any embedding at 281 ms and 9.5 s a passage against bge-small's 47 -
12 hours and 16 days here - so they were dropped, the quantized nomic too at
about 30 times bge-small.)
Passing makes an arm a candidate, not the default: the default changes only if
the same comparison holds on the full corpus. Embedding time per passage and
median search time are reported for every arm. Also reported, deciding nothing
(fixed with the rule, before any run): the same comparisons searched with plain
hybrid, where windows missed 41 first places instead of 23, so an encoder's
effect is less hidden by the graph leg and the reranker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from verbatim.corpus import CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.evaluation import current_commit, load_key
import retrieval_ab

CORPUS = Path("data/evalcorpus.db")
MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "arctic-s": "snowflake/snowflake-arctic-embed-s",
}
SEARCH = "hybrid+graph+colbert-gated-skipdefs"
ARMS_TESTED = 2
ALPHA = 0.05 / ARMS_TESTED


def results_path(arm: str, keys: list[Path], search: str = SEARCH) -> Path:
    searched = "" if search == SEARCH else f".{search}"
    other = "" if keys == retrieval_ab.KEYS else "@" + "+".join(path.stem for path in keys)
    return retrieval_ab.RESULTS / f"encoder-{arm}{searched}{other}.json"


def unembedded(corpus: CorpusStore, model_id: str) -> int:
    """Passages this model has not embedded; scoring before `embed` finished would mix in nothing."""
    return int(corpus.connection.execute(
        """SELECT COUNT(*) FROM chunks AS c WHERE NOT EXISTS (
               SELECT 1 FROM embeddings AS e WHERE e.chunk_id = c.chunk_id AND e.model_id = ?)""",
        (model_id,),
    ).fetchone()[0])


def run_arm(arm: str, keys: list[Path], search: str = SEARCH) -> Path:
    cases = [c for c in load_key(keys) if not c.expect_none and not c.is_compound]
    # Never with context: the stored vectors were made bare.
    embedder = FastEmbedEmbedder(model_name=MODELS[arm], with_context=False)
    with CorpusStore(CORPUS) as corpus:
        missing = unembedded(corpus, embedder.model_id)
        if missing:
            raise SystemExit(
                f"error: {missing:,} passages have no {embedder.model_id} vector; "
                f"run verbatim embed --state {CORPUS} --model {embedder.model_id} first"
            )
        rows = retrieval_ab.rank_cases(corpus, embedder, cases, retrieval_ab.options_for(search))
    out = results_path(arm, keys, search)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "arm": arm, "model": MODELS[arm], "search": search, "commit": current_commit(),
        "keys": [str(k) for k in keys], "rows": rows,
    }, indent=1))
    return out


def load(arm: str, keys: list[Path], search: str = SEARCH) -> dict[str, Any]:
    path = results_path(arm, keys, search)
    if not path.exists():
        raise SystemExit(f"error: no results for {arm!r} on these keys; run --arm {arm} first")
    return json.loads(path.read_text())


def report(first: dict[str, Any], second: dict[str, Any]) -> str:
    lines = [f"  {'arm':<10} cases   r@1   r@5  r@10    mrr  median s"]
    for result in (first, second):
        s = retrieval_ab.summarise(result["rows"])
        lines.append(
            f"  {result['arm']:<10} {s['cases']:>5}  {s['r@1']:.2f}  {s['r@5']:.2f}  {s['r@10']:.2f}"
            f"  {s['mrr']:.3f}  {statistics.median(r['seconds'] for r in result['rows']):>8.2f}"
        )
    verdict = retrieval_ab.compare(first["rows"], second["rows"])
    low, high = verdict["ci95"]
    passes = verdict["p"] < ALPHA and verdict["at_or_below_zero"] < ALPHA and verdict["mean_difference"] > 0
    lines += [
        f"\n  {second['arm']} minus {first['arm']}, both searched with {first['search']}:",
        f"    mrr {verdict['mean_difference']:+.3f}, better on {verdict['wins']}, "
        f"worse on {verdict['losses']}, sign test p = {verdict['p']:.4f}",
        f"    bootstrap 95% [{low:+.3f}, {high:+.3f}], "
        f"{verdict['at_or_below_zero']:.1%} of resamples at or below zero",
        f"    {'clears' if passes else 'does not clear'} p < {ALPHA:.4f} on these keys",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--arm", choices=sorted(MODELS), help="Rank every case with this encoder")
    action.add_argument("--compare", nargs=2, metavar=("FIRST", "SECOND"), choices=sorted(MODELS))
    parser.add_argument("--keys", type=Path, nargs="+", default=retrieval_ab.KEYS)
    parser.add_argument(
        "--search", default=SEARCH, choices=sorted(retrieval_ab.ARMS),
        help="The retrieval arm to search with; any but the shipped default is exploratory",
    )
    args = parser.parse_args()
    if args.arm:
        print(f"wrote {run_arm(args.arm, args.keys, args.search)}")
    else:
        first, second = (load(arm, args.keys, args.search) for arm in args.compare)
        print(report(first, second))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
