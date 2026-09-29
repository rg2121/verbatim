"""Whether letting table passages into retrieval costs anything.

The same held-out key and default retrieval (hybrid+graph, top 10), run on the
code before and after the change and compared case by case.

DECISION RULE, written before either arm ran: tables ship in the default if
the fixed arm is not worse on held-out mrr by a two-sided sign test at
p < 0.05. Restoring what the ingest was built to do needs only to not hurt;
it does not need to win, and on a key written without tables in mind it is
not expected to.

    uv run python evals/tables_ab.py --arm before   # on the unfixed code
    uv run python evals/tables_ab.py --arm after    # on the fixed code
    uv run python evals/tables_ab.py --compare

Per-case ranks go to evals/results/, which git ignores.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

from verbatim import config
from verbatim.corpus import CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.evaluation import load_key, rank_of_first_hit, source_of
from verbatim.retrieval import hybrid_search

KEYS = [Path("evals/heldout.jsonl"), Path("evals/heldout-generated.jsonl")]
RESULTS = Path("evals/results")
CUTOFFS = (1, 5, 10)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test. Ties carry no information and are dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(max(wins, losses), n + 1))
    return min(2 * tail / 2**n, 1.0)


def reciprocal(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def run_arm(arm: str) -> None:
    cases = [
        case for case in load_key(KEYS)
        if not case.expect_none and not case.is_compound
    ]
    embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
    out = RESULTS / f"tables_{arm}.jsonl"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with CorpusStore(config.STATE_PATH) as corpus, out.open("w") as sink:
        for position, case in enumerate(cases, start=1):
            matches = hybrid_search(
                corpus, embedder, case.question, limit=10, use_graph=True
            )
            rank = rank_of_first_hit(matches, case.expect_papers)
            sink.write(json.dumps({
                "question": case.question,
                "source": source_of(case),
                "rank": rank,
                "first_hit_kind": (
                    matches[rank - 1].chunk.kind if rank is not None else None
                ),
                "tables_in_top10": sum(1 for m in matches if m.chunk.kind == "table"),
            }) + "\n")
            if position % 50 == 0:
                print(f"  {arm}: {position}/{len(cases)}", flush=True)
    print(f"wrote {out} ({len(cases)} cases)")


def load(arm: str) -> dict[str, dict]:
    path = RESULTS / f"tables_{arm}.jsonl"
    if not path.exists():
        raise SystemExit(f"error: {path} does not exist; run --arm {arm} first")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["question"]: row for row in rows}


def compare() -> int:
    before, after = load("before"), load("after")
    if before.keys() != after.keys():
        raise SystemExit("error: the two arms scored different cases")
    questions = sorted(before)
    print(f"{len(questions)} held-out cases, hybrid+graph, top 10\n")
    print(f"  {'arm':<8}" + "".join(f"{'r@' + str(k):>8}" for k in CUTOFFS) + f"{'mrr':>9}")
    for name, rows in (("before", before), ("after", after)):
        ranks = [rows[q]["rank"] for q in questions]
        recalls = "".join(
            f"{sum(1 for r in ranks if r is not None and r <= k) / len(ranks):>8.2f}"
            for k in CUTOFFS
        )
        mrr = sum(map(reciprocal, ranks)) / len(ranks)
        print(f"  {name:<8}{recalls}{mrr:>9.3f}")
    wins = sum(reciprocal(after[q]["rank"]) > reciprocal(before[q]["rank"]) for q in questions)
    losses = sum(reciprocal(after[q]["rank"]) < reciprocal(before[q]["rank"]) for q in questions)
    p = sign_test(wins, losses)
    shown = sum(1 for q in questions if after[q]["tables_in_top10"])
    by_table = sum(1 for q in questions if after[q]["first_hit_kind"] == "table")
    print(f"\n  after vs before, per case: better on {wins}, worse on {losses}, p = {p:.3f}")
    print(f"  a table reached the top 10 on {shown}/{len(questions)} cases;"
          f" it was the first hit from the right paper on {by_table}")
    worse = losses > wins and p < 0.05
    print("\n  DECISION: " + (
        "tables stay out - the fixed arm is measurably worse" if worse
        else "tables ship - the fixed arm is not measurably worse"
    ))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--arm", choices=("before", "after"))
    group.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        return compare()
    run_arm(args.arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
