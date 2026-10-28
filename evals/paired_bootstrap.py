"""Per-case reciprocal ranks for each retrieval mode, with paired bootstrap intervals.

Hand-written and generated held-out cases are also reported apart, so a
generated half that is easier to retrieve for would show.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from verbatim import config
from verbatim.corpus import CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.evaluation import load_key, rank_of_first_hit, source_of
from verbatim.retrieval import hybrid_search
from significance import bootstrap_ci as ci, paired_sign_test as sign_test

MODES = {
    "hybrid":        {},
    "hybrid+graph":  {"use_graph": True},
    "hybrid+titles": {"use_titles": True},
    "hybrid+compare": {"use_graph": True, "use_compare": True},
}
KEYS = [Path("evals/heldout.jsonl"), Path("evals/heldout-generated.jsonl")]
COMPARISONS = [
    ("hybrid+graph", "hybrid"),
    ("hybrid+graph", "hybrid+titles"),
    ("hybrid+graph", "hybrid+compare"),
    ("hybrid", "hybrid+titles"),
]
OUT = Path("evals/results/rr.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    args = parser.parse_args()

    cases = [c for c in load_key([p for p in KEYS if p.exists()]) if not c.expect_none]
    sources = [source_of(c) for c in cases]
    embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
    rr = {m: [] for m in MODES}
    with CorpusStore(args.state) as corpus:
        for position, case in enumerate(cases, start=1):
            for mode, flags in MODES.items():
                matches = hybrid_search(corpus, embedder, case.question, limit=10, **flags)
                rank = rank_of_first_hit(matches, case.expect_papers)
                rr[mode].append(0.0 if rank is None else 1.0 / rank)
            if position % 50 == 0:
                print(f"  scored {position}/{len(cases)}", flush=True)

    by_source = defaultdict(list)
    for index, source in enumerate(sources):
        by_source[source].append(index)

    print(f"\n{len(cases)} answerable held-out cases")
    for source, indices in sorted(by_source.items()):
        print(f"  {source}: {len(indices)}")

    print(f"\n{'mode':<17}{'mrr (all)':>11}" + "".join(f"{s[:18]:>20}" for s in sorted(by_source)))
    for mode in MODES:
        overall = sum(rr[mode]) / len(cases)
        cells = [
            sum(rr[mode][i] for i in indices) / len(indices)
            for _source, indices in sorted(by_source.items())
        ]
        print(f"  {mode:<15}{overall:>11.3f}" + "".join(f"{c:>20.3f}" for c in cells))

    print("\npaired differences (positive = first mode better):")
    for a, b in COMPARISONS:
        diffs = [x - y for x, y in zip(rr[a], rr[b], strict=True)]
        mean = sum(diffs) / len(diffs)
        lo, hi, below = ci(diffs)
        wins, losses, p = sign_test(diffs)
        verdict = "distinguishable" if (below < 0.025 and p < 0.05) else "NOT distinguishable"
        print(f"  {a:<15} - {b:<15} {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
        print(f"  {'':<33}differ on {wins + losses}: {wins} better, {losses} worse;"
              f" resamples<=0 {below:.3f}; sign p {p:.4f} -> {verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"modes": rr, "sources": sources,
                    "questions": [c.question for c in cases]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
