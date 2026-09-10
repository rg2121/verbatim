"""Per-case reciprocal ranks, so mode differences get an error bar.

Reads the whole held-out key - the hand-authored cases and the generated ones
together - and reports each source separately as well as combined. The split
is not decoration: the generated cases were written by a model from titles and
abstracts, and if that turns out to be a systematically easier way to author,
their scores are higher for that reason rather than because retrieval is
better. Comparing the halves is how that is caught.
"""
import json, random
from collections import defaultdict
from math import comb
from pathlib import Path

from verbatim import config
from verbatim.corpus import CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.evaluation import load_key, rank_of_first_hit, source_of
from verbatim.retrieval import hybrid_search

MODES = {
    "hybrid":        {},
    "hybrid+graph":  {"use_graph": True},
    "hybrid+titles": {"use_titles": True},
    "hybrid+compare": {"use_graph": True, "use_compare": True},
}
KEYS = [Path("evals/heldout.jsonl"), Path("evals/heldout-generated.jsonl")]

cases = [c for c in load_key([p for p in KEYS if p.exists()]) if not c.expect_none]
sources = [source_of(c) for c in cases]
embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
rr = {m: [] for m in MODES}
with CorpusStore(config.STATE_PATH) as corpus:
    for position, case in enumerate(cases, start=1):
        for mode, flags in MODES.items():
            matches = hybrid_search(corpus, embedder, case.question, limit=10, **flags)
            rank = rank_of_first_hit(matches, case.expect_papers)
            rr[mode].append(0.0 if rank is None else 1.0 / rank)
        if position % 50 == 0:
            print(f"  scored {position}/{len(cases)}", flush=True)


def ci(diffs, rounds=10000, seed=0):
    """Interval, and the share of resamples at or below zero.

    The share is reported because the interval alone put a verdict on the RNG
    seed once: with few distinct per-case differences the 2.5th percentile
    lands on zero for some seeds and just above it for others.
    """
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(rounds))
    below = sum(1 for m in means if m <= 0) / rounds
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)], below


def sign_test(diffs):
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    n = wins + losses
    if n == 0:
        return 0, 0, 1.0
    tail = sum(comb(n, k) for k in range(max(wins, losses), n + 1))
    return wins, losses, min(2 * tail / 2**n, 1.0)


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
for a, b in [("hybrid+graph", "hybrid"), ("hybrid+graph", "hybrid+titles"),
             ("hybrid+graph", "hybrid+compare"), ("hybrid", "hybrid+titles")]:
    diffs = [x - y for x, y in zip(rr[a], rr[b])]
    mean = sum(diffs) / len(diffs)
    lo, hi, below = ci(diffs)
    wins, losses, p = sign_test(diffs)
    verdict = "distinguishable" if (below < 0.025 and p < 0.05) else "NOT distinguishable"
    print(f"  {a:<15} - {b:<15} {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print(f"  {'':<33}differ on {wins + losses}: {wins} better, {losses} worse;"
          f" resamples<=0 {below:.3f}; sign p {p:.4f} -> {verdict}")

Path("evals/results").mkdir(parents=True, exist_ok=True)
Path("evals/results/rr.json").write_text(
    json.dumps({"modes": rr, "sources": sources,
                "questions": [c.question for c in cases]})
)
