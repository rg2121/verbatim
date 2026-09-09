"""Per-case reciprocal ranks, so mode differences get an error bar."""
import json, random
from pathlib import Path
from verbatim import config
from verbatim.corpus import CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.evaluation import load_cases, rank_of_first_hit
from verbatim.retrieval import hybrid_search

MODES = {
    "hybrid":        {},
    "hybrid+graph":  {"use_graph": True},
    "hybrid+titles": {"use_titles": True},
    "hybrid+compare": {"use_graph": True, "use_compare": True},
}
cases = [c for c in load_cases(config.EVAL_PATH.parent / "heldout.jsonl") if not c.expect_none]
embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
rr = {m: [] for m in MODES}
with CorpusStore(config.STATE_PATH) as corpus:
    for case in cases:
        for mode, flags in MODES.items():
            matches = hybrid_search(corpus, embedder, case.question, limit=10, **flags)
            rank = rank_of_first_hit(matches, case.expect_papers)
            rr[mode].append(0.0 if rank is None else 1.0 / rank)

def ci(diffs, rounds=10000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]

print(f"{len(cases)} answerable held-out cases\n")
for mode in MODES:
    print(f"  {mode:<16} mrr {sum(rr[mode])/len(cases):.3f}")
print("\npaired differences (positive = first mode better), 95% bootstrap CI:")
for a, b in [("hybrid+graph", "hybrid"), ("hybrid+graph", "hybrid+titles"),
             ("hybrid+graph", "hybrid+compare"), ("hybrid", "hybrid+titles")]:
    diffs = [x - y for x, y in zip(rr[a], rr[b])]
    mean = sum(diffs) / len(diffs)
    lo, hi = ci(diffs)
    verdict = "distinguishable" if (lo > 0 or hi < 0) else "NOT distinguishable"
    print(f"  {a:<15} - {b:<15} {mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}")
Path("evals/results/rr.json").write_text(json.dumps(rr))
