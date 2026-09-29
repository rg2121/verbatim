"""The built-in and LangGraph supervisors over the same held-out questions.

They differ only in control flow, so they should be indistinguishable; this
measures it. Generation is not deterministic, so the distributions are
compared: acceptance, rungs climbed and cost.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from verbatim import config
from verbatim.answers import OllamaCompletion
from verbatim.corpus import CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.evaluation import load_key, stratified_sample
from verbatim.orchestrator import _supervisor

KEYS = [Path("evals/heldout.jsonl"), Path("evals/heldout-generated.jsonl")]
OUT = Path("evals/results/supervisors.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm")
    parser.add_argument(
        "--arms", default="builtin,langgraph",
        help="Two supervisors to compare. Pass 'builtin,builtin' for the "
             "control: the same implementation twice, which measures how much "
             "of any disagreement is generation non-determinism rather than "
             "the supervisor. Without it, a route that differs proves nothing",
    )
    args = parser.parse_args()
    arms = [a.strip() for a in args.arms.split(",")]
    if len(arms) != 2:
        parser.error("--arms takes exactly two, comma-separated")

    cases = [c for c in load_key([p for p in KEYS if p.exists()]) if not c.expect_none]
    cases = stratified_sample(cases, size=args.sample, seed=args.seed)
    embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
    completion = OllamaCompletion(args.llm or config.OLLAMA_MODEL)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    done = 0
    with CorpusStore(config.STATE_PATH) as corpus, OUT.open("w") as sink:
        for position, name in enumerate(arms):
            climb = _supervisor(name)
            label = f"{name}#{position + 1}"
            for case in cases:
                before = completion.usage.snapshot()
                try:
                    section, ledger = climb(
                        corpus, embedder, completion, case.question,
                        caller=None, per_section=config.ASK_PASSAGES,
                        max_per_paper=config.ASK_MAX_PER_PAPER,
                    )
                except (RuntimeError, OSError) as error:
                    print(f"\nSTOPPED: {error}")
                    break
                spent = completion.usage.since(before)
                row = {
                    "supervisor": label,
                    "question": case.question,
                    "outcome": ledger.outcome,
                    "shape": ledger.shape,
                    "rungs": len(ledger.attempts),
                    "route": [a.strategy for a in ledger.attempts],
                    "verified": len(section.answer.citations),
                    "cited_expected": bool(
                        {c.arxiv_id for c in section.answer.citations}
                        & set(case.expect_papers)
                    ),
                    "calls": spent.calls,
                }
                rows.append(row)
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                done += 1
                print(f"  {done}/{len(cases) * 2}", end="\r", flush=True)

    print(f"\n\n{len(cases)} held-out questions through each supervisor\n")
    labels = [f"{n}#{i + 1}" for i, n in enumerate(arms)]
    print(f"  {'':<22}" + "".join(f"{l:>12}" for l in labels))
    def column(fn):
        return [
            fn([r for r in rows if r["supervisor"] == label]) for label in labels
        ]
    measures = [
        ("answered with a citation", lambda rs: sum(1 for r in rs if r["verified"])),
        ("cited expected paper", lambda rs: sum(1 for r in rs if r["cited_expected"])),
        ("accepted on rung 1", lambda rs: sum(1 for r in rs if r["rungs"] == 1)),
        ("climbed every rung", lambda rs: sum(1 for r in rs if r["rungs"] >= 3)),
        ("model calls, total", lambda rs: sum(r["calls"] for r in rs)),
    ]
    for label, fn in measures:
        a, b = column(fn)
        print(f"  {label:<22}{a:>12}{b:>12}")

    # Non-determinism can still split a question, so count how many differ.
    by_question: dict[str, dict[str, list]] = collections.defaultdict(dict)
    for row in rows:
        by_question[row["question"]][row["supervisor"]] = row["route"]
    both = [q for q, v in by_question.items() if len(v) == 2]
    same = [q for q in both if by_question[q][labels[0]] == by_question[q][labels[1]]]
    print(f"\n  identical route on {len(same)} of {len(both)} question(s)")
    for q in both:
        if q not in same:
            print(f"    differs: {q[:60]}")
            print(f"      {labels[0]:<12}{by_question[q][labels[0]]}")
            print(f"      {labels[1]:<12}{by_question[q][labels[1]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
