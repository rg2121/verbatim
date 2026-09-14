"""Three ways to answer the same held-out questions, and what each costs.

The team is tested but its quality gain is not measured, which is the one
criticism this project cannot wave away. So:

  baseline   one retrieval, one answer - what `ask` does
  ladder     route by shape, verify, escalate - what `ask --deep` does
  team       tasks on a shared board, an adversarial worker dispatched by what
             the board shows missing, one answer from approved evidence

Scored on the answer, not on which paper retrieval named. The team's *final*
answer is the one compared, because that is the thing a reader gets; its
per-task answers are working notes.

The team costs more by construction - it plans, retrieves per task, searches
adversarially and synthesises. The question is not whether it costs more but
whether anything is bought, so cost is reported beside quality rather than
underneath it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from verbatim import config
from verbatim.answer_eval import AnsweredCase, score_answers
from verbatim.answers import OllamaCompletion, answer_from_matches
from verbatim.corpus import CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.evaluation import load_key, stratified_sample
from verbatim.orchestrator import work_section
from verbatim.retrieval import hybrid_search
from verbatim.team import run_team

KEYS = [Path("evals/heldout.jsonl"), Path("evals/heldout-generated.jsonl")]
OUT = Path("evals/results/team_arms.jsonl")
ARMS = ("baseline", "ladder", "team")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm")
    args = parser.parse_args()

    cases = [c for c in load_key([p for p in KEYS if p.exists()]) if not c.expect_none]
    cases = stratified_sample(cases, size=args.sample, seed=args.seed)
    embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
    completion = OllamaCompletion(args.llm or config.OLLAMA_MODEL)
    complete = lambda system, user: completion.complete_text(system, user)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    answered: dict[str, list[AnsweredCase]] = {a: [] for a in ARMS}
    costs: dict[str, list] = {a: [] for a in ARMS}
    adversarial_used: list[tuple[int, int]] = []
    done = 0
    with CorpusStore(config.STATE_PATH) as corpus, OUT.open("w") as sink:
        for arm in ARMS:
            for case in cases:
                before = completion.usage.snapshot()
                try:
                    if arm == "baseline":
                        matches = hybrid_search(
                            corpus, embedder, case.question,
                            limit=config.ASK_PASSAGES,
                            max_per_paper=config.ASK_MAX_PER_PAPER, use_graph=True,
                        )
                        answer = answer_from_matches(
                            corpus, completion, case.question, matches
                        )
                    elif arm == "ladder":
                        section, _ledger = work_section(
                            corpus, embedder, completion, case.question,
                            caller=None, per_section=config.ASK_PASSAGES,
                            max_per_paper=config.ASK_MAX_PER_PAPER,
                        )
                        answer = section.answer
                    else:
                        report = run_team(
                            corpus, embedder, completion, complete, case.question,
                            max_tasks=2, needs=2,
                        )
                        answer = report.final
                        adversarial_used.append(report.citations_from_adversarial)
                except (RuntimeError, OSError) as error:
                    print(f"\nSTOPPED in {arm}: {error}")
                    break
                spent = completion.usage.since(before)
                costs[arm].append(spent)
                answered[arm].append(AnsweredCase(
                    question=case.question, answer=answer,
                    expected_papers=case.expect_papers,
                ))
                sink.write(json.dumps({
                    "arm": arm, "question": case.question,
                    "declined": answer.declined,
                    "verified": len(answer.citations),
                    "rejected": len(answer.rejected),
                    "cited_expected": bool(
                        {c.arxiv_id for c in answer.citations}
                        & set(case.expect_papers)
                    ),
                    "calls": spent.calls,
                }) + "\n")
                sink.flush()
                done += 1
                print(f"  {done}/{len(cases) * len(ARMS)}", end="\r", flush=True)

    print(f"\n\n{len(cases)} held-out questions through each arm\n")
    print(f"  {'':<26}" + "".join(f"{a:>12}" for a in ARMS))
    rows = [
        ("verified citation", lambda s: s.with_verified_citation),
        ("every claim marked", lambda s: s.fully_cited),
        ("cited expected paper", lambda s: s.cited_expected_paper),
        ("declined", lambda s: s.declined),
        ("citations rejected", lambda s: s.rejected_citations),
    ]
    for label, get in rows:
        cells = [str(get(score_answers(answered[a]))) if answered[a] else "-"
                 for a in ARMS]
        print(f"  {label:<26}" + "".join(f"{c:>12}" for c in cells))
    cells = [
        f"{statistics.mean(c.calls for c in costs[a]):.1f}" if costs[a] else "-"
        for a in ARMS
    ]
    print(f"  {'model calls per question':<26}" + "".join(f"{c:>12}" for c in cells))
    print(f"  {'answered':<26}" + "".join(f"{len(answered[a]):>12}" for a in ARMS))

    if adversarial_used:
        used = sum(u for u, _ in adversarial_used)
        total = sum(t for _, t in adversarial_used)
        print(f"\n  team: {used} of {total} citation(s) rested on a passage only"
              " the adversarial search found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
