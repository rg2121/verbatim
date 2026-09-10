"""Baseline retrieval against the role-based workflow, with and without the critic.

The orchestrator has been justified by argument and by one report on one
topic. This is the comparison that decides whether it is worth its cost: the
same frozen held-out questions put through three arms, scored on what the
answers turned out to be rather than on which paper retrieval named.

  baseline   one retrieval, one answer - what `ask` does
  workflow   the ladder: route by shape, verify, escalate to the next strategy
  +critic    the same, plus the critic, which escalates an uncovered subject

The arms share their first rung, so on any question the baseline answers well
the workflow does exactly the same work and costs the same. The difference can
only appear where the baseline fails, which is why the subset scored second -
questions the baseline could not get a verified citation for - matters more
than the overall row.
"""

from __future__ import annotations

import statistics
import sys

from verbatim import config
from verbatim.answer_eval import AnsweredCase, score_answers
from verbatim.answers import OllamaCompletion, answer_from_matches
from verbatim.corpus import CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.evaluation import load_cases
from verbatim.orchestrator import work_section
from verbatim.retrieval import hybrid_search

QUESTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def main() -> int:
    cases = [c for c in load_cases(config.EVAL_PATH.parent / "heldout.jsonl")
             if not c.expect_none][:QUESTIONS]
    embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
    completion = OllamaCompletion(config.OLLAMA_MODEL)
    complete_text = lambda system, user: completion.complete_text(system, user)
    results: dict[str, list] = {"baseline": [], "workflow": [], "+critic": []}
    costs: dict[str, list] = {"baseline": [], "workflow": [], "+critic": []}
    failed_by_baseline: set[str] = set()

    with CorpusStore(config.STATE_PATH) as corpus:
        for run in range(RUNS):
            for arm in ("baseline", "workflow", "+critic"):
                before = completion.usage.snapshot()
                answered = []
                for case in cases:
                    if arm == "baseline":
                        matches = hybrid_search(
                            corpus, embedder, case.question,
                            limit=config.ASK_PASSAGES,
                            max_per_paper=config.ASK_MAX_PER_PAPER, use_graph=True,
                        )
                        answer = answer_from_matches(
                            corpus, completion, case.question, matches
                        )
                        if not answer.citations:
                            failed_by_baseline.add(case.question)
                    else:
                        section, _ledger = work_section(
                            corpus, embedder, completion, case.question,
                            caller=None, per_section=config.ASK_PASSAGES,
                            max_per_paper=config.ASK_MAX_PER_PAPER,
                            critic=(arm == "+critic"),
                            complete_text=complete_text if arm == "+critic" else None,
                        )
                        answer = section.answer
                    answered.append(AnsweredCase(
                        question=case.question, answer=answer,
                        expected_papers=case.expect_papers,
                    ))
                results[arm].append(answered)
                costs[arm].append(completion.usage.since(before))
                print(f"  run {run + 1}/{RUNS} {arm} done", flush=True)

    print(f"\n{len(cases)} held-out questions, {RUNS} runs per arm\n")
    print(f"  {'':<26} {'baseline':>14} {'workflow':>14} {'+critic':>14}")
    rows = [
        ("verified citation", lambda s: s.with_verified_citation),
        ("every claim marked", lambda s: s.fully_cited),
        ("cited expected paper", lambda s: s.cited_expected_paper),
        ("citations rejected", lambda s: s.rejected_citations),
    ]
    for label, get in rows:
        cells = []
        for arm in ("baseline", "workflow", "+critic"):
            values = [get(score_answers(a)) for a in results[arm]]
            cells.append(f"{statistics.mean(values):5.1f} ({min(values)}-{max(values)})")
        print(f"  {label:<26} {cells[0]:>14} {cells[1]:>14} {cells[2]:>14}")
    cells = []
    for arm in ("baseline", "workflow", "+critic"):
        calls = statistics.mean(c.calls for c in costs[arm])
        tokens = statistics.mean(c.prompt_tokens + c.output_tokens for c in costs[arm])
        cells.append(f"{calls:.0f} calls {tokens / 1000:.0f}k")
    print(f"  {'cost per run':<26} {cells[0]:>14} {cells[1]:>14} {cells[2]:>14}")

    # The only place the arms can differ.
    print(f"\n  Of the {len(failed_by_baseline)} question(s) the baseline could not"
          f" get a verified citation for:")
    for arm in ("workflow", "+critic"):
        recovered = 0
        for answered in results[arm]:
            recovered += sum(
                1 for a in answered
                if a.question in failed_by_baseline and a.answer.citations
            )
        print(f"    {arm:<10} recovered {recovered / max(RUNS, 1):.1f} on average")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
