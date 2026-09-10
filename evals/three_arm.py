"""Baseline retrieval against the role-based workflow, over the whole held-out set.

The orchestrator has been justified by argument and by one report on one
topic. This is the comparison that decides whether it is worth its cost: the
same frozen held-out questions put through each arm, scored on what the
answers turned out to be rather than on which paper retrieval named.

  baseline   one retrieval, one answer - what `ask` does
  workflow   the ladder: route by shape, verify, escalate to the next strategy
  +critic    the same, plus the critic, which escalates an uncovered subject

The arms share their first rung, so on any question the baseline answers well
the workflow does exactly the same work and costs the same. The difference can
only appear where the baseline fails.

The first version of this ran the first 20 questions in file order and reported
that the workflow recovered one of three baseline failures. Three is not a
sample, and the first twenty of a file authored in batches is not a draw from
the set. So: every answerable held-out question by default, a paired bootstrap
interval on the difference, and a checkpoint after every single answer, because
a three-hour run that dies at hour two must not cost three hours.

The critic is not in the default arms. It was measured worse at four times the
cost; re-running it here would treble the wall clock to re-decide a question
that is already decided. `--arms baseline,workflow,+critic` puts it back.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from verbatim import config
from verbatim.answer_eval import AnsweredCase, score_answers
from verbatim.answers import (
    Answer,
    OllamaCompletion,
    RejectedCitation,
    answer_from_matches,
)
from verbatim.citations import Citation
from verbatim.corpus import CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.evaluation import load_cases
from verbatim.orchestrator import work_section
from verbatim.retrieval import hybrid_search

ARMS = ("baseline", "workflow", "+critic")
DEFAULT_ARMS = "baseline,workflow"
RESULTS = Path(__file__).parent / "results" / "three_arm.jsonl"


@dataclass(frozen=True)
class Key:
    """One cell of the experiment: this question, this arm, this repeat."""

    run: int
    arm: str
    question: str


def encode(key: Key, case: AnsweredCase) -> str:
    payload = case.answer.to_dict()
    payload.pop("supporting_papers", None)  # derived, not a field
    return json.dumps({
        "run": key.run,
        "arm": key.arm,
        "question": key.question,
        "expected": list(case.expected_papers),
        "answer": payload,
    })


def decode(line: str) -> tuple[Key, AnsweredCase]:
    row = json.loads(line)
    payload = row["answer"]
    answer = Answer(
        question=payload["question"],
        text=payload["text"],
        citations=[Citation(**c) for c in payload["citations"]],
        rejected=[RejectedCitation(**r) for r in payload["rejected"]],
        declined=payload["declined"],
    )
    key = Key(run=row["run"], arm=row["arm"], question=row["question"])
    return key, AnsweredCase(
        question=row["question"],
        answer=answer,
        expected_papers=tuple(row["expected"]),
    )


def ci(values: list[float], rounds: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of a paired difference."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(rng.choice(values) for _ in range(n)) / n for _ in range(rounds)
    )
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def answer_one(
    arm: str, corpus: CorpusStore, embedder, completion, complete_text, question: str
) -> Answer:
    if arm == "baseline":
        matches = hybrid_search(
            corpus, embedder, question,
            limit=config.ASK_PASSAGES,
            max_per_paper=config.ASK_MAX_PER_PAPER,
            use_graph=True,
        )
        return answer_from_matches(corpus, completion, question, matches)
    section, _ledger = work_section(
        corpus, embedder, completion, question,
        caller=None,
        per_section=config.ASK_PASSAGES,
        max_per_paper=config.ASK_MAX_PER_PAPER,
        critic=(arm == "+critic"),
        complete_text=complete_text if arm == "+critic" else None,
    )
    return section.answer


def report(
    answered: dict[Key, AnsweredCase],
    arms: list[str],
    questions: list[str],
    runs: int,
    costs: dict[str, list],
) -> None:
    width = max(14, max(len(a) for a in arms) + 2)
    print(f"\n{len(questions)} held-out questions, {runs} run(s) per arm")
    print(f"  {'':<26}" + "".join(f"{arm:>{width}}" for arm in arms))
    rows = [
        ("verified citation", lambda s: s.with_verified_citation),
        ("every claim marked", lambda s: s.fully_cited),
        ("cited expected paper", lambda s: s.cited_expected_paper),
        ("declined outright", lambda s: s.declined),
        ("citations rejected", lambda s: s.rejected_citations),
    ]
    for label, get in rows:
        cells = []
        for arm in arms:
            per_run = [
                get(score_answers([c for k, c in answered.items()
                                   if k.arm == arm and k.run == r]))
                for r in range(runs)
            ]
            cells.append(f"{statistics.mean(per_run):5.1f} ({min(per_run)}-{max(per_run)})")
        print(f"  {label:<26}" + "".join(f"{c:>{width}}" for c in cells))
    cells = []
    for arm in arms:
        spent = costs.get(arm, [])
        if not spent:
            cells.append("not measured")
            continue
        calls = statistics.mean(c.calls for c in spent)
        tokens = statistics.mean(c.prompt_tokens + c.output_tokens for c in spent)
        cells.append(f"{calls:.0f} calls {tokens / 1000:.0f}k")
    print(f"  {'cost per run':<26}" + "".join(f"{c:>{width}}" for c in cells))

    # Per question, the share of runs that got a verified citation. Averaging
    # inside the question first keeps the pairing: the arms are compared on the
    # same questions, so the bootstrap resamples questions, not answers.
    rate: dict[str, dict[str, float]] = {arm: {} for arm in arms}
    for arm in arms:
        for question in questions:
            got = [c.answer.citations != []
                   for k, c in answered.items()
                   if k.arm == arm and k.question == question]
            if got:
                rate[arm][question] = sum(got) / len(got)
    if "baseline" not in arms:
        return
    shared = [q for q in questions
              if all(q in rate[arm] for arm in arms)]
    print(f"\n  paired on the {len(shared)} question(s) every arm answered,"
          " 95% bootstrap CI:")
    for arm in arms:
        if arm == "baseline":
            continue
        diffs = [rate[arm][q] - rate["baseline"][q] for q in shared]
        mean = sum(diffs) / max(len(diffs), 1)
        lo, hi = ci(diffs)
        verdict = "distinguishable" if (lo > 0 or hi < 0) else "NOT distinguishable"
        print(f"    {arm:<10} - baseline  {mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
              f"  {verdict}")

    hard = [q for q in shared if rate["baseline"][q] < 1.0]
    print(f"\n  On the {len(hard)} question(s) the baseline missed at least once"
          " (selected on the baseline's own failure, so its rate here is")
    print("  biased low and this subset cannot carry the headline):")
    for arm in arms:
        if not hard:
            break
        mean = sum(rate[arm][q] for q in hard) / len(hard)
        print(f"    {arm:<10} verified-citation rate {mean:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=0,
                        help="how many held-out questions; 0 means all")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--arms", default=DEFAULT_ARMS,
                        help=f"comma-separated, from {','.join(ARMS)}")
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any checkpoint and answer everything again")
    parser.add_argument("--report-only", action="store_true",
                        help="score the checkpoint without calling the model")
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        parser.error(f"unknown arm(s): {', '.join(unknown)}")

    cases = [c for c in load_cases(config.EVAL_PATH.parent / "heldout.jsonl")
             if not c.expect_none]
    if args.questions:
        cases = cases[:args.questions]
    questions = [c.question for c in cases]
    expected = {c.question: c.expect_papers for c in cases}

    answered: dict[Key, AnsweredCase] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.fresh:
        for line in args.out.read_text().splitlines():
            if line.strip():
                key, case = decode(line)
                answered[key] = case
        if answered:
            print(f"resuming: {len(answered)} answer(s) already in {args.out}")
    elif args.fresh and args.out.exists():
        args.out.unlink()

    costs: dict[str, list] = defaultdict(list)
    if not args.report_only:
        embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
        completion = OllamaCompletion(config.OLLAMA_MODEL)
        complete_text = lambda system, user: completion.complete_text(system, user)
        todo = [
            Key(run, arm, question)
            for run in range(args.runs)
            for arm in arms
            for question in questions
            if Key(run, arm, question) not in answered
        ]
        print(f"{len(todo)} answer(s) to generate"
              f" ({len(questions)} questions x {len(arms)} arms x {args.runs} runs)")
        # `todo` is ordered run, then arm, then question, so each arm's work is
        # one contiguous block and its cost is the usage across that block. A
        # single snapshot taken up front would charge every arm for all of them.
        stopped: str | None = None
        with CorpusStore(config.STATE_PATH) as corpus, args.out.open("a") as sink:
            block: tuple[int, str] | None = None
            started = completion.usage.snapshot()
            for done, key in enumerate(todo, start=1):
                if (key.run, key.arm) != block:
                    if block is not None:
                        costs[block[1]].append(completion.usage.since(started))
                    block = (key.run, key.arm)
                    started = completion.usage.snapshot()
                try:
                    answer = answer_one(
                        key.arm, corpus, embedder, completion, complete_text,
                        key.question,
                    )
                except (RuntimeError, OSError) as error:
                    # The model server went away. Everything answered so far is
                    # already flushed, so the run stops cleanly and the same
                    # command resumes it - rather than raising a traceback that
                    # says nothing about what survived.
                    stopped = f"{type(error).__name__}: {error}"
                    break
                case = AnsweredCase(
                    question=key.question,
                    answer=answer,
                    expected_papers=expected[key.question],
                )
                answered[key] = case
                sink.write(encode(key, case) + "\n")
                sink.flush()  # the checkpoint is worth nothing unbuffered
                if done % 10 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)}", flush=True)
            if block is not None:
                costs[block[1]].append(completion.usage.since(started))
        # A resumed run generates only part of an arm, so what it spent is not
        # that arm's cost. An arm's cost is reportable only when every one of
        # its answers was generated here rather than read from the checkpoint.
        generated = {key for key in todo if key in answered}
        for arm in arms:
            if any(Key(run, arm, question) not in generated
                   for run in range(args.runs) for question in questions):
                costs.pop(arm, None)
        if stopped is not None:
            print(f"\nSTOPPED after {len(answered)} answer(s): {stopped}")
            print(f"Re-run the same command to resume from {args.out}.")
            report(answered, arms, questions, args.runs, costs)
            return 1

    report(answered, arms, questions, args.runs, costs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
