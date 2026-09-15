"""Run the three arms over the frozen breadth questions, and keep every answer.

Written under evals/breadth/PROTOCOL.md. It refuses to run against a question
file that is not committed and unchanged, because the protocol's whole claim
is that the questions existed, exactly as they are, before any arm touched
them.

Each arm is the command as it ships: baseline is `ask`, ladder is
`ask --deep` with its tool-choosing rung live, team is `team` with its
defaults.

Arms run interleaved - all three on one question before the next - so a model
server that slows or rate-limits over an hour weighs on every arm alike,
rather than on whichever arm happened to run last.

Answers go to evals/results/, which is gitignored: they quote papers under
mixed licences, and a public repository is no place for them. What gets
committed later is scores and aggregates, never the text.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from verbatim import config
from verbatim.answers import OllamaToolCaller
from verbatim.answers import OllamaCompletion, answer_from_matches
from verbatim.corpus import CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.evaluation import load_cases
from verbatim.orchestrator import work_section
from verbatim.retrieval import hybrid_search
from verbatim.team import run_team

ARMS = ("baseline", "ladder", "team")
QUESTIONS = Path("evals/breadth/questions.jsonl")
OUT = Path("evals/results/breadth_answers.jsonl")


def committed_unchanged(path: Path) -> bool:
    """Whether git holds this file at HEAD with no edits since."""
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)], capture_output=True
    ).returncode == 0
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(path)], capture_output=True
    ).returncode == 0
    return tracked and clean


def calls_of(model) -> int:
    usage = getattr(model, "usage", None)
    return usage.calls if usage is not None else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--llm")
    args = parser.parse_args()

    if not committed_unchanged(args.questions):
        raise SystemExit(
            f"error: {args.questions} is not committed, or has changed since. "
            "The protocol requires the questions to be frozen before any arm runs."
        )
    cases = load_cases(args.questions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["arm"], row["question"]))
        if done:
            print(f"resuming: {len(done)} answer(s) already in {args.out}")

    model = args.llm or config.OLLAMA_MODEL
    embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
    completion = OllamaCompletion(model)
    caller = OllamaToolCaller(model)
    complete = lambda system, user: completion.complete_text(system, user)

    with CorpusStore(config.STATE_PATH) as corpus, args.out.open("a") as sink:
        for position, case in enumerate(cases, start=1):
            for arm in ARMS:
                if (arm, case.question) in done:
                    continue
                before = calls_of(completion) + calls_of(caller)
                try:
                    if arm == "baseline":
                        matches = hybrid_search(
                            corpus, embedder, case.question,
                            limit=config.ASK_PASSAGES,
                            max_per_paper=config.ASK_MAX_PER_PAPER, use_graph=True,
                        )
                        answer = answer_from_matches(corpus, completion, case.question, matches)
                    elif arm == "ladder":
                        section, ledger = work_section(
                            corpus, embedder, completion, case.question,
                            caller=caller, per_section=config.ASK_PASSAGES,
                            max_per_paper=config.ASK_MAX_PER_PAPER,
                        )
                        if ledger.outcome.startswith("failed"):
                            # The ladder contains a dead model as a declined
                            # section. Recording that as the ladder's answer
                            # would score an outage as a result.
                            raise RuntimeError(ledger.outcome)
                        answer = section.answer
                    else:
                        answer = run_team(
                            corpus, embedder, completion, complete, case.question,
                            max_tasks=3, needs=2, max_steps=12,
                        ).final
                except (RuntimeError, OSError) as error:
                    print(f"\nSTOPPED on question {position} ({arm}): {error}")
                    print("Everything answered so far is kept. Re-run the same command to resume.")
                    return 1
                sink.write(json.dumps({
                    "arm": arm,
                    "question": case.question,
                    "require_papers": list(case.require_papers),
                    "calls": calls_of(completion) + calls_of(caller) - before,
                    "answer": answer.to_dict(),
                }) + "\n")
                sink.flush()
            print(f"  {position}/{len(cases)} questions", end="\r", flush=True)

    print(f"\n{len(cases)} questions x {len(ARMS)} arms -> {args.out}")
    print("Next: uv run python evals/breadth_sheet.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
