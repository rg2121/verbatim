"""Four ways to retrieve for a follow-up, scored on the paper the case names.

Every key in this project is single-turn, so the one thing a conversation adds
- that the second question does not carry its own subject - has never been
measured. `evals/conversational.jsonl` is two-turn: an opening question that
stands on its own, and a follow-up that does not, both answered by the same
paper. So retrieval for the follow-up is scored exactly as retrieval for the
opening, and the gap between them is what a conversation costs.

The arms all retrieve for the SECOND turn:

  as typed     the follow-up exactly as written, with no resolving at all
  rewritten    what ships: one model call turns it into a standalone question
  fused        both rankings merged by position, so a term the rewrite dropped
               is not lost
  scoped       the rewritten question searched within the papers the opening
               turn retrieved, falling back to the whole corpus if that finds
               nothing

The opening turn is answered for real before the rewrite, because that is what
the page sends: the rewrite sees the previous question and its answer.

Answers and per-case ranks go to evals/results/, which is gitignored. What
gets committed is the aggregate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from math import comb
from pathlib import Path

from verbatim import config
from verbatim.answers import OllamaCompletion, answer_from_matches
from verbatim.conversation import Turn, standalone_question
from verbatim.corpus import CorpusStore, PaperFilter
from verbatim.embedding import FastEmbedEmbedder
from verbatim.evaluation import rank_of_first_hit
from verbatim.expansion import fuse_rankings
from verbatim.retrieval import hybrid_search

ARMS = ("as typed", "rewritten", "fused", "scoped")
CASES = Path("evals/conversational.jsonl")
OUT = Path("evals/results/conversation_arms.jsonl")
CUTOFFS = (1, 5, 10)


def committed_unchanged(path: Path) -> bool:
    """Whether git holds this file at HEAD with no edits since."""
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)], capture_output=True
    ).returncode == 0
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(path)], capture_output=True
    ).returncode == 0
    return tracked and clean


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test. Ties carry no information and are dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(max(wins, losses), n + 1))
    return min(2 * tail / 2**n, 1.0)


def load_cases(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def reciprocal(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--llm")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore the checkpoint and retrieve everything again")
    args = parser.parse_args()

    if not committed_unchanged(args.cases):
        raise SystemExit(
            f"error: {args.cases} is not committed, or has changed since. The key "
            "is frozen before any arm runs against it."
        )
    cases = load_cases(args.cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if args.out.exists() and not args.fresh:
        for line in args.out.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["follow_up"]] = row
        if done:
            print(f"resuming: {len(done)} case(s) already in {args.out}")
    elif args.fresh and args.out.exists():
        args.out.unlink()

    embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
    completion = OllamaCompletion(args.llm or config.OLLAMA_MODEL)
    complete_text = lambda system, user: completion.complete_text(system, user)

    def retrieve(question: str, papers: PaperFilter | None = None):
        return hybrid_search(
            corpus, embedder, question,
            limit=config.ASK_PASSAGES, max_per_paper=config.ASK_MAX_PER_PAPER,
            use_graph=True, papers=papers,
        )

    stopped: str | None = None
    with CorpusStore(config.STATE_PATH) as corpus, args.out.open("a") as sink:
        for position, case in enumerate(cases, start=1):
            if case["follow_up"] in done:
                continue
            expected = case["expect_papers"]
            try:
                opening = retrieve(case["question"])
                # What the page sends as history is the question and its
                # answer, so the rewrite is measured against the same thing.
                answered = answer_from_matches(
                    corpus, completion, case["question"], opening
                )
                rewritten = standalone_question(
                    complete_text,
                    [Turn(case["question"], answered.text)],
                    case["follow_up"],
                )
                as_typed = retrieve(case["follow_up"])
                resolved = retrieve(rewritten)
                within = tuple(dict.fromkeys(m.chunk.arxiv_id for m in opening))
                scoped = retrieve(rewritten, PaperFilter(include_ids=within))
            except (RuntimeError, OSError) as error:
                stopped = f"{type(error).__name__}: {error}"
                break
            fell_back = not scoped
            if fell_back:
                # An empty scoped search is no answer at all; the honest
                # comparison is what the reader would have got instead.
                scoped = resolved
            ranks = {
                "as typed": rank_of_first_hit(as_typed, expected),
                "rewritten": rank_of_first_hit(resolved, expected),
                "fused": rank_of_first_hit(
                    fuse_rankings([resolved, as_typed])[: config.ASK_PASSAGES], expected
                ),
                "scoped": rank_of_first_hit(scoped, expected),
            }
            row = {
                "question": case["question"],
                "follow_up": case["follow_up"],
                "expected": expected,
                "opening_rank": rank_of_first_hit(opening, expected),
                "rewritten_as": rewritten,
                "rewrite_changed": rewritten.strip() != case["follow_up"].strip(),
                "scoped_fell_back": fell_back,
                "ranks": ranks,
            }
            done[case["follow_up"]] = row
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            print(f"  {position}/{len(cases)}", end="\r", flush=True)

    rows = list(done.values())
    if not rows:
        print("\nnothing scored")
        return 1

    print(f"\n\n{len(rows)} two-turn cases, retrieval for the follow-up\n")
    opening_mrr = sum(reciprocal(r["opening_rank"]) for r in rows) / len(rows)
    opening_first = sum(1 for r in rows if r["opening_rank"] == 1)
    print(f"  the opening question alone      mrr {opening_mrr:.3f}"
          f"   found first {opening_first}/{len(rows)}")
    print(f"  the rewrite changed the question on {sum(1 for r in rows if r['rewrite_changed'])}"
          f"/{len(rows)}; the scoped arm fell back on "
          f"{sum(1 for r in rows if r['scoped_fell_back'])}\n")

    print(f"  {'arm':<12}" + "".join(f"{'r@' + str(k):>8}" for k in CUTOFFS) + f"{'mrr':>9}")
    for arm in ARMS:
        ranks = [r["ranks"][arm] for r in rows]
        recalls = "".join(
            f"{sum(1 for rank in ranks if rank is not None and rank <= k) / len(rows):>8.2f}"
            for k in CUTOFFS
        )
        print(f"  {arm:<12}{recalls}{sum(map(reciprocal, ranks)) / len(rows):>9.3f}")

    print("\n  against what ships (rewritten), per case:")
    for arm in ARMS:
        if arm == "rewritten":
            continue
        wins = losses = 0
        for row in rows:
            mine, theirs = reciprocal(row["ranks"][arm]), reciprocal(row["ranks"]["rewritten"])
            wins += mine > theirs
            losses += mine < theirs
        p = sign_test(wins, losses)
        verdict = "distinguishable" if p < 0.05 else "NOT distinguishable"
        print(f"    {arm:<12} better on {wins:>2}, worse on {losses:>2}, "
              f"p = {p:.3f} -> {verdict}")

    misses = [r for r in rows if r["ranks"]["rewritten"] is None]
    if misses:
        print(f"\n  {len(misses)} follow-up(s) the shipped path found nothing for:")
        for row in misses[:5]:
            print(f"    {row['follow_up'][:70]}")
            print(f"      read as: {row['rewritten_as'][:70]}")
    if stopped is not None:
        print(f"\nSTOPPED after {len(rows)}: {stopped}\n  Re-run to resume.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
