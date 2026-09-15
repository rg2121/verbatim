"""Score the breadth evaluation under its pre-registered decision rule.

Mechanical measures (M1-M4) are computed from the answers at any time. The
human measures need the blind labels, and the script refuses to read the
blinding key until evals/breadth/labels.json is committed and unchanged: the
protocol commits the scores before anyone learns which arm wrote which answer,
and a check nobody enforces is a promise, not a protocol.

The aggregate result goes to evals/breadth/result.json - numbers only, never
the answer text, which quotes papers under mixed licences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from math import comb
from pathlib import Path

from verbatim.corpus import split_arxiv_version

ARMS = ("baseline", "ladder", "team")
CRITERIA = ("coverage", "balance", "limitations")
PRIMARY = ("coverage", "limitations", "both_papers")
ALPHA = 0.05


def question_id(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:10]


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test. Ties carry no information and are dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(max(wins, losses), n + 1))
    return min(2 * tail / 2**n, 1.0)


def committed_unchanged(path: Path) -> bool:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)], capture_output=True
    ).returncode == 0
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(path)], capture_output=True
    ).returncode == 0
    return tracked and clean


def mechanical(row: dict) -> dict:
    """M1-M4 for one answer. No model, no judgement."""
    answer = row["answer"]
    cited = {split_arxiv_version(c["arxiv_id"])[0] for c in answer["citations"]}
    required = {split_arxiv_version(p)[0] for p in row["require_papers"]}
    return {
        "both_papers": int(required <= cited),
        "distinct_papers": len(cited),
        "rejected": len(answer["rejected"]),
        "calls": row["calls"],
    }


def decide(per_question: dict[str, dict[str, dict]]) -> dict:
    """The protocol's decision rule, and every comparison behind it."""
    comparisons: dict[str, dict] = {}
    beats_both, worse_somewhere = [], []
    for measure in PRIMARY:
        comparisons[measure] = {}
        wins_over = []
        for other in ("baseline", "ladder"):
            wins = losses = 0
            for scores in per_question.values():
                team, rival = scores["team"][measure], scores[other][measure]
                wins += team > rival
                losses += team < rival
            p = sign_test(wins, losses)
            comparisons[measure][other] = {"wins": wins, "losses": losses, "p": p}
            wins_over.append(wins > losses and p < ALPHA)
            if losses > wins and p < ALPHA:
                worse_somewhere.append(f"{measure} vs {other}")
        if all(wins_over):
            beats_both.append(measure)
    earns = bool(beats_both) and not worse_somewhere
    return {
        "comparisons": comparisons,
        "beats_both_on": beats_both,
        "significantly_worse_on": worse_somewhere,
        "earns_its_cost": earns,
    }


def load_rows(answers: Path) -> dict[str, dict[str, dict]]:
    by_question: dict[str, dict[str, dict]] = {}
    for line in answers.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            by_question.setdefault(row["question"], {})[row["arm"]] = row
    return by_question


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=Path("evals/results/breadth_answers.jsonl"))
    parser.add_argument("--mapping", type=Path, default=Path("evals/results/breadth_mapping.json"))
    parser.add_argument("--labels", type=Path, default=Path("evals/breadth/labels.json"))
    parser.add_argument("--result", type=Path, default=Path("evals/breadth/result.json"))
    args = parser.parse_args()

    rows = load_rows(args.answers)
    complete = {q: arms for q, arms in rows.items() if set(arms) == set(ARMS)}
    if len(complete) < len(rows):
        print(f"note: {len(rows) - len(complete)} question(s) lack an answer from every arm and are left out")

    per_question: dict[str, dict[str, dict]] = {
        q: {arm: mechanical(arms[arm]) for arm in ARMS} for q, arms in complete.items()
    }
    n = len(per_question)
    print(f"\n{n} breadth questions\n")
    print(f"  {'':<34}" + "".join(f"{arm:>11}" for arm in ARMS))
    print(f"  {'M1 both required papers cited':<34}" + "".join(
        f"{sum(s[a]['both_papers'] for s in per_question.values()):>8}/{n:<2}" for a in ARMS))
    print(f"  {'M2 distinct papers cited, mean':<34}" + "".join(
        f"{statistics.mean(s[a]['distinct_papers'] for s in per_question.values()):>11.1f}" for a in ARMS))
    print(f"  {'M3 citations rejected, total':<34}" + "".join(
        f"{sum(s[a]['rejected'] for s in per_question.values()):>11}" for a in ARMS))
    print(f"  {'M4 model calls, mean':<34}" + "".join(
        f"{statistics.mean(s[a]['calls'] for s in per_question.values()):>11.1f}" for a in ARMS))

    if not args.labels.exists():
        print(f"\n  No human scores yet. Score blind in evals/results/breadth_sheet.html,"
              f"\n  save the download as {args.labels}, commit it, then run this again.")
        return 0
    if not committed_unchanged(args.labels):
        raise SystemExit(
            f"error: {args.labels} is not committed, or has changed since. Commit the "
            "scores before unblinding - that order is the point of the protocol."
        )

    labels = json.loads(args.labels.read_text())
    mapping = json.loads(args.mapping.read_text())
    missing = 0
    for question, scores in per_question.items():
        qid = question_id(question)
        letters = mapping.get(qid)
        if letters is None:
            # Answered, but not on the sheet that was scored: nothing to unblind.
            missing += len(ARMS) * len(CRITERIA)
            continue
        for letter, arm in letters.items():
            given = labels.get(qid, {}).get(letter, {})
            for criterion in CRITERIA:
                value = given.get(criterion)
                if value not in (0, 1, 2):
                    missing += 1
                    continue
                scores[arm][criterion] = value
    if missing:
        raise SystemExit(f"error: {missing} score(s) missing or outside 0-2. Finish the sheet first.")

    print()
    for criterion in CRITERIA:
        label = f"H {criterion}, mean of 0-2"
        print(f"  {label:<34}" + "".join(
            f"{statistics.mean(s[a][criterion] for s in per_question.values()):>11.2f}" for a in ARMS))

    verdict = decide(per_question)
    print("\n  team against each simpler arm, per question, two-sided sign test:")
    for measure, against in verdict["comparisons"].items():
        for other, c in against.items():
            print(f"    {measure:<12} vs {other:<9} better {c['wins']:>2}, worse {c['losses']:>2}, p = {c['p']:.4f}")
    if verdict["earns_its_cost"]:
        print(f"\n  VERDICT: the team earns its cost - it beats both simpler arms on "
              f"{', '.join(verdict['beats_both_on'])} and is not significantly worse on any primary measure.")
    else:
        why = (f"significantly worse on {', '.join(verdict['significantly_worse_on'])}"
               if verdict["significantly_worse_on"]
               else "it does not beat both simpler arms on any primary measure at p < 0.05")
        print(f"\n  VERDICT: the team does not earn its cost here - {why}. It stays opt-in.")

    means = {
        arm: {key: statistics.mean(s[arm][key] for s in per_question.values())
              for key in ("both_papers", "distinct_papers", "rejected", "calls", *CRITERIA)}
        for arm in ARMS
    }
    args.result.write_text(json.dumps({
        "questions": n,
        "protocol": "evals/breadth/PROTOCOL.md",
        "means": means,
        **verdict,
    }, indent=2) + "\n")
    print(f"\n  Written to {args.result} - commit it with the README update.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
