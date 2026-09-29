"""Score the breadth evaluation under its pre-registered decision rule.

Mechanical measures (M1-M4) come from the answers alone. Human scores are
unblinded only when the blind manifest was committed first, the labels after
it, and the mapping and answers still match its hashes (see the amendment in
evals/breadth/PROTOCOL.md). Effect sizes and rater agreement are reported
beside the verdict but never change it. The result, numbers only, goes to
evals/breadth/result.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
from math import comb
from pathlib import Path

from verbatim.corpus import split_arxiv_version

ARMS = ("baseline", "ladder", "team")
CRITERIA = ("coverage", "balance", "limitations")
PRIMARY = ("coverage", "limitations", "both_papers")
ALPHA = 0.05
# Secondary, descriptive only: every primary measure plus balance (H2).
SECONDARY = ("coverage", "balance", "limitations", "both_papers")
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 0
MANIFEST_PATH = Path("evals/breadth/blind_manifest.json")
# Kept outside the repository until unblinding, then copied to
# REVEALED_MAPPING_PATH, where anyone can check it against the manifest.
MAPPING_PATH = Path.home() / ".verbatim-blind" / "breadth_mapping.json"
REVEALED_MAPPING_PATH = Path("evals/breadth/mapping.json")


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


def last_commit(path: Path) -> str | None:
    """The last commit that touched a file, or None if git holds none."""
    found = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return found or None


def committed_before(first: Path, then: Path) -> bool:
    """Whether `first` was last committed strictly before `then`.

    Committed together, or the manifest touched after the scores, is refused.
    """
    earlier, later = last_commit(first), last_commit(then)
    if not earlier or not later or earlier == later:
        return False
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", earlier, later], capture_output=True
    ).returncode == 0


def verify_blinding(manifest: dict, mapping_bytes: bytes, answers_bytes: bytes) -> dict:
    """The mapping, once it is proved to be the one fixed before scoring."""
    if hashlib.sha256(mapping_bytes).hexdigest() != manifest.get("mapping_sha256"):
        raise SystemExit(
            "error: this mapping is not the one whose hash was committed in the blind "
            "manifest before scoring. Unblinding with it would prove nothing."
        )
    if hashlib.sha256(answers_bytes).hexdigest() != manifest.get("answers_sha256"):
        raise SystemExit(
            "error: the answers have changed since the blind manifest was committed, "
            "so the scores may not be about these answers."
        )
    return json.loads(mapping_bytes)["mapping"]


def scores_for_this_blinding(labels: dict, manifest: dict) -> dict:
    """The scores, once shown to come from this blinding's sheet.

    Each blinding orders the answers afresh, so scores from an older sheet
    would unblind to the wrong arms. The sheet writes its key into the download.
    """
    scores = dict(labels)
    if scores.pop("blinding", None) != manifest.get("storage_key"):
        raise SystemExit(
            "error: these scores were not made on the sheet of the committed blind "
            "manifest, whose answers are in a different order. Score the current sheet."
        )
    return scores


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


def effect_sizes(
    per_question: dict[str, dict[str, dict]],
    *,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, dict]]:
    """The team's mean per-question difference from each simpler arm, with a
    seeded paired bootstrap 95% interval. Descriptive only; decide() is the rule.
    """
    rows = list(per_question.values())
    measures = [m for m in SECONDARY if rows and all(m in r["team"] for r in rows)]
    effects: dict[str, dict[str, dict]] = {}
    for measure in measures:
        effects[measure] = {}
        for other in ("baseline", "ladder"):
            diffs = [r["team"][measure] - r[other][measure] for r in rows]
            rng = random.Random(seed)
            means = sorted(
                statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(rounds)
            )
            low, high = means[int(rounds * 0.025)], means[int(rounds * 0.975) - 1]
            effects[measure][other] = {
                "mean_difference": round(float(statistics.mean(diffs)), 3),
                "ci95": [round(float(low), 3), round(float(high), 3)],
            }
    return effects


def weighted_kappa(first: list[int], second: list[int], categories=(0, 1, 2)) -> float | None:
    """Quadratic-weighted Cohen's kappa between two raters' ordinal scores.

    1 is perfect agreement, 0 is what chance would give, and a 0-versus-2
    disagreement costs four times a 0-versus-1. None when neither rater ever
    varied: agreement beyond chance is then undefined, not perfect.
    """
    n = len(first)
    if n == 0 or n != len(second):
        return None
    k = len(categories)
    index = {c: i for i, c in enumerate(categories)}
    observed = [[0.0] * k for _ in range(k)]
    for a, b in zip(first, second, strict=True):
        observed[index[a]][index[b]] += 1 / n
    rows = [sum(observed[i]) for i in range(k)]
    cols = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    weight = lambda i, j: ((i - j) / (k - 1)) ** 2
    disagreement = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k))
    expected = sum(weight(i, j) * rows[i] * cols[j] for i in range(k) for j in range(k))
    if expected == 0:
        return None
    return 1 - disagreement / expected


def agreement(first: dict, second: dict) -> dict[str, dict]:
    """How far two blind raters agree, per criterion, over answers both scored.

    Computed on the letters, never on the arms, so it needs no unblinding.
    """
    result: dict[str, dict] = {}
    for criterion in CRITERIA:
        a_scores, b_scores = [], []
        for qid, letters in first.items():
            for letter, given in letters.items():
                a = given.get(criterion)
                b = second.get(qid, {}).get(letter, {}).get(criterion)
                if a in (0, 1, 2) and b in (0, 1, 2):
                    a_scores.append(a)
                    b_scores.append(b)
        pairs = len(a_scores)
        kappa = weighted_kappa(a_scores, b_scores)
        result[criterion] = {
            "pairs": pairs,
            "exact": (sum(x == y for x, y in zip(a_scores, b_scores)) / pairs) if pairs else None,
            "kappa_w": None if kappa is None else round(kappa, 3),
        }
    return result


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
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--labels", type=Path, default=Path("evals/breadth/labels.json"))
    parser.add_argument("--result", type=Path, default=Path("evals/breadth/result.json"))
    parser.add_argument(
        "--second-labels", type=Path,
        help="A second blind rater's labels.json, for inter-rater agreement only; "
             "it must be committed before unblinding too, and never changes the verdict",
    )
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

    if args.second_labels is not None and not committed_unchanged(args.second_labels):
        raise SystemExit(
            f"error: {args.second_labels} is not committed, or has changed since. Both "
            "raters' scores are committed before unblinding."
        )

    if not args.manifest.exists() or not committed_unchanged(args.manifest):
        raise SystemExit(
            f"error: {args.manifest} is missing, not committed, or changed since. The "
            "blind manifest is committed before any score - that is the audit trail."
        )
    for scored in [args.labels, *([args.second_labels] if args.second_labels else [])]:
        if not committed_before(args.manifest, scored):
            raise SystemExit(
                f"error: {scored} was not committed after {args.manifest}. Scores "
                "committed with or before the manifest do not show the mapping was "
                "fixed first."
            )
    if not args.mapping.exists():
        raise SystemExit(f"error: no mapping at {args.mapping}; it is written by breadth_sheet.py.")
    manifest = json.loads(args.manifest.read_text())
    mapping_bytes = args.mapping.read_bytes()

    labels = scores_for_this_blinding(json.loads(args.labels.read_text()), manifest)
    mapping = verify_blinding(manifest, mapping_bytes, args.answers.read_bytes())
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

    # Secondary statistics: printed after the verdict and kept apart in the result.
    effects = effect_sizes(per_question)
    print("\n  secondary - team minus each simpler arm, mean per question, 95% bootstrap interval:")
    for measure, against in effects.items():
        for other, e in against.items():
            low, high = e["ci95"]
            print(f"    {measure:<12} vs {other:<9} {e['mean_difference']:+.2f}  [{low:+.2f}, {high:+.2f}]")
    agreed = None
    if args.second_labels is not None:
        second = scores_for_this_blinding(json.loads(args.second_labels.read_text()), manifest)
        agreed = agreement(labels, second)
        print("\n  secondary - agreement between the two blind raters:")
        for criterion, a in agreed.items():
            exact = "n/a" if a["exact"] is None else f"{a['exact']:.0%}"
            kappa = "undefined" if a["kappa_w"] is None else f"{a['kappa_w']:.2f}"
            print(f"    {criterion:<12} {a['pairs']:>3} answers, exact {exact}, weighted kappa {kappa}")

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
        "secondary": {
            "note": "descriptive only; the verdict is the preregistered sign-test rule",
            "effect_sizes": effects,
            "rater_agreement": agreed,
        },
        "audit": {
            "manifest_commit": last_commit(args.manifest),
            "labels_commit": last_commit(args.labels),
            "second_labels_commit": (
                last_commit(args.second_labels) if args.second_labels else None
            ),
            "mapping_sha256": manifest["mapping_sha256"],
            "answers_sha256": manifest["answers_sha256"],
        },
    }, indent=2) + "\n")
    # The mapping joins the repository only now; its hash matches the manifest's.
    REVEALED_MAPPING_PATH.write_bytes(mapping_bytes)
    print(f"\n  Written to {args.result}, and the mapping revealed at {REVEALED_MAPPING_PATH}.")
    print("  Commit both together: the mapping's hash matches the manifest committed at")
    print(f"  {last_commit(args.manifest)}, which is the audit trail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
