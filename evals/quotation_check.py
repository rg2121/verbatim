"""Quotation marks in an answer's own text: what the check does to saved answers.

    uv run python evals/quotation_check.py

The quote check proves the words on each citation line, and nothing checked the
words an answer puts in quotation marks in its text (src/verbatim/quotations.py).
This replays the new check over every answer the reflection experiments saved,
all three arms, written by gemma4:e4b-mlx without it. No model is called, so the
same answers are read with the check and without it.

Rule fixed 2026-11-05, before the check was run over these answers. It ships on
by default only if all three hold:

1. Nothing else changes. For every answer, the checked text is the saved text
   less the quotation marks of the phrases it unquotes, two marks each, and
   nothing more: the markers and every other character are untouched.
2. It acts only on quotations. Every phrase it unquotes or reports, all printed
   here and read by hand, is one the model put in quotation marks: none runs
   across two quotations or across a marker.
3. It finds what it is for. On the reflect arm of the fresh questions, each of
   the eleven phrases a first scan (2026-11-05) found in no cited passage is
   unquoted, and the one it forgives by rule, a capital first letter, is not.
   That scan found twelve and paired one pair of marks wrongly, which is what
   the second condition guards against.

Otherwise it ships off. Reported, deciding nothing: phrases checked, kept,
unquoted and reported as from another passage, per arm; answers changed; and
for each unquoted phrase how many of its words the closest passage sentence
shares, which separates a quote bent to fit a sentence from words no passage
holds.

Run once, 2026-11-05: 2,301 answers and 1,395 quoted phrases, 84 unquoted and
14 reported as from another passage. The first and third conditions held. All
98 phrases it acted on are quotations the model wrote, and none pairs the wrong
marks, but one answer, in two arms, quotes a paper's own reference, "A recent
work by [52] makes a broader... claim", which marker renumbering had already
rewritten as "[unverified]". Those quoted words run across a marker, so the
second condition fails by its letter, and the check ships off. Unquoting them was
right, since the text no longer said what the paper says (docs/findings.md).

Turned on the same day at Rishab's word, once renumbering was fixed to keep a
bracket inside a quotation a passage holds word for word: the failure was the
fault the check exposed, not one of its own. A rerun over these saved answers
still reports it, since they were written before the fix.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, closing
import json
from pathlib import Path
import re
import sqlite3
import sys

from verbatim.quotations import MIN_WORDS, check_quotations, quotation_spans
from verbatim.reflection import closest_words

SAVED = ("reflection-dev", "reflection-heldout", "reflection-fresh")
ARMS = ("plain", "resample", "reflect")
CORPORA = ("data/corpus.db", "data/evalcorpus.db")
MARKS = re.compile("[\"“”]")
MARKER = re.compile(r"\[[^\[\]]*\]")
CITATION_MARKER = re.compile(r"\[(?:\d+|unverified)\]")
# The first scan's phrases in no cited passage, by their opening words.
FIRST_SCAN = (
    "perform competitively on a range of state-of-the-art",
    "a single VGGish matches the best fusion",
    "lock the volatile weights into a safe generalization floor",
    "exhibit comparable redshift evolution to that defined",
    "to define natural-language preference axes",
    "to optimize across the multiple human-specified dimensions",
    "forc[es] a deformed minimum in the gs energy",
    "state-of-the-art controllability and temporal fidelity",
    "minimize compression entropy while maximizing relevance entropy",
    "human neural representations",
    "outperform larger models, including Qwen3.5-9B",
)
FORGIVEN = ("when monitoring is relatively cheap and punishment",)
# At or above this share of its words in one passage sentence, a phrase is a bent quote.
BENT = 0.75


def shared_words(phrase: str, passages: list[str]) -> float:
    """The most of the phrase's words any one cited passage sentence holds."""
    wanted = set(phrase.lower().split())
    best = 0.0
    for passage in passages:
        closest = closest_words(phrase, " ".join(passage.split())) or ""
        best = max(best, len(wanted & set(closest.lower().split())) / max(len(wanted), 1))
    return best


def replay(results: Path, stores: list[sqlite3.Connection]) -> dict:
    def passage(chunk_id: str) -> str | None:
        for store in stores:
            row = store.execute("SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
            if row:
                return row[0]
        return None

    tally: Counter = Counter()
    by_arm: dict[str, Counter] = {arm: Counter() for arm in ARMS}
    broken: list[str] = []
    acted: list[dict] = []
    fresh_reflect: list[str] = []
    for name in SAVED:
        with open(results / f"{name}.jsonl", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                for arm in ARMS:
                    answer = record["arms"].get(arm)
                    if not answer or answer["declined"] or not answer["citations"]:
                        continue
                    passages = [passage(citation["chunk_id"]) for citation in answer["citations"]]
                    if any(text is None for text in passages):
                        tally["answers skipped, a cited passage not found"] += 1
                        continue
                    text = answer["text"]
                    checked, problems = check_quotations(text, passages)
                    unquoted = [p for p in problems if p.found_in is None]
                    phrases = [
                        (opening, closing) for opening, closing in quotation_spans(text)
                        if len(text[opening + 1:closing].split()) >= MIN_WORDS
                    ]
                    count = by_arm[arm]
                    count["answers"] += 1
                    count["phrases checked"] += len(phrases)
                    count["unquoted"] += len(unquoted)
                    count["reported as from another passage"] += len(problems) - len(unquoted)
                    count["kept"] += len(phrases) - len(problems)
                    count["answers changed"] += bool(unquoted)
                    # Condition 1: only the unquoted phrases' marks went.
                    intact = (
                        len(checked) == len(text) - 2 * len(unquoted)
                        and MARKS.sub("", checked) == MARKS.sub("", text)
                        and MARKER.findall(checked) == MARKER.findall(text)
                    )
                    if not intact:
                        broken.append(f"{name} {arm}: {record['question'][:70]}")
                    for problem in problems:
                        at = text.find(problem.quote[:40])
                        acted.append({
                            "set": name, "arm": arm, "question": record["question"][:90],
                            "quote": problem.quote, "marker": problem.marker,
                            "found_in": problem.found_in,
                            "context": text[max(at - 60, 0):at + len(problem.quote) + 60] if at >= 0 else "",
                            "shared": None if problem.found_in else round(shared_words(problem.quote, passages), 2),
                        })
                        if name == "reflection-fresh" and arm == "reflect" and problem.found_in is None:
                            fresh_reflect.append(problem.quote)
    for count in by_arm.values():
        tally.update(count)
    return {"tally": tally, "by_arm": by_arm, "broken": broken, "acted": acted,
            "fresh_reflect": fresh_reflect}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("evals/results"))
    parser.add_argument("--out", type=Path, default=Path("evals/results/quotation_check.json"))
    args = parser.parse_args()

    with ExitStack() as stack:
        stores = [
            stack.enter_context(closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)))
            for path in CORPORA if Path(path).exists()
        ]
        found = replay(args.results, stores)

    tally, by_arm, acted = found["tally"], found["by_arm"], found["acted"]
    print("\nWhat the check does to the saved answers")
    for key in ("answers", "phrases checked", "kept", "unquoted",
                "reported as from another passage", "answers changed",
                "answers skipped, a cited passage not found"):
        per_arm = "  ".join(f"{arm} {by_arm[arm][key]}" for arm in ARMS) if key in by_arm["plain"] else ""
        print(f"  {tally[key]:>6}  {key:<44} {per_arm}")
    unquoted = [item for item in acted if item["found_in"] is None]
    bent = sum(item["shared"] >= BENT for item in unquoted)
    print(f"\n  of {len(unquoted)} unquoted: {bent} bent quotes (at least {BENT:.0%} of their words in "
          f"one passage sentence), {len(unquoted) - bent} further from any")

    print("\nEvery phrase it acted on, to read by hand (condition 2):")
    for item in acted:
        what = "unquoted" if item["found_in"] is None else f"from [{item['found_in']}], not [{item['marker']}]"
        shared = "" if item["shared"] is None else f", {item['shared']:.0%} shared"
        print(f"  [{what}{shared}] {item['set']}/{item['arm']}")
        print(f"      ...{' '.join(item['context'].split())}...")

    passed_one = not found["broken"]
    # The part of condition 2 code can check; whether each is a quotation is read by hand.
    across = [item for item in acted if CITATION_MARKER.search(item["quote"])]
    scan_found = [p for p in FIRST_SCAN if any(q.startswith(p) for q in found["fresh_reflect"])]
    forgiven_left = [p for p in FORGIVEN if any(q.startswith(p) for q in found["fresh_reflect"])]
    passed_three = len(scan_found) == len(FIRST_SCAN) and not forgiven_left
    print("\nRule")
    print(f"  1 nothing else changes:      {'pass' if passed_one else 'FAIL'}"
          + ("" if passed_one else f" ({len(found['broken'])} answers: {found['broken'][:3]})"))
    print("  2 acts only on quotations:   "
          + (f"FAIL ({len(across)} phrase(s) run across a marker: {across[0]['quote'][:60]!r})"
             if across else "read the list above for any phrase that is not a quotation"))
    print(f"  3 finds what it is for:      {'pass' if passed_three else 'FAIL'}"
          f" ({len(scan_found)} of {len(FIRST_SCAN)} first-scan phrases unquoted;"
          f" forgiven one {'left alone' if not forgiven_left else 'UNQUOTED'})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "tally": dict(tally), "by_arm": {arm: dict(count) for arm, count in by_arm.items()},
        "broken": found["broken"], "acted": acted, "first_scan_unquoted": scan_found,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0 if passed_one and passed_three and not across else 1


if __name__ == "__main__":
    sys.exit(main())
