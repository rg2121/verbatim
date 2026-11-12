"""Build the v2 evaluation's frozen sets and their manifest (plan revision 6, M0 step 2).

The questions are written by `verbatim author-cases`, as every earlier key was.
This script does the rest, and every draw uses a seed made from a fixed label,
so none was picked by looking at a result:

  split       the 60 dev questions drawn from heldout-generated.jsonl; test-a.jsonl
              starts with the other 340 and all of heldout-generated-2.jsonl, and
              author-cases then appends 460 new ones
  claims      planted-gap claims written once, with planted_gaps.py's prompt and
              leak filter, so both arms read the same claims
  abstention  400 questions from each test set, each hiding the paper that answers
              it; set A also gets heldout.jsonl's hand-written unanswerable cases
  topics      your 20 report topics, split 10/10 by the seed
  manifest    every set's SHA-256 and kind, the seeds, and the authoring model

Every file is written once; redoing one takes a deliberate deletion.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

from verbatim import config
from verbatim.authoring import MAX_SHARED_RUN, candidate_papers, excluded_papers, longest_shared_run
from verbatim.corpus import CorpusStore, unversioned
from verbatim.gaps import checkable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from planted_gaps import CLAIM_LINE, CLAIM_SYSTEM  # noqa: E402

EVALS = Path(__file__).resolve().parent
V2 = EVALS / "v2"
MODEL = "gemma4:e4b-mlx"

DEV_SIZE = 60
TEST_SIZE = 1_040
ABSTENTION_SIZE = 400
TOPICS = 20
CLAIMS = (("gaps-a", 50), ("gaps-b", 50), ("dev-claims", 10))

SETS: dict[str, tuple[Path, str]] = {
    "dev": (V2 / "dev.jsonl", "questions"),
    "dev-pairs": (V2 / "dev-pairs.jsonl", "breadth"),
    "dev-claims": (V2 / "dev-claims.jsonl", "claims"),
    "test-a": (V2 / "test-a.jsonl", "questions"),
    "test-b": (V2 / "test-b.jsonl", "questions"),
    "abstention-a": (V2 / "abstention-a.jsonl", "abstention"),
    "abstention-b": (V2 / "abstention-b.jsonl", "abstention"),
    "team-a": (EVALS / "breadth" / "questions.jsonl", "breadth"),
    "team-b": (V2 / "team-b.jsonl", "breadth"),
    "reports-a": (V2 / "reports-a.jsonl", "topics"),
    "reports-b": (V2 / "reports-b.jsonl", "topics"),
    "gaps-a": (V2 / "gaps-a.jsonl", "claims"),
    "gaps-b": (V2 / "gaps-b.jsonl", "claims"),
}
SEEDS = ("dev", "abstention-a", "abstention-b", "gaps-a", "gaps-b", "dev-claims", "reports", "human")

# The keys author-cases already excludes, and the v2 keys written before the claims.
QUESTION_KEYS = (
    EVALS / "retrieval.jsonl", EVALS / "heldout.jsonl", EVALS / "heldout-generated.jsonl",
    EVALS / "heldout-generated-2.jsonl", EVALS / "gate.jsonl",
    EVALS / "breadth" / "questions.jsonl", EVALS / "conversational.jsonl",
    V2 / "test-a.jsonl", V2 / "test-b.jsonl", V2 / "team-b.jsonl", V2 / "dev-pairs.jsonl",
)

DEV_HEADER = """\
# v2 evaluation, dev set (plan revision 6): the only questions used for
# iterating. Drawn from heldout-generated.jsonl with the frozen seed "dev" (see
# manifest.json); that file's other rows open test-a.jsonl.
"""

TEST_A_HEADER = """\
# v2 evaluation, test set A (plan revision 6), frozen before any v2 code ran:
# the rows of heldout-generated.jsonl not drawn for dev, all of
# heldout-generated-2.jsonl, then 460 more written by `verbatim author-cases`
# (model in manifest.json) from titles and abstracts only, under the rules in
# those files' headers. No retrieval ran against any question before this file
# was committed.
"""

ABSTENTION_HEADER = """\
# v2 evaluation, abstention set (plan revision 6): questions drawn from {source}
# with the frozen seed "{label}". Each row's `hide` names the paper that answers
# it, so for that run the corpus does not.{extra}
"""

CLAIMS_HEADER = """\
# v2 evaluation, planted gaps (plan revision 6): each claim was written once
# (model in manifest.json) with evals/planted_gaps.py's prompt, from one paper's
# title and abstract, as a claim that paper disproves. Papers were taken in the
# order the frozen seed "{label}" shuffles them; claims that cannot be disproved
# or that borrow a {run}-word run from their abstract were rejected. `topic` is
# the paper's title, as planted_gaps.py gives it to the Falsifier.
"""

TOPICS_HEADER = """\
# v2 evaluation, report topics (plan revision 6): written on 2026-11-12, before any
# v2 code existed, from web searches of current research in cs, math and physics;
# the corpus was not searched. Listed alphabetically, then split 10/10 between
# reports-a and reports-b by the frozen seed "reports".
"""


def seed(label: str) -> int:
    """A seed from a fixed label, so no draw was chosen by its outcome."""
    return int(hashlib.sha256(f"verbatim-v2:{label}".encode()).hexdigest()[:8], 16)


def rows_of(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def write_once(path: Path, header: str, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists() and path.read_text(encoding="utf-8").strip():
        raise SystemExit(f"error: {path} exists. A frozen set is written once; "
                         "delete it deliberately to redo it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def drawn(count: int, size: int, label: str) -> list[int]:
    """Which positions a seeded draw takes, kept in their original order."""
    if size > count:
        raise SystemExit(f"error: cannot draw {size} from {count}")
    return sorted(random.Random(seed(label)).sample(range(count), size))


def split(first: Path, second: Path, dev_out: Path, test_out: Path,
          *, size: int = DEV_SIZE) -> tuple[list[dict], list[dict]]:
    for path in (dev_out, test_out):
        if path.exists():
            raise SystemExit(f"error: {path} exists. A frozen set is written once; "
                             "delete it deliberately to redo it.")
    generated = rows_of(first)
    chosen = set(drawn(len(generated), size, "dev"))
    dev = [row for i, row in enumerate(generated) if i in chosen]
    test = [row for i, row in enumerate(generated) if i not in chosen] + rows_of(second)
    write_once(dev_out, DEV_HEADER, dev)
    write_once(test_out, TEST_A_HEADER, test)
    return dev, test


def claimed(paths: Iterable[Path]) -> set[str]:
    """Every paper a claims file planted, with and without its version."""
    seen: set[str] = set()
    for path in paths:
        if path.exists():
            for row in rows_of(path):
                seen |= {row["arxiv_id"], unversioned(row["arxiv_id"])}
    return seen


def write_claims(corpus: CorpusStore, complete: Callable[[str, str], str], *, count: int,
                 label: str, exclude: set[str],
                 rejected: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Claims each disproved by one paper, written blind from its title and abstract."""
    counts = rejected if rejected is not None else {}
    papers = candidate_papers(corpus, exclude=exclude, limit=10**9, stride=1)
    rows: list[dict[str, Any]] = []
    for arxiv_id, title, abstract in random.Random(seed(label)).sample(papers, len(papers)):
        found = CLAIM_LINE.search(complete(CLAIM_SYSTEM, f"TITLE: {title}\n\nABSTRACT: {abstract}") or "")
        if found is None:
            reason = "no claim written"
        elif not checkable(found.group(1)):
            reason = "cannot be disproved"
        elif (run := longest_shared_run(found.group(1), abstract)) >= MAX_SHARED_RUN:
            reason = f"borrowed a {run}-word phrase"
        else:
            rows.append({"claim": found.group(1), "topic": title, "arxiv_id": arxiv_id,
                         "shared_run": run, "note": f"source=v2-planted-gap; seed={label}"})
            if len(rows) == count:
                break
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return rows


def abstention(test: Sequence[dict[str, Any]], *, size: int, label: str,
               unanswerable: Sequence[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    rows = [{**test[i], "hide": list(test[i]["expect_papers"])}
            for i in drawn(len(test), size, label)]
    return rows + list(unanswerable)


def split_topics(topics: Sequence[str]) -> tuple[list[str], list[str]]:
    """Your topics, half to the first evaluation and half sealed, by the seed."""
    if len(set(topics)) != TOPICS or len(topics) != TOPICS:
        raise SystemExit(f"error: write {TOPICS} different topics, one per line")
    first = set(drawn(TOPICS, TOPICS // 2, "reports"))
    return ([t for i, t in enumerate(topics) if i in first],
            [t for i, t in enumerate(topics) if i not in first])


def write_manifest(path: Path, sets: dict[str, tuple[Path, str]], seeds: dict[str, int],
                   authoring: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise SystemExit(f"error: {path} exists. The manifest is written once.")
    payload = {
        "plan": "revision 6",
        "authoring": authoring,
        "sets": {name: {"path": os.path.relpath(file, path.parent), "kind": kind,
                        "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
                 for name, (file, kind) in sets.items()},
        "seeds": seeds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["split", "claims", "abstention", "topics", "manifest"])
    parser.add_argument("topics_file", nargs="?", type=Path,
                        help="topics: a text file with your 20 topics, one per line")
    parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    parser.add_argument("--llm", default=MODEL)
    args = parser.parse_args(argv)

    if args.command == "split":
        dev, test = split(EVALS / "heldout-generated.jsonl", EVALS / "heldout-generated-2.jsonl",
                          *(SETS[name][0] for name in ("dev", "test-a")))
        print(f"dev {len(dev)}, test-a {len(test)}; author-cases now appends "
              f"{TEST_SIZE - len(test)} to test-a")
    elif args.command == "claims":
        from verbatim.answers import OllamaCompletion

        completion = OllamaCompletion(args.llm)
        complete = lambda system, user: completion.complete_text(system, user)  # noqa: E731
        with CorpusStore(args.state) as corpus:
            for name, count in CLAIMS:
                path = SETS[name][0]
                if path.exists():
                    print(f"{name}: already written, kept")
                    continue
                exclude = excluded_papers(map(str, QUESTION_KEYS)) | claimed(
                    SETS[other][0] for other, _ in CLAIMS)
                rejected: dict[str, int] = {}
                rows = write_claims(corpus, complete, count=count, label=name,
                                    exclude=exclude, rejected=rejected)
                print(f"{name}: {len(rows)}/{count} kept, rejected {rejected}")
                if len(rows) < count:
                    raise SystemExit(f"error: {name} is short; nothing was written for it")
                write_once(path, CLAIMS_HEADER.format(label=name, run=MAX_SHARED_RUN), rows)
    elif args.command == "abstention":
        unanswerable = [row for row in rows_of(EVALS / "heldout.jsonl") if row.get("expect_none")]
        note = " The last rows are heldout.jsonl's unanswerable cases, written by hand."
        for name, source, added, extra in (("abstention-a", "test-a", unanswerable, note),
                                           ("abstention-b", "test-b", [], "")):
            test = rows_of(SETS[source][0])
            if len(test) != TEST_SIZE:
                raise SystemExit(f"error: {source} holds {len(test)} rows, not {TEST_SIZE}")
            rows = abstention(test, size=ABSTENTION_SIZE, label=name, unanswerable=added)
            header = ABSTENTION_HEADER.format(source=f"{source}.jsonl", label=name, extra=extra)
            write_once(SETS[name][0], header, rows)
            print(f"{name}: {len(rows)}")
    elif args.command == "topics":
        if args.topics_file is None:
            parser.error("topics needs the file with your 20 topics")
        lines = [line.strip() for line in args.topics_file.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.startswith("#")]
        first, sealed = split_topics(lines)
        write_once(SETS["reports-a"][0], TOPICS_HEADER, [{"topic": t} for t in first])
        write_once(SETS["reports-b"][0], TOPICS_HEADER, [{"topic": t} for t in sealed])
        print(f"reports-a: {first}\nreports-b (sealed): {len(sealed)} topics")
    else:
        from v1_arms import model_digests

        missing = [name for name, (path, _) in SETS.items() if not path.exists()]
        if missing:
            raise SystemExit(f"error: not written yet: {', '.join(missing)}")
        digest = model_digests(config.OLLAMA_HOST).get(args.llm)
        payload = write_manifest(V2 / "manifest.json", SETS, {label: seed(label) for label in SEEDS},
                                 {"model": args.llm, "digest": digest})
        print(json.dumps(payload["sets"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
