"""Ground truth for an absence-detector, by planting gaps whose answer is known.

A standing gap is only worth anything if the Falsifier would have found the
paper had it been there. So the experiment is: take a real paper, write a claim
that *this paper refutes*, and run the Falsifier twice.

  paper visible   the Falsifier must refute the claim. Failing here is the
                  dangerous error - it would confirm a gap that is not one.
  paper hidden    the Falsifier cannot reach that paper. If it still refutes,
                  another paper covers the same ground, which is correct
                  behaviour and says the "gap" was never one to begin with.

**The headline number is recall with the paper visible.** That is the
Falsifier's ability to find evidence that demonstrably exists, and it is what
licenses reading a standing gap as absence rather than as a failed search.

The hidden arm is deliberately *not* scored as accuracy. Hiding one paper does
not create absence: the corpus holds 7,921 papers and neighbours overlap. A
refutation from another paper is the system being right, not wrong, so that
count is reported as what it is - how often a one-paper gap was never a gap.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from verbatim import config
from verbatim.answers import OllamaCompletion
from verbatim.corpus import CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.authoring import MAX_SHARED_RUN, longest_shared_run
from verbatim.gaps import REFUTED, Gap, checkable, falsify

CLAIM_SYSTEM = """You are given the title and abstract of a research paper.

Write one claim that this paper disproves - a statement that no work exists on what this paper in fact did.

Rules:
- The claim must be specific enough that this paper alone refutes it.
- Name the subject and the thing supposedly not done. Write "no paper evaluates
  sparse attention on multilingual retrieval".
- Do not write "more work is needed" or "under-explored"; those cannot be
  disproved.
- Do not name the paper, its authors or its title.

Reply with exactly one line and nothing else:

CLAIM: <claim>"""

CLAIM_LINE = re.compile(r"^\s*CLAIM:\s*(.+?)\s*$", re.MULTILINE)
OUT = Path("evals/results/planted_gaps.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm")
    args = parser.parse_args()

    embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
    completion = OllamaCompletion(args.llm or config.OLLAMA_MODEL)
    complete = lambda system, user: completion.complete_text(system, user)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with CorpusStore(config.STATE_PATH) as corpus, OUT.open("w") as sink:
        candidates = corpus.connection.execute(
            """
            SELECT p.arxiv_id, p.title, p.summary
            FROM papers AS p
            JOIN chunks AS c ON c.arxiv_id = p.arxiv_id AND c.kind = 'body'
            WHERE p.is_latest = 1 AND p.summary IS NOT NULL AND p.summary != ''
            GROUP BY p.arxiv_id
            HAVING COUNT(c.chunk_id) BETWEEN 25 AND 90
            ORDER BY p.arxiv_id
            """
        ).fetchall()
        rng = random.Random(args.seed)
        # Over-draw: most candidates are lost to the leak filter, and a run
        # that quietly returns eight gaps is not the experiment asked for.
        picked = rng.sample(candidates, min(args.papers * 5, len(candidates)))

        skipped = 0
        leaky = 0
        for row in picked:
            reply = complete(CLAIM_SYSTEM, f"TITLE: {row['title']}\n\nABSTRACT: {row['summary']}")
            found = CLAIM_LINE.search(reply or "")
            if found is None or not checkable(found.group(1)):
                skipped += 1
                continue
            # The abstract's own phrasing would hand the Falsifier the
            # paper. Rejected by the threshold the held-out key already uses:
            # the first run of this measured 100% and 16 of 25 claims had
            # borrowed a five-or-more-word run from the abstract.
            borrowed = longest_shared_run(found.group(1), row["summary"])
            if borrowed >= MAX_SHARED_RUN:
                leaky += 1
                continue
            gap = Gap(topic=row["title"], claim=found.group(1))
            try:
                visible = falsify(corpus, embedder, completion, complete, gap)
                hidden = falsify(
                    corpus, embedder, completion, complete, gap,
                    hidden=[row["arxiv_id"]],
                )
            except (RuntimeError, OSError) as error:
                print(f"\nSTOPPED: {error}")
                break
            record = {
                "arxiv_id": row["arxiv_id"],
                "claim": gap.claim,
                "shared_run": borrowed,
                "visible_verdict": visible.verdict,
                "visible_cited": (
                    visible.citation.arxiv_id if visible.citation else None
                ),
                "hidden_verdict": hidden.verdict,
                "hidden_cited": hidden.citation.arxiv_id if hidden.citation else None,
                "visible_papers": len(visible.effort.papers),
                "hidden_papers": len(hidden.effort.papers),
            }
            rows.append(record)
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            print(f"  {len(rows)}/{args.papers} kept", end="\r", flush=True)
            if len(rows) >= args.papers:
                break

    if not rows:
        print("\nno planted gaps were usable")
        return 1

    refuted_visible = [r for r in rows if r["visible_verdict"] == REFUTED]
    # Found the planted paper itself, rather than something else that happens
    # to fit - the strictest reading of the same result.
    found_the_paper = [
        r for r in refuted_visible if r["visible_cited"] == r["arxiv_id"]
    ]
    refuted_hidden = [r for r in rows if r["hidden_verdict"] == REFUTED]

    n = len(rows)
    print(f"\n\n{n} planted gap(s), each refutable by one known paper"
          f"\n  {skipped} rejected as unfalsifiable, {leaky} as borrowing a "
          f"{MAX_SHARED_RUN}+ word run from the abstract they were written from\n")
    print("  WITH the paper visible - the Falsifier should always refute:")
    print(f"    refuted                       {len(refuted_visible)}/{n}"
          f" ({len(refuted_visible) / n:.0%})")
    print(f"      citing the planted paper    {len(found_the_paper)}/{n}"
          f" ({len(found_the_paper) / n:.0%})")
    print(f"      citing a different paper    {len(refuted_visible) - len(found_the_paper)}")
    print(f"    MISSED evidence that is there {n - len(refuted_visible)}"
          "   <- would confirm a false gap")
    print("\n  WITH the paper hidden - not an accuracy score, see the header:")
    print(f"    still refuted, by a neighbour {len(refuted_hidden)}/{n}"
          f" ({len(refuted_hidden) / n:.0%})")
    print(f"    the gap stood                 {n - len(refuted_hidden)}/{n}")
    print(f"\n  Written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
