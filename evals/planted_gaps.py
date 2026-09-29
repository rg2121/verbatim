"""Plant gaps whose answer is known, to measure the gap Falsifier.

Each claim is one a real paper refutes, and the Falsifier runs twice. With the
paper visible it must refute the claim: recall, the headline number. With the
paper hidden, a refutation from another paper means the gap was never a gap.
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
        # Over-draw: most candidates are lost to the leak filter.
        picked = rng.sample(candidates, min(args.papers * 5, len(candidates)))

        skipped = 0
        leaky = 0
        for row in picked:
            reply = complete(CLAIM_SYSTEM, f"TITLE: {row['title']}\n\nABSTRACT: {row['summary']}")
            found = CLAIM_LINE.search(reply or "")
            if found is None or not checkable(found.group(1)):
                skipped += 1
                continue
            # A claim borrowing the abstract's phrasing would hand the Falsifier
            # the paper, so the held-out key's five-word limit applies.
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
    # Refuted by the planted paper itself: the strictest reading.
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
