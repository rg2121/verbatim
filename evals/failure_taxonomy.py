"""Why retrieval misses, classified from three mechanical signals.

  rank           where the expected paper landed, if at all
  term_coverage  how much of the question's vocabulary appears in that paper
  neighbour      abstract similarity between the winning paper and the expected one

Low coverage means the paper does not use the question's words; high
similarity means a near neighbour won.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from verbatim import config
from verbatim.corpus import CorpusStore, unpack_vector
from verbatim.embedding import FastEmbedEmbedder
from verbatim.evaluation import load_cases, rank_of_first_hit
from verbatim.retrieval import hybrid_search

STOPWORDS = frozenset("""a an and are as at be by can do does for from how in
into is it its of on or that the their there this to was were what when where
which who why with you your be been being have has had not no than then them
they these those it's does did doing over under between across using use used
about would could should may might will shall must""".split())
# Below this share of the question's content words present in the paper, the
# question and the paper are not speaking the same language.
VOCABULARY_FLOOR = 0.5
# Abstract similarity above which the paper that won is a genuine neighbour
# rather than an unrelated hit.
NEIGHBOUR_SIMILARITY = 0.75


def content_words(question: str) -> list[str]:
    words = re.findall(r"[a-z0-9']+", question.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def coverage(corpus: CorpusStore, arxiv_id: str, question: str) -> float:
    """Share of the question's content words appearing in the paper's text."""
    words = content_words(question)
    if not words:
        return 1.0
    text = " ".join(c.text.lower() for c in corpus.chunks_for_paper(arxiv_id))
    return sum(1 for w in words if w in text) / len(words)


def abstract_similarity(corpus: CorpusStore, model_id: str, one: str, two: str) -> float | None:
    rows = {
        row["arxiv_id"]: unpack_vector(row["vector"])
        for row in corpus.connection.execute(
            "SELECT arxiv_id, vector FROM paper_embeddings"
            " WHERE model_id = ? AND arxiv_id IN (?, ?)",
            (model_id, one, two),
        )
    }
    if one not in rows or two not in rows:
        return None
    return sum(a * b for a, b in zip(rows[one], rows[two], strict=True))


def classify(rank: int | None, covered: float, neighbour: float | None) -> str:
    if rank == 1:
        return "first"
    if rank is not None and rank <= 5:
        return "top five"
    if rank is not None:
        return "found deep (6-10)"
    if covered < VOCABULARY_FLOOR:
        return "miss: question's words are not in the paper"
    if neighbour is not None and neighbour >= NEIGHBOUR_SIMILARITY:
        return "miss: a near neighbour took the top spot"
    return "miss: unexplained"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=config.EVAL_PATH.parent / "heldout.jsonl")
    parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    args = parser.parse_args()
    cases = [c for c in load_cases(args.cases) if not c.expect_none]
    embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
    model_id = embedder.model_id
    counts: dict[str, int] = {}
    rows = []
    with CorpusStore(args.state) as corpus:
        for case in cases:
            matches = hybrid_search(
                corpus, embedder, case.question, limit=10, use_graph=True
            )
            expected = case.expect_papers[0]
            rank = rank_of_first_hit(matches, case.expect_papers)
            covered = coverage(corpus, expected, case.question)
            top = matches[0].chunk.arxiv_id if matches else None
            neighbour = (
                abstract_similarity(corpus, model_id, top, expected)
                if top and top != expected else None
            )
            label = classify(rank, covered, neighbour)
            counts[label] = counts.get(label, 0) + 1
            rows.append({
                "question": case.question, "expected": expected, "rank": rank,
                "coverage": round(covered, 2),
                "neighbour": None if neighbour is None else round(neighbour, 3),
                "top": top, "label": label,
            })

    print(f"{len(cases)} answerable held-out cases, default retrieval\n")
    for label in sorted(counts, key=lambda k: -counts[k]):
        share = counts[label] / len(cases)
        print(f"  {counts[label]:>3}  {share:>5.0%}  {label}")
    print("\nEverything not ranked first:\n")
    for row in sorted(rows, key=lambda r: (r["rank"] is not None, r["rank"] or 0), reverse=True):
        if row["label"] == "first":
            continue
        print(f"  rank {str(row['rank'] or '-'):>2}  vocab {row['coverage']:.2f}  "
              f"neighbour {str(row['neighbour'] or '-'):>6}  {row['question'][:74]}")
    print(json.dumps(rows), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
