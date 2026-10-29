"""Chunking A/B: the same papers cut three ways, each searched as the shipped default searches.

    uv run python evals/chunking_ab.py --build
    VERBATIM_CHUNKER=sentences uv run verbatim chunk --state data/evalcorpus-sentences.db \\
        --output data/evalcorpus-sentences.jsonl
    VERBATIM_CHUNKER=sentences uv run verbatim embed --state data/evalcorpus-sentences.db
    (and the same for semantic)
    uv run python evals/chunking_ab.py --arm windows
    uv run python evals/chunking_ab.py --compare windows sentences
    uv run python evals/chunking_ab.py --compare windows sentences --keys evals/retrieval.jsonl

Re-chunking means re-embedding, about 12 hours for the full corpus, so the arms
run on an eval corpus (sampling.export_eval_corpus): every paper the keys name,
each one's nearest paper by abstract, and a seeded random fill. windows holds
today's passages and vectors, copied; the other arms are cut again from the same
source files by `verbatim chunk`. Ranks go to evals/results/chunking-<arm>.json,
or chunking-<arm>@<key>.json for another key; git ignores both.

Experiment 3, chunking. Rule fixed 2026-10-28, before any arm was built. Two
arms are tested against windows: sentences (whole sentences, as many as fit in
1,200 characters, the 200-character overlap made of whole sentences) and
semantic (each passage at least 600 characters, ending at the sentence boundary
in its room where the sentences either side are least alike to bge-small).
Each is compared with windows on per-case reciprocal rank over the 488
held-out cases, searched with the shipped default. Each must clear p < 0.025 by
the two-sided sign test with under 2.5% of bootstrap resamples at or below
zero, and be worse on no more development-key cases than it is better on.
Passing makes an arm a candidate, not the default: the default changes only if
the same comparison holds on the full corpus. Passage counts and lengths, the
share of passages ending at a sentence end, and build times are reported for
every arm; answers over a fixed sample, from a local model, are reported and
decide nothing.

Exploratory, added after windows scored mrr 0.969 here (23 first-place misses
in 488) and before any other arm was scored: the same comparisons searched with
plain hybrid (`--search hybrid`: no graph leg, no reranker), which leaves more
misses for a chunking to move. They decide nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
from typing import Any

from verbatim import config
from verbatim.corpus import CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.evaluation import current_commit, load_key
import retrieval_ab

EVAL_CORPUS = Path("data/evalcorpus.db")
ARMS = {
    "windows": EVAL_CORPUS,
    "sentences": Path("data/evalcorpus-sentences.db"),
    "semantic": Path("data/evalcorpus-semantic.db"),
}
PAPERS = 1_500
SEED = 0
# The eval corpus holds every paper these keys name, so any of them can be scored.
NAMED_BY = [*retrieval_ab.KEYS, Path("evals/retrieval.jsonl"), Path("evals/conversational.jsonl")]
# The shipped default, by name, so a later default cannot change what this measured.
SEARCH = "hybrid+graph+colbert-gated-skipdefs"


def named_papers(keys: list[Path]) -> list[str]:
    """Every paper a key names, including each alternative in a compound group."""
    papers: set[str] = set()
    for case in load_key(keys):
        for group in (*case.expect_papers, *case.require_papers):
            papers.update(group.split("|"))
    return sorted(papers)


def ends_a_sentence(text: str) -> bool:
    return text.rstrip("\"')]").endswith((".", "!", "?"))


def describe(corpus: CorpusStore) -> dict[str, Any]:
    """What an arm's searchable passages look like, and which papers they come from."""
    rows = corpus.connection.execute(
        "SELECT arxiv_id, chunker_version, text FROM chunks WHERE kind != 'references'"
    ).fetchall()
    papers = sorted({row["arxiv_id"] for row in rows})
    lengths = [len(row["text"]) for row in rows]
    return {
        "passages": len(rows),
        "papers": len(papers),
        "papers_sha": hashlib.sha256("\n".join(papers).encode()).hexdigest()[:16],
        "chunkers": sorted({row["chunker_version"] for row in rows}),
        "mean_chars": round(statistics.mean(lengths)) if lengths else 0,
        "median_chars": round(statistics.median(lengths)) if lengths else 0,
        "sentence_ends": round(sum(ends_a_sentence(row["text"]) for row in rows) / max(len(rows), 1), 3),
    }


def cut_by(chunker_version: str) -> str:
    """The way a fingerprint says passages were cut: windows has none named."""
    parts = chunker_version.split(":")
    return parts[3] if len(parts) > 3 else "windows"


def check_built(arm: str, chunkers: list[str]) -> None:
    """Refuse to score an arm whose passages were not cut the arm's way."""
    if not chunkers or any(cut_by(version) != arm for version in chunkers):
        raise SystemExit(
            f"error: {arm} is not built: its passages come from {', '.join(chunkers) or 'nothing'}; "
            f"run VERBATIM_CHUNKER={arm} verbatim chunk and embed on {ARMS[arm]} first"
        )


def results_path(arm: str, keys: list[Path], search: str = SEARCH) -> Path:
    searched = "" if search == SEARCH else f".{search}"
    other = "" if keys == retrieval_ab.KEYS else "@" + "+".join(path.stem for path in keys)
    return retrieval_ab.RESULTS / f"chunking-{arm}{searched}{other}.json"


def build(source: Path) -> None:
    from verbatim.sampling import export_eval_corpus

    EVAL_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    model_id = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL).model_id
    report = export_eval_corpus(
        source, EVAL_CORPUS, required=named_papers(NAMED_BY), total=PAPERS,
        model_id=model_id, seed=SEED,
    )
    print(f"wrote {EVAL_CORPUS}: {report.render()}")
    # Each other arm is cut again from a copy of its own.
    for arm, path in ARMS.items():
        if path != EVAL_CORPUS:
            shutil.copyfile(EVAL_CORPUS, path)
            print(f"copied it to {path} for {arm}")


def run_arm(arm: str, keys: list[Path], search: str = SEARCH) -> Path:
    cases = [c for c in load_key(keys) if not c.expect_none and not c.is_compound]
    embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
    with CorpusStore(ARMS[arm]) as corpus:
        built = describe(corpus)
        check_built(arm, built["chunkers"])
        rows = retrieval_ab.rank_cases(corpus, embedder, cases, retrieval_ab.options_for(search))
    out = results_path(arm, keys, search)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "arm": arm, "state": str(ARMS[arm]), "search": search, "commit": current_commit(),
        "keys": [str(k) for k in keys], **built, "rows": rows,
    }, indent=1))
    return out


def answer_arm(arm: str, *, sample: int, llm: str) -> Path:
    """Answer a fixed stratified sample on one arm; counted, as eval-answers counts, with no judge."""
    from verbatim.answer_eval import AnsweredCase
    from verbatim.answers import OllamaCompletion, answer_from_matches
    from verbatim.critique import uncited_sentences
    from verbatim.evaluation import stratified_sample
    from verbatim.reranking import make_reranker
    from verbatim.retrieval import hybrid_search

    answerable = [c for c in load_key(retrieval_ab.KEYS) if not c.expect_none and not c.is_compound]
    cases = stratified_sample(answerable, size=sample, seed=SEED)
    options = retrieval_ab.options_for(SEARCH)
    options = {**options, "reranker": make_reranker(options["reranker"])}
    embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
    completion = OllamaCompletion(llm)
    rows = []
    with CorpusStore(ARMS[arm]) as corpus:
        check_built(arm, describe(corpus)["chunkers"])
        for number, case in enumerate(cases, start=1):
            matches = hybrid_search(
                corpus, embedder, case.question, limit=config.ASK_PASSAGES,
                max_per_paper=config.ASK_MAX_PER_PAPER, **options,
            )
            answer = answer_from_matches(corpus, completion, case.question, matches)
            answered = AnsweredCase(case.question, answer, case.expect_papers)
            rows.append({
                "question": case.question,
                "verified": len(answer.citations),
                "rejected": len(answer.rejected),
                "declined": answer.declined,
                "cited_expected": answered.cited_an_expected_paper,
                "fully_cited": bool(answer.citations) and not uncited_sentences(answer.text),
            })
            print(f"  answered {number}/{len(cases)}", end="\r", flush=True)
    out = retrieval_ab.RESULTS / f"chunking-answers-{arm}.json"
    out.write_text(json.dumps({"arm": arm, "llm": llm, "sample": sample, "seed": SEED,
                               "commit": current_commit(), "rows": rows}, indent=1))
    return out


def answer_report(arms: list[str]) -> str:
    """Each arm's answers, counted side by side; generation is not seeded, so small gaps are noise."""
    results = [json.loads((retrieval_ab.RESULTS / f"chunking-answers-{arm}.json").read_text()) for arm in arms]
    lines = [f"  {'arm':<10} answered  any verified  fully cited  cited expected  rejected quotes"]
    for result in results:
        rows = result["rows"]
        lines.append(
            f"  {result['arm']:<10} {len(rows):>8}  {sum(r['verified'] > 0 for r in rows):>12}"
            f"  {sum(r['fully_cited'] for r in rows):>11}  {sum(r['cited_expected'] for r in rows):>14}"
            f"  {sum(r['rejected'] for r in rows):>15}"
        )
    return "\n".join(lines)


def load(arm: str, keys: list[Path], search: str = SEARCH) -> dict[str, Any]:
    path = results_path(arm, keys, search)
    if not path.exists():
        raise SystemExit(f"error: no results for {arm!r} on these keys; run --arm {arm} first")
    return json.loads(path.read_text())


def report(first: dict[str, Any], second: dict[str, Any]) -> str:
    if first["papers_sha"] != second["papers_sha"]:
        raise SystemExit(
            "error: the arms hold different papers, so a difference could be the papers "
            "rather than the chunking; rebuild them from one eval corpus"
        )
    lines = [f"  {'arm':<10} cases   r@1   r@5  r@10    mrr  passages  median chars  end a sentence"]
    for result in (first, second):
        s = retrieval_ab.summarise(result["rows"])
        lines.append(
            f"  {result['arm']:<10} {s['cases']:>5}  {s['r@1']:.2f}  {s['r@5']:.2f}  {s['r@10']:.2f}"
            f"  {s['mrr']:.3f}  {result['passages']:>8,}  {result['median_chars']:>12,}"
            f"  {result['sentence_ends']:>14.0%}"
        )
    sources = sorted({row["source"] for row in first["rows"]})
    if len(sources) > 1:
        lines.append("\n  mrr by batch:")
        for source in sources:
            pair = [retrieval_ab.summarise([r for r in result["rows"] if r["source"] == source])
                    for result in (first, second)]
            lines.append(f"    {source:<24} {pair[0]['mrr']:.3f} -> {pair[1]['mrr']:.3f}"
                         f"  ({pair[0]['cases']} cases)")
    verdict = retrieval_ab.compare(first["rows"], second["rows"])
    low, high = verdict["ci95"]
    lines += [
        f"\n  {second['arm']} minus {first['arm']}, both searched with {first['search']}:",
        f"    mrr {verdict['mean_difference']:+.3f}, better on {verdict['wins']}, "
        f"worse on {verdict['losses']}, sign test p = {verdict['p']:.4f}",
        f"    bootstrap 95% [{low:+.3f}, {high:+.3f}], "
        f"{verdict['at_or_below_zero']:.1%} of resamples at or below zero",
        f"\n  the same {first['papers']:,} papers; measured on {first['commit']} and {second['commit']}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true", help="Write the eval corpus and a copy per arm")
    action.add_argument("--arm", choices=sorted(ARMS), help="Rank every case on this arm's corpus")
    action.add_argument("--compare", nargs=2, metavar=("FIRST", "SECOND"), choices=sorted(ARMS))
    action.add_argument("--answers", choices=sorted(ARMS), help="Answer a fixed sample on this arm")
    action.add_argument("--answer-report", action="store_true", help="Count every arm's answers")
    parser.add_argument("--state", type=Path, default=config.STATE_PATH, help="The corpus --build samples")
    parser.add_argument("--keys", type=Path, nargs="+", default=retrieval_ab.KEYS)
    parser.add_argument(
        "--search", default=SEARCH, choices=sorted(retrieval_ab.ARMS),
        help="The retrieval arm to search with; any but the shipped default is exploratory",
    )
    parser.add_argument("--sample", type=int, default=60, help="Questions --answers answers")
    # A -cloud model would send passages off the machine, so one must be named.
    parser.add_argument("--llm", help="The local Ollama model --answers uses")
    args = parser.parse_args()

    if args.build:
        build(args.state)
    elif args.arm:
        print(f"wrote {run_arm(args.arm, args.keys, args.search)}")
    elif args.answers:
        if not args.llm:
            raise SystemExit("error: --answers needs --llm naming a local Ollama model")
        print(f"wrote {answer_arm(args.answers, sample=args.sample, llm=args.llm)}")
    elif args.answer_report:
        print(answer_report([arm for arm in ARMS if (retrieval_ab.RESULTS / f"chunking-answers-{arm}.json").exists()]))
    else:
        first, second = (load(arm, args.keys, args.search) for arm in args.compare)
        print(report(first, second))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
