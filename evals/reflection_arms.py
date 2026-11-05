"""Verifier-driven reflection: does telling the writer what the quote check found
repair an answer better than asking it again?

    uv run python evals/reflection_arms.py --split dev --llm gemma4:e4b-mlx
    uv run python evals/reflection_arms.py --llm gemma4:e4b-mlx
    uv run python evals/reflection_arms.py --report-only

Every question gets one draft, and the three arms start from it:

  plain     the draft, verified: today's `ask`
  resample  a flagged draft is asked for again with the same prompt
  reflect   a flagged draft goes back with what the check found (VERBATIM_REFLECT)

A draft is flagged when a quote is not in its passage, a marker has no verified
citation behind it, a claim-carrying sentence has no marker, or the reply has
no ANSWER line. A draft that cites nothing is an abstention and is never sent
back. Both repairing arms keep the second answer only if it still cites every
passage the draft cited with a verified quote, has no more unbacked sentences
than the draft and has fewer problems, so they differ only in the feedback.
Against plain either can only gain, by construction, which is why the test that
decides is against resample. An unbacked sentence is a marked one with under half
of its content words in the passages it cites.

Rule fixed 2026-10-30, before any arm was run, and scored once over the 488
answerable held-out questions: more clean answers than resample by the two-sided
sign test at p < 0.05, with under 2.5% of bootstrap resamples at or below zero,
and no more answers than plain saying the passages lack the answer while citing
them. Reflect beat resample 71-11 and failed the last count by one answer, whose
draft already had those words, and repairs were found making sentences clean by
marking them with quotes that did not back them (docs/findings.md).

Rule fixed 2026-11-03, before any arm was run on these questions, scored once
over 240 new ones written blind from papers no other key names
(evals/heldout-generated-2.jsonl). That run showed what is changed here, so it
cannot be the test of it. The primary measure is a clean answer: at least one
verified citation, no rejected quote, no [unverified] marker, a marker on every
claim-carrying sentence, and no unbacked sentence. Reflection becomes the default
only if reflect has more clean answers than resample by the two-sided sign test
at p < 0.05, with under 2.5% of bootstrap resamples at or below zero, and adds
words saying the passages lack the answer, while citing them, to no more answers
than resample does. Passing makes it eligible: what repairs newly marked is read
before the default changes. The repair prompt may be tuned on the development
questions (evals/retrieval.jsonl, answerable, not title-seeded) and the spent
held-out questions before the run, which is made once. Reflect beat resample 38-10
and passed; its repairs were read, and it became the default on 2026-11-04
(docs/findings.md). Reported for every arm and deciding nothing: how often the
check fires and the second answer is kept, by kind of problem; clean answers at
higher floors; cited the expected paper; how much of the draft a kept answer keeps
word for word; and the extra calls, tokens and seconds. Results go to evals/results/reflection-<split>.jsonl, checkpointed
after every question, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
from typing import Any

from verbatim import config
from verbatim.answers import (
    ANSWER_SCHEMA,
    SYSTEM_PROMPT,
    Answer,
    FormatError,
    RejectedCitation,
    RoleCompletion,
    Usage,
    answer_from_payload,
    build_prompt,
)
from verbatim.citations import Citation
from verbatim.corpus import CorpusStore
from verbatim.critique import MIN_BACKED_SHARE, SENTENCE, unbacked_sentences, uncited_sentences
from verbatim.distill import DISOWNS_ITS_EVIDENCE
from verbatim.evaluation import EvalCase, current_commit, load_key, source_of
from verbatim.reflection import (
    NO_ANSWER_LINE,
    UNVERIFIED,
    better,
    passage_texts,
    problem_count,
    reflect,
)
from verbatim.retrieval import default_search_options, hybrid_search
from significance import bootstrap_ci, paired_sign_test
import retrieval_ab

ARMS = ("plain", "resample", "reflect")
SPLITS = {"dev": [Path("evals/retrieval.jsonl")], "fresh": [Path("evals/heldout-generated-2.jsonl")]}
# Finding a paper from its own title is not a question anyone asks.
TITLE_SEEDED = "title (auto-seeded floor)"


def questions(split: str) -> list[EvalCase]:
    """The split's answerable questions, without the title-seeded floor."""
    return [
        case for case in load_key(SPLITS[split])
        if not case.expect_none and source_of(case) != TITLE_SEEDED
    ]


def clean(answer: Answer, texts: dict[str, str], floor: float = MIN_BACKED_SHARE) -> bool:
    """A verified citation, nothing the check would send back, and no unbacked sentence."""
    return (
        bool(answer.citations) and problem_count(answer) == 0
        and not unbacked_sentences(answer, texts, floor)
    )


def disowns(answer: Answer) -> bool:
    """Says the passages lack the answer while citing them: distill's check."""
    return bool(answer.citations) and DISOWNS_ITS_EVIDENCE.search(answer.text) is not None


def added_disowning(draft: Answer, answer: Answer) -> bool:
    """Disowns its evidence in words the draft did not have.

    Fixing a rejected quote beside words already there makes an answer count as
    disowning without the repair having written them.
    """
    return disowns(answer) and DISOWNS_ITS_EVIDENCE.search(draft.text) is None


def cited_passages(corpus: CorpusStore, rows: list[dict]) -> dict[str, str]:
    """The text of every passage any answer in `rows` cites."""
    ids = {
        citation["chunk_id"] for row in rows
        for answers in (row["arms"], row["second"]) for answer in answers.values() if answer
        for citation in answer["citations"]
    }
    return {chunk.chunk_id: chunk.text for chunk in corpus.chunks_by_id(sorted(ids))}


def answer_arms(corpus: CorpusStore, embedder: Any, completion: Any, question: str) -> dict:
    """One draft, what each arm makes of it, and what each step spent."""
    matches = hybrid_search(
        corpus, embedder, question, limit=config.ASK_PASSAGES,
        max_per_paper=config.ASK_MAX_PER_PAPER, **default_search_options(),
    )
    spent: dict[str, Usage] = {}
    if not matches:
        declined = Answer(question, "No passages in the corpus match this question.", [], [],
                          declined=True)
        return {"arms": dict.fromkeys(ARMS, declined), "draft": None, "findings": [], "unformatted": False,
                "kept": {"resample": False, "reflect": False},
                "second": {"resample": None, "reflect": None}, "spent": spent}
    papers = corpus.papers_by_id(sorted({match.chunk.arxiv_id for match in matches}))
    prompt = build_prompt(question, matches)

    mark = completion.usage.snapshot()
    try:
        payload, reply = completion.complete(SYSTEM_PROMPT, prompt, ANSWER_SCHEMA), ""
    except FormatError as failure:
        payload, reply = None, failure.reply
    spent["draft"] = completion.usage.since(mark)
    plain = (
        Answer(question, "", [], []) if payload is None
        else answer_from_payload(question, matches, payload, papers)
    )

    mark = completion.usage.snapshot()
    try:
        reflection = reflect(completion, question, matches, papers, payload, reply)
        reflected, found = reflection.answer, list(reflection.findings)
        second = {"reflect": reflection.revision}
    except FormatError:
        reflected, found, second = plain, [NO_ANSWER_LINE], {"reflect": None}
    spent["reflect"] = completion.usage.since(mark)

    mark = completion.usage.snapshot()
    resampled = plain
    second["resample"] = None
    if found:
        try:
            second["resample"] = answer_from_payload(
                question, matches, completion.complete(SYSTEM_PROMPT, prompt, ANSWER_SCHEMA), papers
            )
        except FormatError:
            pass
        if second["resample"] is not None:
            resampled = (
                second["resample"] if payload is None
                else better(plain, second["resample"], passage_texts(matches))
            )
    spent["resample"] = completion.usage.since(mark)
    return {
        "arms": {"plain": plain, "resample": resampled, "reflect": reflected},
        "draft": payload,
        "findings": found,
        "unformatted": payload is None,
        "kept": {arm: second[arm] is not None and answer is second[arm]
                 for arm, answer in (("resample", resampled), ("reflect", reflected))},
        # Kept or not, so a discarded repair can be read.
        "second": second,
        "spent": spent,
    }


def encode_answer(answer: Answer) -> dict:
    payload = answer.to_dict()
    payload.pop("supporting_papers", None)  # derived, not a field
    return payload


def decode_answer(payload: dict) -> Answer:
    return Answer(
        question=payload["question"],
        text=payload["text"],
        citations=[Citation(**c) for c in payload["citations"]],
        rejected=[RejectedCitation(**r) for r in payload["rejected"]],
        declined=payload["declined"],
    )


def verdict(rows: list[dict], texts: dict[str, str]) -> dict:
    """The rule: reflect against resample on clean answers, and the disowning guard."""
    by_question: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for arm in ("resample", "reflect"):
            by_question[row["question"]][arm].append(clean(decode_answer(row["arms"][arm]), texts))
    # Averaged within each question first, so the bootstrap resamples questions.
    diffs = [
        statistics.mean(arms["reflect"]) - statistics.mean(arms["resample"])
        for arms in by_question.values()
    ]
    wins, losses, p = paired_sign_test(diffs)
    low, high, at_or_below = bootstrap_ci(diffs)
    added = {
        arm: sum(added_disowning(decode_answer(row["arms"]["plain"]), decode_answer(row["arms"][arm]))
                 for row in rows)
        for arm in ("resample", "reflect")
    }
    return {
        "questions": len(diffs), "wins": wins, "losses": losses, "p": p,
        "low": low, "high": high, "at_or_below": at_or_below, "added": added,
        "passes": p < 0.05 and at_or_below < 0.025 and added["reflect"] <= added["resample"],
    }


def _kept_word_for_word(draft: Answer, answer: Answer) -> float | None:
    """The share of the draft's sentences the answer keeps word for word, markers aside."""
    def sentences(text: str) -> list[str]:
        unmarked = re.sub(r"\s*\[(?:\d+|unverified)\]", "", text.strip())
        return [s.strip() for s in SENTENCE.split(unmarked) if s.strip()]

    drafted = sentences(draft.text)
    if not drafted:
        return None
    kept = set(sentences(answer.text))
    return sum(sentence in kept for sentence in drafted) / len(drafted)


def report(rows: list[dict], texts: dict[str, str]) -> str:
    """Every arm counted side by side, the problems by kind, the cost, and the verdict."""
    runs = len({row["run"] for row in rows})
    models = rows[0]["models"]
    answers = {arm: [decode_answer(row["arms"][arm]) for row in rows] for arm in ARMS}
    expected = [set(row["expected"]) for row in rows]

    def per_run(count: float) -> str:
        return f"{count / runs:.1f}"

    lines = [
        f"{len({row['question'] for row in rows})} questions, {runs} run(s); writer "
        f"{models['writer']}, reflector {models['reflector']}",
        "",
        f"  {'per run':<34}" + "".join(f"{arm:>10}" for arm in ARMS),
    ]
    measures = [
        ("clean answer", lambda a: clean(a, texts)),
        ("clean, with 60% of words shared", lambda a: clean(a, texts, 0.6)),
        ("clean, with 70% of words shared", lambda a: clean(a, texts, 0.7)),
        ("at least one verified citation", lambda a: bool(a.citations)),
        ("every claim carries a marker",
         lambda a: bool(a.citations) and not uncited_sentences(a.text)),
        ("citations rejected", lambda a: len(a.rejected)),
        ("an [unverified] marker", lambda a: UNVERIFIED in a.text),
        ("an unbacked sentence", lambda a: bool(unbacked_sentences(a, texts))),
        ("disowns its evidence", disowns),
    ]
    for label, measure in measures:
        cells = [per_run(sum(measure(a) for a in answers[arm])) for arm in ARMS]
        lines.append(f"  {label:<34}" + "".join(f"{c:>10}" for c in cells))
    cells = [
        per_run(sum(bool(want & {c.arxiv_id for c in a.citations})
                    for want, a in zip(expected, answers[arm], strict=True)))
        for arm in ARMS
    ]
    lines.append(f"  {'cited the expected paper':<34}" + "".join(f"{c:>10}" for c in cells))
    cells = [f"{statistics.mean(len(a.text.split()) for a in answers[arm]):.0f}" for arm in ARMS]
    lines.append(f"  {'words in an answer, mean':<34}" + "".join(f"{c:>10}" for c in cells))

    flagged = [row for row in rows if row["findings"]]
    lines += ["", f"  the check flagged {len(flagged)} of {len(rows)} drafts"
                  f" ({sum(row['unformatted'] for row in rows)} with no ANSWER line)"]
    for arm in ("resample", "reflect"):
        kept = [row for row in flagged if row["kept"][arm]]
        shares = [
            share for row in kept
            if (share := _kept_word_for_word(decode_answer(row["arms"]["plain"]),
                                             decode_answer(row["arms"][arm]))) is not None
        ]
        word_for_word = f"{statistics.mean(shares):.0%}" if shares else "-"
        lines.append(f"    {arm:<9} kept the second answer {len(kept)} times;"
                     f" a kept answer keeps {word_for_word} of the draft's sentences word for word")
        unbacked = sum(
            row["second"][arm] is not None and not row["kept"][arm]
            and len(unbacked_sentences(decode_answer(row["second"][arm]), texts))
            > len(unbacked_sentences(decode_answer(row["arms"]["plain"]), texts))
            for row in flagged
        )
        lines.append(f"    {arm:<9} discarded {unbacked} because it marks a sentence its passage does not share")

    kinds = [
        ("a rejected quote", lambda a: bool(a.rejected)),
        ("an [unverified] marker", lambda a: UNVERIFIED in a.text),
        ("an unmarked claim", lambda a: bool(a.citations or a.rejected) and bool(uncited_sentences(a.text))),
    ]
    lines += ["", f"  {'drafts with':<26}{'drafts':>8}{'resample fixed':>16}{'reflect fixed':>16}"]
    for label, has in kinds:
        with_it = [row for row in rows if has(decode_answer(row["arms"]["plain"]))]
        fixed = [sum(not has(decode_answer(row["arms"][arm])) for row in with_it)
                 for arm in ("resample", "reflect")]
        lines.append(f"  {label:<26}{len(with_it):>8}{fixed[0]:>16}{fixed[1]:>16}")
    unformatted = [row for row in rows if row["unformatted"]]
    fixed = [sum(bool(decode_answer(row["arms"][arm]).text) for row in unformatted)
             for arm in ("resample", "reflect")]
    lines.append(f"  {'no ANSWER line':<26}{len(unformatted):>8}{fixed[0]:>16}{fixed[1]:>16}")

    lines += ["", f"  {'spent per question':<26}{'calls':>8}{'tokens':>10}{'seconds':>10}"]
    for step in ("draft", "resample", "reflect"):
        spent = [row["spent"][step] for row in rows if step in row["spent"]]
        if spent:
            lines.append(
                f"  {step:<26}{statistics.mean(s[0] for s in spent):>8.2f}"
                f"{statistics.mean(s[1] + s[2] for s in spent):>10.0f}"
                f"{statistics.mean(s[3] for s in spent):>10.1f}"
            )

    found = verdict(rows, texts)
    lines += [
        "",
        f"  reflect against resample, clean answers, over {found['questions']} questions:",
        f"    they differ on {found['wins'] + found['losses']}: reflect better on"
        f" {found['wins']}, worse on {found['losses']}; sign test p {found['p']:.3f}",
        f"    mean difference 95% CI [{found['low']:+.3f}, {found['high']:+.3f}];"
        f" bootstrap resamples at or below zero {found['at_or_below']:.3f}",
        f"    answers given words disowning their evidence: resample {found['added']['resample']},"
        f" reflect {found['added']['reflect']}",
        f"    -> {'PASSES' if found['passes'] else 'does not pass'} the rule",
    ]
    return "\n".join(lines)


def _encode_row(run: int, case: EvalCase, result: dict, models: dict) -> str:
    return json.dumps({
        "run": run,
        "question": case.question,
        "source": source_of(case),
        "expected": list(case.expect_papers),
        "unformatted": result["unformatted"],
        "draft": result["draft"],
        "findings": result["findings"],
        "kept": result["kept"],
        "models": models,
        "commit": current_commit(),
        "spent": {
            step: [usage.calls, usage.prompt_tokens, usage.output_tokens, usage.seconds]
            for step, usage in result["spent"].items()
        },
        "arms": {arm: encode_answer(answer) for arm, answer in result["arms"].items()},
        "second": {arm: None if answer is None else encode_answer(answer)
                   for arm, answer in result["second"].items()},
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=sorted(SPLITS), default="fresh")
    parser.add_argument("--llm", help="The local Ollama model every role uses unless its own is set")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Answer only the first N questions")
    parser.add_argument("--report-only", action="store_true",
                        help="Score the results file without calling the model")
    args = parser.parse_args()

    out = retrieval_ab.RESULTS / f"reflection-{args.split}.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()] if out.exists() else []
    if not args.report_only:
        if not args.llm:
            parser.error("--llm must name a local Ollama model; none is picked for you")
        from verbatim.embedding import FastEmbedEmbedder

        completion = RoleCompletion(args.llm)
        models = {"writer": completion.writer.model, "reflector": completion.reflector.model}
        if rows and rows[0]["models"] != models:
            raise SystemExit(f"error: {out} holds answers from {rows[0]['models']}, not {models};"
                             " move it aside to start again")
        cases = questions(args.split)[: args.limit or None]
        done = {(row["run"], row["question"]) for row in rows}
        todo = [(run, case) for run in range(args.runs) for case in cases
                if (run, case.question) not in done]
        print(f"{len(todo)} question(s) to answer, {len(done)} already in {out}")
        embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
        out.parent.mkdir(parents=True, exist_ok=True)
        with CorpusStore(config.STATE_PATH) as corpus, out.open("a") as sink:
            for number, (run, case) in enumerate(todo, start=1):
                try:
                    result = answer_arms(corpus, embedder, completion, case.question)
                except (RuntimeError, OSError) as error:
                    # Everything so far is flushed, so the same command resumes.
                    print(f"\nSTOPPED after {number - 1} question(s): {error}")
                    break
                line = _encode_row(run, case, result, models)
                sink.write(line + "\n")
                sink.flush()
                rows.append(json.loads(line))
                print(f"  {number}/{len(todo)}", end="\r", flush=True)
        print()
    if not rows:
        print(f"no answers in {out} yet")
        return 1
    with CorpusStore(config.STATE_PATH) as corpus:
        print(report(rows, cited_passages(corpus, rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
