"""The reflection experiment's arms and rule, with no model and no real corpus."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
import reflection_arms  # noqa: E402

from verbatim.answers import Answer, FormatError, RejectedCitation, Usage  # noqa: E402
from verbatim.citations import Citation  # noqa: E402
from verbatim.corpus import ChunkMatch  # noqa: E402
from verbatim.evaluation import EvalCase, source_of  # noqa: E402

from helpers import chunked_corpus  # noqa: E402

PASSAGE = "The model reaches 28.4 BLEU on the English-to-German translation task."
UNBACKED = "Larger ensembles would generalise better across unseen languages"
# The one passage `_answer` cites.
TEXTS = {"c1": "A claim that rests on these words and this result."}
DISOWNING = "The passages do not state this result [1]."


class Writer:
    """Drafts from a queue, revises with one payload, and counts every call."""

    def __init__(self, drafts: list, revision: dict | None = None) -> None:
        self.drafts = list(drafts)
        self.revision = revision
        self.prompts: list[str] = []
        self.revised = 0
        self.usage = Usage()

    def complete(self, system, user, schema):
        self.prompts.append(user)
        self.usage.record({})
        draft = self.drafts.pop(0)
        if isinstance(draft, Exception):
            raise draft
        return draft

    def revise(self, system, user, schema):
        self.revised += 1
        self.usage.record({})
        return self.revision


def _arms(tmp_path: Path, monkeypatch, writer_for):
    with chunked_corpus(tmp_path, {"2608.11111v1": [PASSAGE]}) as corpus:
        matches = [ChunkMatch(chunk=c, score=1.0, snippet=c.text) for c in corpus.iter_chunks()]
        monkeypatch.setattr(reflection_arms, "hybrid_search", lambda *args, **kwargs: matches)
        writer = writer_for(matches[0].chunk.chunk_id)
        return writer, reflection_arms.answer_arms(corpus, None, writer, "how good is it?")


def _payload(chunk_id: str, quote: str) -> dict:
    return {"answer": "It scores 28.4 BLEU [1].",
            "citations": [{"marker": 1, "chunk_id": chunk_id, "quoted_text": quote}]}


def test_the_splits_hold_only_answerable_questions_and_no_titles() -> None:
    dev, fresh = reflection_arms.questions("dev"), reflection_arms.questions("fresh")

    assert len(fresh) == 240
    assert not {case.question for case in fresh} & {case.question for case in dev}
    for case in dev + fresh:
        assert not case.expect_none
        assert not source_of(case).startswith("title")


def test_every_arm_starts_from_one_draft(tmp_path: Path, monkeypatch) -> None:
    writer, result = _arms(tmp_path, monkeypatch, lambda chunk_id: Writer(
        [_payload(chunk_id, "It reached 28.4 BLEU on English-to-German"),
         _payload(chunk_id, "28.4 BLEU on the English-to-German")],
        _payload(chunk_id, PASSAGE),
    ))

    # Resampling asks again with exactly the draft's prompt; reflecting revises once.
    assert len(writer.prompts) == 2 and writer.prompts[0] == writer.prompts[1]
    assert writer.revised == 1
    plain, resample, reflect = (result["arms"][arm] for arm in reflection_arms.ARMS)
    assert plain.rejected and not resample.rejected and not reflect.rejected
    assert result["kept"] == {"resample": True, "reflect": True}
    assert [result["spent"][step].calls for step in ("draft", "resample", "reflect")] == [1, 1, 1]


def test_a_clean_draft_costs_one_call_in_every_arm(tmp_path: Path, monkeypatch) -> None:
    writer, result = _arms(tmp_path, monkeypatch, lambda chunk_id: Writer(
        [_payload(chunk_id, PASSAGE)]
    ))

    assert len(writer.prompts) == 1 and writer.revised == 0
    assert result["findings"] == [] and result["kept"] == {"resample": False, "reflect": False}


def test_a_draft_with_no_answer_line_is_asked_for_again_in_both_arms(tmp_path: Path, monkeypatch) -> None:
    _writer, result = _arms(tmp_path, monkeypatch, lambda chunk_id: Writer(
        [FormatError("did not answer in the expected format", "An essay."),
         _payload(chunk_id, PASSAGE)],
        _payload(chunk_id, PASSAGE),
    ))

    assert result["unformatted"]
    assert not result["arms"]["plain"].citations
    assert result["arms"]["resample"].citations and result["arms"]["reflect"].citations


def test_both_arms_discard_a_repair_that_marks_a_sentence_its_passage_does_not_share(
    tmp_path: Path, monkeypatch
) -> None:
    def draft(chunk_id: str) -> dict:
        return {"answer": f"It scores 28.4 BLEU [1]. {UNBACKED}.",
                "citations": [{"marker": 1, "chunk_id": chunk_id, "quoted_text": PASSAGE}]}

    def marked(chunk_id: str) -> dict:
        return {**draft(chunk_id), "answer": f"It scores 28.4 BLEU [1]. {UNBACKED} [1]."}

    _writer, result = _arms(tmp_path, monkeypatch, lambda chunk_id: Writer(
        [draft(chunk_id), marked(chunk_id)], marked(chunk_id)
    ))

    assert result["findings"]
    assert result["kept"] == {"resample": False, "reflect": False}
    assert result["arms"]["reflect"].text == result["arms"]["plain"].text


def test_a_row_keeps_the_draft_as_the_model_wrote_it(tmp_path: Path, monkeypatch) -> None:
    """A repair can be replayed from it: the verified answer has lost its rejected markers."""
    _writer, result = _arms(tmp_path, monkeypatch, lambda chunk_id: Writer(
        [_payload(chunk_id, "It reached 28.4 BLEU"), _payload(chunk_id, PASSAGE)],
        _payload(chunk_id, PASSAGE),
    ))

    line = reflection_arms._encode_row(0, EvalCase("how good is it?"), result, {"writer": "w"})

    assert json.loads(line)["draft"]["citations"][0]["quoted_text"] == "It reached 28.4 BLEU"


def _answer(*, clean: bool, text: str | None = None) -> Answer:
    citation = Citation(
        arxiv_id="2608.11111v1", base_arxiv_id="2608.11111", version=1, title="T",
        authors=["A"], page_number=1, char_start=0, char_end=10, quoted_text="words",
        chunk_id="c1", source_sha256="s", source_url="u", license_url=None, retrieved_at="t",
    )
    if clean:
        return Answer(question="q", text=text or "Claim [1].", citations=[citation], rejected=[])
    return Answer(question="q", text=text or "Claim [unverified].", citations=[],
                  rejected=[RejectedCitation("c1", "word", "quoted text does not appear")])


def _rows(pairs: list[tuple[bool, bool]], *, reflect_text: str = "Claim [1].",
          plain_text: str = "Claim [unverified].") -> list[dict]:
    """One row per question, from (resample clean, reflect clean)."""
    plain = _answer(clean=False, text=plain_text)
    return [
        {"run": 0, "question": f"q{n}", "source": "s", "expected": [], "unformatted": False,
         "findings": ["x"], "kept": {"resample": False, "reflect": True},
         "second": {"resample": None, "reflect": None},
         "models": {"writer": "w", "reflector": "w"},
         "spent": {"draft": [1, 10, 5, 1.0], "resample": [1, 10, 5, 1.0], "reflect": [1, 12, 5, 1.0]},
         "arms": {"plain": reflection_arms.encode_answer(plain),
                  "resample": reflection_arms.encode_answer(_answer(clean=resample)),
                  "reflect": reflection_arms.encode_answer(_answer(clean=reflect, text=reflect_text))}}
        for n, (resample, reflect) in enumerate(pairs)
    ]


def test_reflect_passes_with_more_clean_answers_than_resample_by_both_tests() -> None:
    rows = _rows([(False, True)] * 10 + [(True, True)] * 20)

    found = reflection_arms.verdict(rows, TEXTS)

    assert (found["wins"], found["losses"]) == (10, 0)
    assert found["passes"]


def test_reflect_fails_when_the_arms_split_the_questions_they_differ_on() -> None:
    found = reflection_arms.verdict(
        _rows([(False, True)] * 5 + [(True, False)] * 3 + [(True, True)] * 20), TEXTS
    )

    assert found["p"] > 0.05 and not found["passes"]


def test_reflect_fails_if_it_adds_more_words_disowning_the_evidence_than_resample() -> None:
    rows = _rows([(False, True)] * 10 + [(True, True)] * 20, reflect_text=DISOWNING)

    found = reflection_arms.verdict(rows, TEXTS)

    assert found["added"] == {"resample": 0, "reflect": 30}
    assert not found["passes"]


def test_words_the_draft_already_had_do_not_fail_the_guard() -> None:
    """The held-out run failed by one such answer: a repair fixed the quote beside them."""
    rows = _rows([(False, True)] * 10 + [(True, True)] * 20,
                 reflect_text=DISOWNING, plain_text=DISOWNING)

    found = reflection_arms.verdict(rows, TEXTS)

    assert found["added"] == {"resample": 0, "reflect": 0}
    assert found["passes"]


def test_a_marked_sentence_its_passage_does_not_share_is_not_clean() -> None:
    backed = _answer(clean=True, text="A claim that rests on these words [1].")
    unbacked = _answer(clean=True, text=f"{UNBACKED} [1].")

    assert reflection_arms.clean(backed, TEXTS)
    assert not reflection_arms.clean(unbacked, TEXTS)


def test_the_report_reads_back_every_arm() -> None:
    rows = _rows([(False, True)] * 10 + [(True, True)] * 20)
    rows[0]["kept"]["reflect"] = False
    rows[0]["second"]["reflect"] = reflection_arms.encode_answer(_answer(clean=True, text=f"{UNBACKED} [1]."))

    text = reflection_arms.report(rows, TEXTS)

    assert "clean answer" in text and "resample" in text and "reflect" in text
    assert "reflect   discarded 1 because it marks a sentence its passage does not share" in text
