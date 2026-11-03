"""Reflection: what the quote check found goes back to the writer, once."""

from pathlib import Path

import pytest

from verbatim import config
from verbatim.answers import FormatError, answer_from_matches, answer_from_payload, verify_quote
from verbatim.corpus import ChunkMatch
from verbatim.reflection import (
    HINT_MAX_WORDS,
    better,
    closest_words,
    findings,
    passage_texts,
    problem_count,
    reflect,
)

from helpers import chunked_corpus

PASSAGE = (
    "Transformers replace recurrence with attention. "
    "The model reaches 28.4 BLEU on the English-to-German translation task. "
    "Training took three and a half days on eight GPUs."
)
OTHER = "Dropout of 0.1 was applied to the output of each sub-layer."
SENTENCE = "The model reaches 28.4 BLEU on the English-to-German translation task."
PARAPHRASE = "The model reached 28.4 BLEU on the English-to-German task"
# A claim that shares nothing with either passage.
UNBACKED = "Larger ensembles would generalise better across unseen languages"


class Scripted:
    """A writer that drafts one payload and revises with the next, recording each revision's prompt."""

    def __init__(self, draft, *revisions) -> None:
        self.draft = draft
        self.revisions = list(revisions)
        self.revised: list[tuple[str, str]] = []

    def complete(self, system, user, schema):
        if isinstance(self.draft, Exception):
            raise self.draft
        return self.draft

    def revise(self, system, user, schema):
        self.revised.append((system, user))
        revision = self.revisions.pop(0)
        if isinstance(revision, Exception):
            raise revision
        return revision


def _corpus(tmp_path: Path):
    return chunked_corpus(tmp_path, {"2608.11111v1": [PASSAGE], "2608.22222v1": [OTHER]})


def _matches(corpus) -> list[ChunkMatch]:
    return [ChunkMatch(chunk=c, score=1.0, snippet=c.text) for c in corpus.iter_chunks()]


def _ids(matches) -> tuple[str, str]:
    by_paper = {m.chunk.arxiv_id: m.chunk.chunk_id for m in matches}
    return by_paper["2608.11111v1"], by_paper["2608.22222v1"]


def _papers(corpus, matches):
    return corpus.papers_by_id(sorted({m.chunk.arxiv_id for m in matches}))


def _cite(marker: int, chunk_id: str, quote: str) -> dict:
    return {"marker": marker, "chunk_id": chunk_id, "quoted_text": quote}


def _verified(corpus, matches, payload):
    return answer_from_payload("q", matches, payload, _papers(corpus, matches))


def test_the_closest_words_are_the_sentence_a_paraphrase_came_from() -> None:
    hint = closest_words(PARAPHRASE, PASSAGE)

    assert hint == SENTENCE
    # Copied as given, the hint passes the check the paraphrase failed.
    assert verify_quote(hint, PASSAGE) is None


def test_a_quote_across_a_sentence_boundary_is_offered_both_sentences() -> None:
    hint = closest_words("replaces recurrence with attention, and reaches 28.4 BLEU", PASSAGE)

    assert hint == f"Transformers replace recurrence with attention. {SENTENCE}"


def test_no_words_are_offered_when_under_half_the_quote_is_there() -> None:
    """An unrelated sentence offered as a hint invites a quote that does not say the claim."""
    assert closest_words("convolutions outperform attention on every benchmark", PASSAGE) is None


def test_a_long_sentence_is_cut_to_the_words_around_the_match() -> None:
    filler = " ".join(f"word{n}" for n in range(80))
    passage = f"{filler} the decoder attends to every encoder position {filler}."

    hint = closest_words("the decoder attends to each encoder position", passage)

    assert hint is not None and "the decoder attends to every encoder position" in hint
    assert len(hint.split()) <= HINT_MAX_WORDS
    assert verify_quote(hint, passage) is None


def test_a_rejected_quote_is_named_with_the_closest_words_in_its_passage(tmp_path: Path) -> None:
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        first, _ = _ids(matches)
        payload = {"answer": "It scores 28.4 BLEU [1].", "citations": [_cite(1, first, PARAPHRASE)]}
        draft = _verified(corpus, matches, payload)

    found = findings(payload, draft, matches)

    assert len(found) == 1
    assert found[0].startswith(f"CITE [1] quotes words that are not in passage {first}")
    assert PARAPHRASE in found[0] and SENTENCE in found[0]


def test_words_cited_to_the_wrong_passage_are_pointed_at_the_right_one(tmp_path: Path) -> None:
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        first, second = _ids(matches)
        payload = {"answer": "Dropout was applied at 0.1 [1].",
                   "citations": [_cite(1, first, "Dropout of 0.1 was applied")]}
        draft = _verified(corpus, matches, payload)

    (found,) = findings(payload, draft, matches)

    assert f"word for word in passage {second}" in found


def test_a_passage_that_was_never_given_is_named(tmp_path: Path) -> None:
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        first, _ = _ids(matches)
        payload = {"answer": "Training took three and a half days [1].",
                   "citations": [_cite(1, "feedface00000000", "Training took three and a half days")]}
        draft = _verified(corpus, matches, payload)

    (found,) = findings(payload, draft, matches)

    assert "feedface00000000, which is not one of the passages given" in found
    assert f"word for word in passage {first}" in found


def test_a_marker_with_no_cite_line_is_named(tmp_path: Path) -> None:
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        first, _ = _ids(matches)
        payload = {"answer": "Attention replaces recurrence [1]. Dropout was 0.1 [2].",
                   "citations": [_cite(1, first, "Transformers replace recurrence with attention")]}
        draft = _verified(corpus, matches, payload)

    assert findings(payload, draft, matches) == [
        "The answer uses [2], but there is no CITE [2] line."
    ]


def test_a_claim_with_no_marker_is_named(tmp_path: Path) -> None:
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        first, _ = _ids(matches)
        payload = {"answer": "Attention replaces recurrence in this model [1]. "
                             "Training took three and a half days on eight GPUs.",
                   "citations": [_cite(1, first, "Transformers replace recurrence with attention")]}
        draft = _verified(corpus, matches, payload)

    assert findings(payload, draft, matches) == [
        'This sentence makes a claim but has no marker: '
        '"Training took three and a half days on eight GPUs."'
    ]


def test_an_answer_that_cites_nothing_is_an_abstention_with_nothing_to_find(tmp_path: Path) -> None:
    """Pushing it to cite would push it to answer what the passages do not cover."""
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        payload = {"answer": "These passages are about translation, not protein folding, "
                             "so they cannot answer the question.", "citations": []}
        draft = _verified(corpus, matches, payload)

    assert problem_count(draft) == 0
    assert findings(payload, draft, matches) == []


def test_a_clean_draft_has_nothing_to_find(tmp_path: Path) -> None:
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        first, _ = _ids(matches)
        payload = {"answer": "It reaches 28.4 BLEU on English to German [1].",
                   "citations": [_cite(1, first, SENTENCE)]}
        draft = _verified(corpus, matches, payload)

    assert findings(payload, draft, matches) == []


def _kept(tmp_path: Path, payloads):
    """The draft, the revision, and which `better` keeps, from `payloads(first, second)`."""
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        draft_payload, revision_payload = payloads(*_ids(matches))
        draft = _verified(corpus, matches, draft_payload)
        revision = _verified(corpus, matches, revision_payload)
    return draft, revision, better(draft, revision, passage_texts(matches))


def test_a_revision_that_drops_a_verified_passage_is_discarded(tmp_path: Path) -> None:
    draft, revision, kept = _kept(tmp_path, lambda first, second: (
        {"answer": "It reaches 28.4 BLEU on this task [1]. It trains for three and a half days.",
         "citations": [_cite(1, first, SENTENCE)]},
        {"answer": "Dropout is 0.1 on each sub-layer [1].",
         "citations": [_cite(1, second, "Dropout of 0.1 was applied")]},
    ))

    assert problem_count(revision) < problem_count(draft)
    assert kept is draft


def test_a_revision_no_better_than_the_draft_is_discarded(tmp_path: Path) -> None:
    draft, _revision, kept = _kept(tmp_path, lambda first, _second: (
        {"answer": "It reaches 28.4 BLEU on this task [1]. It trains for three and a half days.",
         "citations": [_cite(1, first, SENTENCE)]},
        {"answer": "It reaches 28.4 BLEU on this task [1]. It trains on eight GPUs in all.",
         "citations": [_cite(1, first, SENTENCE)]},
    ))

    assert kept is draft


def test_a_revision_that_keeps_every_verified_passage_and_fixes_more_is_kept(tmp_path: Path) -> None:
    _draft, revision, kept = _kept(tmp_path, lambda first, _second: (
        {"answer": "It reaches 28.4 BLEU on this task [1]. It trains for three and a half days.",
         "citations": [_cite(1, first, SENTENCE)]},
        {"answer": "It reaches 28.4 BLEU on this task [1]. It trains for three and a half days [2].",
         "citations": [_cite(1, first, SENTENCE), _cite(2, first, "three and a half days")]},
    ))

    assert kept is revision


def test_a_marker_on_a_sentence_its_passage_does_not_share_is_no_fix(tmp_path: Path) -> None:
    """The quote's words are in the passage; the sentence's are not. It looked clean."""
    draft, revision, kept = _kept(tmp_path, lambda first, _second: (
        {"answer": f"It reaches 28.4 BLEU on this task [1]. {UNBACKED}.",
         "citations": [_cite(1, first, SENTENCE)]},
        {"answer": f"It reaches 28.4 BLEU on this task [1]. {UNBACKED} [1].",
         "citations": [_cite(1, first, SENTENCE)]},
    ))

    assert problem_count(revision) < problem_count(draft)
    assert kept is draft


def test_fixing_a_quote_does_not_buy_a_marker_its_passage_does_not_share(tmp_path: Path) -> None:
    draft, revision, kept = _kept(tmp_path, lambda first, _second: (
        {"answer": f"It scores 28.4 BLEU [1]. {UNBACKED}.",
         "citations": [_cite(1, first, PARAPHRASE)]},
        {"answer": f"It scores 28.4 BLEU [1]. {UNBACKED} [1].",
         "citations": [_cite(1, first, SENTENCE)]},
    ))

    assert problem_count(revision) == 0 and problem_count(draft) == 3
    assert kept is draft


def test_a_repair_may_remove_a_sentence_no_passage_shares(tmp_path: Path) -> None:
    _draft, revision, kept = _kept(tmp_path, lambda first, _second: (
        {"answer": f"It reaches 28.4 BLEU on this task [1]. {UNBACKED}.",
         "citations": [_cite(1, first, SENTENCE)]},
        {"answer": "It reaches 28.4 BLEU on this task [1].",
         "citations": [_cite(1, first, SENTENCE)]},
    ))

    assert kept is revision


def test_a_repair_is_held_only_to_what_it_adds(tmp_path: Path) -> None:
    """A marker the draft already had is no worse for the repair's having kept it."""
    _draft, revision, kept = _kept(tmp_path, lambda first, _second: (
        {"answer": f"{UNBACKED} [1]. Training took three and a half days on eight GPUs.",
         "citations": [_cite(1, first, SENTENCE)]},
        {"answer": f"{UNBACKED} [1]. Training took three and a half days on eight GPUs [1].",
         "citations": [_cite(1, first, SENTENCE)]},
    ))

    assert kept is revision


def test_a_repair_that_only_adds_such_a_marker_leaves_the_draft_standing(tmp_path: Path) -> None:
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        first, _ = _ids(matches)
        draft = {"answer": f"It reaches 28.4 BLEU on this task [1]. {UNBACKED}.",
                 "citations": [_cite(1, first, SENTENCE)]}
        marked = {"answer": f"It reaches 28.4 BLEU on this task [1]. {UNBACKED} [1].",
                  "citations": [_cite(1, first, SENTENCE)]}
        writer = Scripted(draft, marked)
        result = reflect(writer, "q", matches, _papers(corpus, matches), draft)

    assert result.revision is not None and result.answer is result.draft


def test_a_clean_draft_is_not_sent_back(tmp_path: Path) -> None:
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        first, _ = _ids(matches)
        payload = {"answer": "It reaches 28.4 BLEU on English to German [1].",
                   "citations": [_cite(1, first, SENTENCE)]}
        writer = Scripted(payload)
        result = reflect(writer, "q", matches, _papers(corpus, matches), payload)

    assert writer.revised == []
    assert result.findings == () and result.answer is result.draft


def test_a_flagged_draft_goes_back_once_with_what_the_check_found(tmp_path: Path) -> None:
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        first, _ = _ids(matches)
        draft = {"answer": "It scores 28.4 BLEU [1].", "citations": [_cite(1, first, PARAPHRASE)]}
        fixed = {"answer": "It scores 28.4 BLEU [1].", "citations": [_cite(1, first, SENTENCE)]}
        writer = Scripted(draft, fixed)
        result = reflect(writer, "how good is it?", matches, _papers(corpus, matches), draft)

    ((system, user),) = writer.revised
    assert "checker" in system
    # The draft goes back as the model wrote it, beside what was found in it.
    assert f"CITE [1] {first} :: {PARAPHRASE}" in user
    assert SENTENCE in user and "how good is it?" in user
    assert result.answer is result.revision
    assert result.answer.rejected == [] and result.answer.citations[0].quoted_text == SENTENCE


def test_an_unreadable_revision_leaves_the_draft_standing(tmp_path: Path) -> None:
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        first, _ = _ids(matches)
        draft = {"answer": "It scores 28.4 BLEU [1].", "citations": [_cite(1, first, PARAPHRASE)]}
        writer = Scripted(draft, FormatError("no ANSWER line", "Here is my revision."))
        result = reflect(writer, "q", matches, _papers(corpus, matches), draft)

    assert result.revision is None and result.answer is result.draft


def test_a_reply_with_no_answer_line_is_asked_for_once_more(tmp_path: Path) -> None:
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        first, _ = _ids(matches)
        fixed = {"answer": "It scores 28.4 BLEU [1].", "citations": [_cite(1, first, SENTENCE)]}
        writer = Scripted(None, fixed)
        result = reflect(writer, "q", matches, _papers(corpus, matches), None,
                         "I would rather write an essay.")

    ((_system, user),) = writer.revised
    assert "ANSWER:" in user and "I would rather write an essay." in user
    assert result.draft is None and result.answer.citations


def test_a_second_reply_with_no_answer_line_fails_as_the_first_would_have(tmp_path: Path) -> None:
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        writer = Scripted(None, FormatError("no ANSWER line", "Still an essay."))

        with pytest.raises(FormatError):
            reflect(writer, "q", matches, _papers(corpus, matches), None, "An essay.")


def test_answering_reflects_only_when_asked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "REFLECT", False)
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        first, _ = _ids(matches)
        draft = {"answer": "It scores 28.4 BLEU [1].", "citations": [_cite(1, first, PARAPHRASE)]}
        fixed = {"answer": "It scores 28.4 BLEU [1].", "citations": [_cite(1, first, SENTENCE)]}
        writer = Scripted(draft, fixed)

        plain = answer_from_matches(corpus, writer, "q", matches)
        assert writer.revised == [] and plain.rejected

        repaired = answer_from_matches(corpus, writer, "q", matches, reflection=True)
        assert len(writer.revised) == 1 and repaired.rejected == []


def test_the_setting_turns_reflection_on_wherever_answers_are_written(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "REFLECT", True)
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        first, _ = _ids(matches)
        draft = {"answer": "It scores 28.4 BLEU [1].", "citations": [_cite(1, first, PARAPHRASE)]}
        fixed = {"answer": "It scores 28.4 BLEU [1].", "citations": [_cite(1, first, SENTENCE)]}
        writer = Scripted(draft, fixed)

        answer = answer_from_matches(corpus, writer, "q", matches)

    assert len(writer.revised) == 1 and answer.rejected == []


def test_without_reflection_a_reply_with_no_answer_line_still_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "REFLECT", False)
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        writer = Scripted(FormatError("did not answer in the expected format", "An essay."))

        with pytest.raises(FormatError):
            answer_from_matches(corpus, writer, "q", matches)


def test_with_reflection_a_reply_with_no_answer_line_is_asked_for_again(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "REFLECT", True)
    with _corpus(tmp_path) as corpus:
        matches = _matches(corpus)
        first, _ = _ids(matches)
        fixed = {"answer": "It scores 28.4 BLEU [1].", "citations": [_cite(1, first, SENTENCE)]}
        writer = Scripted(FormatError("did not answer in the expected format", "An essay."), fixed)

        answer = answer_from_matches(corpus, writer, "q", matches)

    assert answer.citations and len(writer.revised) == 1
