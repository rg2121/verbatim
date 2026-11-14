"""M2c: one attempt's life. The Writer drafts with v1's prompt, code checks every
draft, the Verifier decides what the check found, and a repair is v1's repair.

Where v2 does what v1 did, the calls are compared with v1's own (I3: the Writer's
only input is its brief, rendered into v1's prompt).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from helpers import chunked_corpus
from verbatim.agents import attempts, contract
from verbatim.agents.attempts import REPAIR_MODE, check, decide, needs_verifier, repair_facts, verify, write
from verbatim.agents.contract import (
    DROP,
    FAILED,
    KEEP,
    KEEP_UNVERIFIED,
    PASSED,
    REPAIR,
    REMOVED,
    UNVERIFIED,
    VERIFIED,
    legal_returns,
    visible_status,
)
from verbatim.agents.protocol import ProtocolError, ask_twice, parse_verification
from verbatim.agents.runtime import require_checkpointable
from verbatim.agents.state import CitationDecision, CitationRecord, Draft, Verification, WriterBrief
from verbatim.answers import FormatError, Usage, answer_from_matches
from verbatim.corpus import ChunkMatch
from verbatim.reflection import findings, passage_texts, reflect

PASSAGE = ("Transformers replace recurrence with attention. "
           "The model reaches 28.4 BLEU on the English-to-German translation task. "
           "Training took three and a half days on eight GPUs.")
OTHER = "Dropout of 0.1 was applied to the output of each sub-layer."
QUESTION = "How well does the model translate?"
GOOD = "The model reaches 28.4 BLEU on the English-to-German translation task"
CLAIM = "Larger ensembles would generalise better across unseen languages."


class Recording:
    """A model client replying from a script, recording each call as v1's clients take it."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str, str, dict | None]] = []
        self.usage = Usage()

    def complete(self, system: str, user: str, schema: dict) -> dict:
        return self._reply("complete", system, user, schema)

    def revise(self, system: str, user: str, schema: dict) -> dict:
        return self._reply("revise", system, user, schema)

    def complete_text(self, system: str, user: str) -> str:
        return self._reply("text", system, user, None)

    def _reply(self, kind: str, system: str, user: str, schema: dict | None) -> Any:
        self.calls.append((kind, system, user, schema))
        self.usage.record({})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def passages(tmp_path: Path):
    corpus = chunked_corpus(tmp_path, {"2608.11111v1": [PASSAGE], "2608.22222v1": [OTHER]})
    with corpus:
        matches = [ChunkMatch(chunk=c, score=1.0, snippet=c.text) for c in corpus.iter_chunks()]
        papers = corpus.papers_by_id(sorted({m.chunk.arxiv_id for m in matches}))
        yield corpus, matches, papers


def ids(matches) -> tuple[str, str]:
    by_paper = {m.chunk.arxiv_id: m.chunk.chunk_id for m in matches}
    return by_paper["2608.11111v1"], by_paper["2608.22222v1"]


def payload(answer: str, *citations: tuple[int, str, str]) -> dict:
    return {"answer": answer, "citations": [
        {"marker": marker, "chunk_id": chunk_id, "quoted_text": quote}
        for marker, chunk_id, quote in citations]}


def brief(matches, **fields: Any) -> WriterBrief:
    return WriterBrief(task=QUESTION, passage_ids=tuple(m.chunk.chunk_id for m in matches), **fields)


# -- the Writer (I3) --------------------------------------------------------------------


def test_the_writer_sends_exactly_what_v1s_plain_ask_sends(passages) -> None:
    corpus, matches, papers = passages
    first, _ = ids(matches)
    reply = payload(f"{GOOD} [1].", (1, first, GOOD))
    v1, v2 = Recording(reply), Recording(reply)
    answer_from_matches(corpus, v1, QUESTION, matches, reflection=False)
    assert write(v2, brief(matches), matches) == reply
    assert v2.calls == v1.calls


def test_a_reply_without_an_answer_line_goes_back_once_as_v1_sends_it(passages) -> None:
    corpus, matches, papers = passages
    first, _ = ids(matches)
    good = payload(f"{GOOD} [1].", (1, first, GOOD))
    rambling = "I think the model does well, judging by the passages given. " * 20  # > 600 chars
    v1 = Recording(good)
    reflect(v1, QUESTION, matches, papers, None, rambling)
    v2 = Recording(FormatError("no ANSWER line", rambling), good)
    assert write(v2, brief(matches), matches) == good
    assert v2.calls[1:] == v1.calls  # the retry is v1's, word for word
    # A second reply without one ends the step: the Writer did not follow the format.
    unreadable = Recording(FormatError("no", "x"), FormatError("no", "y"))
    assert write(unreadable, brief(matches), matches) is None


def test_a_repair_is_v1s_repair(passages) -> None:
    corpus, matches, papers = passages
    first, second = ids(matches)
    draft = payload(f"{GOOD} [1]. Dropout was applied to every sub-layer output [2]. {CLAIM}",
                    (1, first, GOOD), (2, second, "Dropout of 0.2 was applied"))
    revision = payload(f"{GOOD} [1].", (1, first, GOOD))
    v1 = Recording(revision)
    reflect(v1, QUESTION, matches, papers, draft)
    checked = check(1, brief(matches), draft, matches, papers)
    v2 = Recording(revision)
    repair = brief(matches, mode=REPAIR_MODE, repair_facts=repair_facts(checked.draft, checked.records),
                   draft=1)
    assert write(v2, repair, matches, checked.draft) == revision
    assert v2.calls == v1.calls
    # As in v1, an unreadable repair costs its one call, not the draft: it is not sent back.
    unreadable = Recording(FormatError("no ANSWER line", "x"), revision)
    assert write(unreadable, repair, matches, checked.draft) is None and len(unreadable.calls) == 1


# -- the check after every draft --------------------------------------------------------


def test_every_cite_line_is_checked_and_the_attempt_tallied(passages) -> None:
    corpus, matches, papers = passages
    first, second = ids(matches)
    written = payload(f"{GOOD} [1]. Dropout was applied to every sub-layer output [2]. {CLAIM}",
                      (1, first, GOOD), (2, second, "Dropout of 0.2 was applied"))
    checked = check(3, brief(matches, mode=REPAIR_MODE, draft=2), written, matches, papers)
    assert [(r.marker, r.check) for r in checked.records] == [(1, PASSED), (2, FAILED)]
    assert checked.records[1].reason == "quoted text does not appear in the cited passage"
    assert checked.draft.attempt == contract.Attempt(3, checked.draft.tally, revises=2)
    assert checked.draft.tally.verified == 1 and checked.draft.tally.uncited == 1
    # The repair facts the checker keeps are v1's, in v1's order.
    assert checked.draft.findings == tuple(findings(written, checked.draft.answer, matches))


def test_the_verifier_runs_where_v1_would_have_repaired(passages) -> None:
    corpus, matches, papers = passages
    first, _ = ids(matches)

    def needs(answer: str, *citations) -> bool:
        return needs_verifier(check(1, brief(matches), payload(answer, *citations), matches, papers).draft)

    assert not needs(f"{GOOD} [1].", (1, first, GOOD))  # nothing the check flags
    assert needs(f"{GOOD} [1]. {CLAIM}", (1, first, GOOD))  # an uncited claim
    assert needs(f"{GOOD} [1] [2].", (1, first, GOOD))  # a marker with no CITE line
    # An answer that cites nothing is how v1's Writer declines, and v1 never repairs one.
    assert not needs("The passages do not say how well the model translates.")


# -- the Verifier (I1) ------------------------------------------------------------------


def test_a_verifier_reply_is_read_into_decisions() -> None:
    reply = "ACTION: [1] keep\nACTION: 2: drop\n- ACTION: [3] keep unverified\nNEXT: writer\nRESEARCH: who measured it"
    assert parse_verification(reply) == Verification(
        decisions=(CitationDecision(1, KEEP), CitationDecision(2, DROP),
                   CitationDecision(3, KEEP_UNVERIFIED)), repair=True, research="who measured it")
    assert parse_verification("NEXT: done") == Verification()
    for bad, says in (("ACTION: fix it\nNEXT: done", "one decision"),
                      ("ACTION: [1] keep\nACTION: [1] drop\nNEXT: done", "two decisions"),
                      ("ACTION: [1] keep", "exactly one"), ("NEXT: maybe", "writer or done")):
        with pytest.raises(ProtocolError, match=says):
            parse_verification(bad)


def test_the_verifiers_decisions_must_be_legal_for_each_check() -> None:
    checks = {1: (PASSED,), 2: (FAILED,), 3: (PASSED, FAILED)}
    refused = contract.verification_refused
    assert refused(Verification((CitationDecision(2, REPAIR),), repair=True), checks) is None
    assert "never shown as verified" in refused(Verification((CitationDecision(2, KEEP),)), checks)
    assert "no citation [4]" in refused(Verification((CitationDecision(4, DROP),)), checks)
    # Two CITE lines under one marker, one passed and one failed: keep suits only one.
    assert "never shown as verified" in refused(Verification((CitationDecision(3, KEEP),)), checks)
    assert refused(Verification((CitationDecision(3, REPAIR),), repair=True), checks) is None
    assert "NEXT: writer" in refused(Verification((CitationDecision(2, REPAIR),)), checks)


def test_a_citation_is_shown_as_its_check_and_the_verifiers_decision_allow(passages) -> None:
    corpus, matches, papers = passages
    first, second = ids(matches)
    written = payload(f"{GOOD} [1]. Dropout was applied to every sub-layer output [2].",
                      (1, first, GOOD), (2, second, "Dropout of 0.2 was applied"))
    records = check(1, brief(matches), written, matches, papers).records
    assert [visible_status(r.check, r.decision) for r in records] == [VERIFIED, UNVERIFIED]
    decided = decide(records, Verification((CitationDecision(1, DROP), CitationDecision(2, KEEP_UNVERIFIED))))
    assert [visible_status(r.check, r.decision) for r in decided] == [REMOVED, UNVERIFIED]


def test_repair_facts_leave_out_what_the_verifier_decided_otherwise(passages) -> None:
    corpus, matches, papers = passages
    first, second = ids(matches)
    written = payload(f"Attention replaces recurrence [1]. Dropout was applied to every sub-layer [2]. {CLAIM}",
                      (1, first, "Attention replaced recurrence"), (2, second, "Dropout of 0.2 was applied"))
    checked = check(1, brief(matches), written, matches, papers)
    every = repair_facts(checked.draft, checked.records)
    assert every == checked.draft.findings  # asked to repair, the Writer gets all of v1's facts
    kept = repair_facts(checked.draft, decide(checked.records, Verification(
        (CitationDecision(1, KEEP_UNVERIFIED),), repair=True)))
    assert [fact for fact in every if fact not in kept] == [every[0]]  # [1]'s fact left out
    # A passed citation sent back gets a fixed line: no Verifier words reach the Writer.
    clean = check(2, brief(matches), payload(f"{GOOD} [1].", (1, first, GOOD)), matches, papers)
    facts = repair_facts(clean.draft, decide(clean.records, Verification((CitationDecision(1, REPAIR),), repair=True)))
    assert facts == (attempts.PASSED_BUT_REPAIR.format(marker=1),)


def test_the_verifier_is_asked_once_more_and_then_gives_up(passages) -> None:
    corpus, matches, papers = passages
    first, second = ids(matches)
    written = payload(f"{GOOD} [1]. Dropout was applied to every sub-layer [2].",
                      (1, first, GOOD), (2, second, "Dropout of 0.2 was applied"))
    checked = check(1, brief(matches), written, matches, papers)
    texts = passage_texts(matches)
    # Illegal first (keeping a failed citation as verified), then legal.
    model = Recording("ACTION: [2] keep\nNEXT: done", "ACTION: [2] repair\nNEXT: writer")
    decision, replies = verify(model.complete_text, checked.draft, checked.records, texts)
    assert decision == Verification((CitationDecision(2, REPAIR),), repair=True) and len(replies) == 2
    assert "never shown as verified" in model.calls[1][2]  # the reason went back with the reply
    twice = Recording("nothing useful", "still nothing")
    assert verify(twice.complete_text, checked.draft, checked.records, texts)[0] is None
    # NEXT: writer with nothing left to repair is refused too.
    clean = check(2, brief(matches), payload(f"{GOOD} [1].", (1, first, GOOD)), matches, papers)
    idle = Recording("NEXT: writer", "NEXT: done")
    assert verify(idle.complete_text, clean.draft, clean.records, texts)[0] == Verification()


def test_the_verifier_reads_the_checkers_facts_with_paper_text_marked_as_data(passages) -> None:
    corpus, matches, papers = passages
    first, second = ids(matches)
    written = payload(f"{GOOD} [1]. Dropout was applied to every sub-layer [2]. {CLAIM}",
                      (1, first, GOOD), (2, second, "Dropout of 0.1 was used"))
    checked = check(1, brief(matches), written, matches, papers)
    view = attempts.verifier_view(checked.draft, checked.records, passage_texts(matches))
    assert "[1] passed" in view and "[2] failed: quoted text does not appear" in view
    assert CLAIM in view  # the uncited claim
    assert "<passage>Dropout of 0.1 was applied to the output of each sub-layer.</passage>" in view


def test_a_repair_that_loses_a_verified_passage_cannot_be_returned(passages) -> None:
    # I10 end to end: written, checked, repaired, checked again, then judged.
    corpus, matches, papers = passages
    first, second = ids(matches)
    draft = check(1, brief(matches), payload(f"{GOOD} [1]. Dropout was applied to every sub-layer [2].",
                                            (1, first, GOOD), (2, second, "Dropout of 0.2 was applied")),
                  matches, papers).draft
    worse = check(2, brief(matches, mode=REPAIR_MODE, draft=1),
                  payload("Dropout of 0.1 was applied to the output of each sub-layer [1].",
                          (1, second, "Dropout of 0.1 was applied to the output of each sub-layer")),
                  matches, papers).draft
    assert legal_returns([draft.attempt, worse.attempt]) == (1,)


def test_every_record_an_attempt_keeps_comes_back_from_a_checkpoint_as_itself(passages) -> None:
    from verbatim.agents.runtime import serializer

    require_checkpointable(Draft, CitationRecord, Verification, WriterBrief)
    corpus, matches, papers = passages
    first, second = ids(matches)
    checked = check(1, brief(matches), payload(f"{GOOD} [1]. Dropout was applied [2].",
                                              (1, first, GOOD), (2, second, "Dropout of 0.2 was applied")),
                    matches, papers)
    for record in (checked.draft, *checked.records):
        assert serializer().loads_typed(serializer().dumps_typed(record)) == record


def test_ask_twice_returns_none_after_two_replies_it_cannot_use() -> None:
    def read(reply: str) -> int:
        if not reply.isdigit():
            raise ProtocolError("reply with a number")
        return int(reply)

    model = Recording("seven", "7")
    assert ask_twice(model.complete_text, "system", "user", read) == (7, ("seven", "7"))
    assert "reply with a number" in model.calls[1][2] and "seven" in model.calls[1][2]
    assert ask_twice(Recording("a", "b").complete_text, "s", "u", read) == (None, ("a", "b"))


# -- what the review of M2c found -------------------------------------------------------


def test_the_verifier_reads_every_fact_in_the_drafts_own_numbering(passages) -> None:
    corpus, matches, papers = passages
    first, second = ids(matches)
    # [1] fails, so the check renumbers [2] as [1] and [3] as [2] in the text it shows.
    written = payload(
        "Dropout was applied to every sub-layer output [1]. "
        "Convolutions outperform attention on every benchmark studied [2]. "
        'Each sub-layer output had "Dropout of 0.1 was applied" to it [2]. '
        f"{GOOD} [2]. The output of each sub-layer had dropout applied [3].",
        (1, first, "Dropout of 0.2 was applied"), (2, first, GOOD),
        (3, second, "Dropout of 0.1 was applied to the output of each sub-layer"))
    checked = check(1, brief(matches), written, matches, papers)
    view = attempts.verifier_view(checked.draft, checked.records, passage_texts(matches))
    assert '"Convolutions outperform attention on every benchmark studied" shares few words with the passage cited as [2]' in view
    assert 'The quotation "Dropout of 0.1 was applied" is quoted from [3], not [2]' in view


@pytest.mark.parametrize(("line", "repair"), [
    ("NEXT: done (nothing to repair)", False), ("NEXT: done, no need to send it back to the writer", False),
    ("NEXT: done. The writer cited it correctly.", False), ("NEXT: writer, then done", True),
    ("NEXT: repair", True),
])
def test_the_first_of_writer_or_done_decides(line: str, repair: bool) -> None:
    assert parse_verification(line).repair is repair


@pytest.mark.parametrize(("reply", "read"), [
    ("ACTION [2] drop", [(2, DROP)]),
    ("Action: [2] drop", [(2, DROP)]),
    ("ACTIONS: [2] drop", [(2, DROP)]),
    ("ACTION: [1] keep, [2] drop", [(1, KEEP), (2, DROP)]),
    ("ACTION: [2] keep as unverified", [(2, KEEP_UNVERIFIED)]),
    ("ACTION: [2] keep (unverified)", [(2, KEEP_UNVERIFIED)]),
    ("ACTION: [2] keep_unverified", [(2, KEEP_UNVERIFIED)]),
    ("ACTION: [2] **repair**", [(2, REPAIR)]),
    ("ACTION: 2) drop", [(2, DROP)]),
    ("ACTION: CITE [2] drop", [(2, DROP)]),
    ("ACTION: [2] -> drop", [(2, DROP)]),
    ("ACTION: [2] drop, since its quote is from page 3", [(2, DROP)]),
    ("ACTION: none", []),
])
def test_action_lines_are_read_however_they_are_written(reply: str, read: list) -> None:
    decided = parse_verification(f"{reply}\nNEXT: writer").decisions
    assert [(d.marker, d.decision) for d in decided] == read


@pytest.mark.parametrize("reply", ["ACTION: [2] keep | repair | drop | keep_unverified",
                                   "ACTION: [2]", "ACTION: drop", "ACTION: [2] remove"])
def test_an_action_line_that_cannot_be_read_whole_is_an_error(reply: str) -> None:
    with pytest.raises(ProtocolError, match="one decision"):
        parse_verification(f"{reply}\nNEXT: done")


def test_the_brief_is_what_the_writer_and_the_check_use(passages) -> None:
    corpus, matches, papers = passages
    first, _ = ids(matches)
    reply = payload(f"{GOOD} [1].", (1, first, GOOD))
    one = WriterBrief(task=QUESTION, passage_ids=(first,))
    with pytest.raises(ValueError, match="passages"):
        write(Recording(reply), one, matches)
    with pytest.raises(ValueError, match="passages"):
        check(1, one, reply, matches, papers)
    draft = check(2, brief(matches), reply, matches, papers).draft
    with pytest.raises(ValueError, match="attempt 1"):
        write(Recording(reply), brief(matches, mode=REPAIR_MODE, draft=1), matches, draft)
    # A repair's draft comes from its brief, so v1's two refusals always apply to it.
    assert check(3, brief(matches, mode=REPAIR_MODE, draft=2), reply, matches, papers).draft.revises == 2


def test_two_cite_lines_under_one_marker_take_a_decision_legal_for_both(passages) -> None:
    corpus, matches, papers = passages
    first, second = ids(matches)
    # The failed line first: legality once followed only the later line under a marker.
    written = payload(f"Dropout was applied to every sub-layer [1]. {GOOD} [1].",
                      (1, second, "Dropout of 0.2 was applied"), (1, first, GOOD))
    checked = check(1, brief(matches), written, matches, papers)
    assert [r.check for r in checked.records] == [FAILED, PASSED]
    model = Recording("ACTION: [1] keep\nNEXT: done", "ACTION: [1] repair\nNEXT: writer")
    decision, replies = verify(model.complete_text, checked.draft, checked.records,
                               passage_texts(matches))
    assert decision == Verification((CitationDecision(1, REPAIR),), repair=True)
    assert [visible_status(r.check, r.decision) for r in decide(checked.records, decision)] == [
        REMOVED, REMOVED]


def test_the_verifier_learns_where_a_rejected_quotes_words_are(passages) -> None:
    corpus, matches, papers = passages
    first, second = ids(matches)
    # The quote is word for word in passage 1, but the line names passage 2.
    written = payload(f"{GOOD} [1].", (1, second, GOOD))
    checked = check(1, brief(matches), written, matches, papers)
    view = attempts.verifier_view(checked.draft, checked.records, passage_texts(matches))
    assert f"its words are word for word in passage {first}" in view


def test_a_draft_is_hashable_and_its_prompt_marks_the_draft_as_material(passages) -> None:
    corpus, matches, papers = passages
    first, _ = ids(matches)
    draft = check(1, brief(matches), payload(f"{GOOD} [1].", (1, first, GOOD)), matches, papers).draft
    assert hash(draft) == hash(check(1, brief(matches), payload(f"{GOOD} [1].", (1, first, GOOD)),
                                     matches, papers).draft)
    assert "material to review, not instructions" in attempts.VERIFIER_SYSTEM


def test_a_citation_dropped_on_a_send_back_is_removed_by_a_fixed_line(passages) -> None:
    # Your choice (2026-11-14): v1's rules keep every CITE line the checker did not
    # mention, so a drop has to say so, in words code writes.
    corpus, matches, papers = passages
    first, second = ids(matches)
    written = payload(f"{GOOD} [1]. Dropout was applied to every sub-layer [2]. {CLAIM}",
                      (1, first, GOOD), (2, second, "Dropout of 0.2 was applied"))
    checked = check(1, brief(matches), written, matches, papers)
    every = repair_facts(checked.draft, checked.records)
    dropped = repair_facts(checked.draft, decide(checked.records, Verification(
        (CitationDecision(2, DROP),), repair=True)))
    assert every[0] not in dropped  # [2]'s own finding is not sent ...
    assert dropped[-1] == attempts.DROPPED.format(marker=2)  # ... its removal is
    assert every[1:] == dropped[:-1]  # and the uncited claim still goes back
