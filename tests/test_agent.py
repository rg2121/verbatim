from pathlib import Path

from verbatim.agent import answer_question, gather

from helpers import chunked_corpus, paper


class ScriptedCaller:
    """Chooses tools from a fixed script, so routing is testable offline."""

    def __init__(self, script):
        self.script = list(script)
        self.seen: list[str] = []

    def choose_tool(self, system, messages, tools):
        self.seen.append(system)
        return self.script.pop(0) if self.script else None


class ScriptedCompletion:
    def __init__(self, payload):
        self.payload = payload
        self.called = False

    def complete(self, system, user, schema):
        self.called = True
        return self.payload


def corpus(tmp_path):
    c = chunked_corpus(tmp_path, {"2608.11111v1": ["## Intro\n\nMagnetic flux threads the ring."]})
    c.upsert_papers([paper("2608.22222v1", updated="2026-08-09T10:00:00Z")])
    return c


def test_a_recency_question_is_answered_from_records_not_prose(tmp_path: Path) -> None:
    """No passages means nothing to quote, so no model narration is invited."""
    with corpus(tmp_path) as c:
        caller = ScriptedCaller([("list_papers", {"limit": 5})])
        completion = ScriptedCompletion({"answer": "should not be used", "citations": []})
        answer, trace = answer_question(c, None, caller, completion, "what is new?")

    assert completion.called is False
    assert "2608.22222v1" in answer.text
    assert answer.citations == []
    assert trace.describe().startswith("list_papers")


def test_gathered_passages_become_citable_evidence(tmp_path: Path) -> None:
    with corpus(tmp_path) as c:
        chunk_id = c.all_chunks()[0].chunk_id
        caller = ScriptedCaller([("read_pages", {"arxiv_id": "2608.11111v1", "pages": [1]})])
        matches, trace = gather(c, None, caller, "what threads the ring?")

    assert [m.chunk.chunk_id for m in matches] == [chunk_id]
    assert trace.steps[0][0] == "read_pages"


def test_citations_are_still_verified_after_tool_use(tmp_path: Path) -> None:
    with corpus(tmp_path) as c:
        chunk_id = c.all_chunks()[0].chunk_id
        caller = ScriptedCaller([("read_pages", {"arxiv_id": "2608.11111v1", "pages": [1]})])
        completion = ScriptedCompletion({
            "answer": "It is threaded by something invented [1].",
            "citations": [{"marker": 1, "chunk_id": chunk_id, "quoted_text": "invented wording"}],
        })
        answer, _ = answer_question(c, None, caller, completion, "?")

    assert answer.citations == []            # fabricated quote rejected
    assert len(answer.rejected) == 1
    assert "[unverified]" in answer.text


def test_a_malformed_tool_call_does_not_crash_the_loop(tmp_path: Path) -> None:
    with corpus(tmp_path) as c:
        caller = ScriptedCaller([("no_such_tool", {}), ("list_papers", {"limit": 2})])
        matches, trace = gather(c, None, caller, "anything")

    assert [name for name, _ in trace.steps] == ["no_such_tool", "list_papers"]
    assert matches == []


def test_the_loop_stops_when_the_model_stops_choosing(tmp_path: Path) -> None:
    with corpus(tmp_path) as c:
        caller = ScriptedCaller([])           # chooses nothing at all
        matches, trace = gather(c, None, caller, "anything")

    assert trace.steps == []
    assert trace.describe() == "no tools used"


def test_an_empty_metadata_result_is_not_reported_as_no_passages(tmp_path: Path) -> None:
    """A recency window with nothing in it searched no passages at all."""
    with corpus(tmp_path) as c:
        caller = ScriptedCaller([("list_papers", {"since": "2099-01-01T00:00:00Z"})])
        completion = ScriptedCompletion({"answer": "unused", "citations": []})
        answer, trace = answer_question(c, None, caller, completion, "what is new today?")

    assert completion.called is False
    assert "no records matching" in answer.text
    assert "list_papers" in answer.text
    assert "passages" not in answer.text
