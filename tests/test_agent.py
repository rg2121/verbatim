from pathlib import Path

from verbatim.agent import _summarise, answer_question, gather
from verbatim.tools import TOOL_SCHEMAS, get_citations, read_pages

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
        chunk_id = list(c.iter_chunks())[0].chunk_id
        caller = ScriptedCaller([("read_pages", {"arxiv_id": "2608.11111v1", "pages": [1]})])
        matches, trace = gather(c, None, caller, "what threads the ring?")

    assert [m.chunk.chunk_id for m in matches] == [chunk_id]
    assert trace.steps[0][0] == "read_pages"


def test_citations_are_still_verified_after_tool_use(tmp_path: Path) -> None:
    with corpus(tmp_path) as c:
        chunk_id = list(c.iter_chunks())[0].chunk_id
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


def test_a_bad_call_goes_back_to_the_model_with_what_was_wrong(tmp_path: Path) -> None:
    """Validation raises, and the loop feeds that back so the model can correct itself."""

    class Remembering(ScriptedCaller):
        def choose_tool(self, system, messages, tools):
            self.messages = list(messages)
            return super().choose_tool(system, messages, tools)

    with corpus(tmp_path) as c:
        caller = Remembering([("list_papers", {"since": "last week"}), ("list_papers", {"limit": 1})])
        gather(c, None, caller, "what is new?")

    assert "since must be" in str(caller.messages)


def test_the_local_model_is_told_only_what_it_was_told_before(tmp_path: Path) -> None:
    """Fields added for MCP clients stay out of a prompt that was measured without them."""
    with corpus(tmp_path) as c:
        read = read_pages(c, "2608.11111v1", [1])
        citers = get_citations(c, "2608.11111v1")

    passages = _summarise(read)
    assert passages.startswith("1 result(s): ")
    assert '"chunk_id"' in passages and '"arxiv_id"' in passages and '"page": 1' in passages
    for added in ("locator", "license_url", "next_skip", "position", "Magnetic flux"):
        assert added not in passages
    assert '"total"' not in _summarise(citers)
    assert '"papers"' in _summarise(citers)


def test_the_local_model_is_offered_only_the_tools_it_had() -> None:
    """A small model gets one more way to choose wrong with every tool."""
    assert {schema["name"] for schema in TOOL_SCHEMAS} == {
        "search_chunks", "get_paper", "list_papers", "get_citations", "read_pages",
    }
    read = next(schema for schema in TOOL_SCHEMAS if schema["name"] == "read_pages")
    assert set(read["parameters"]["properties"]) == {"arxiv_id", "pages"}
