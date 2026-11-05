"""The MCP front end onto the corpus, driven as a client drives it."""

from contextlib import closing
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import threading

from mcp import Client, StdioServerParameters
import pytest

from verbatim import config
from verbatim.answers import Usage
from verbatim.corpus import ChunkMatch, CorpusStore, EmbeddingModel
from verbatim.mcp_server import CLIENT_TOOL_SCHEMAS, build_server, render_result, serve
from verbatim.runs import recent_runs, record_run
import verbatim.tools as tools
from verbatim.tools import TOOL_SCHEMAS

from helpers import chunked_corpus

MODEL = "BAAI/bge-small-en-v1.5"
SAMPLE = Path(__file__).parents[1] / "sample" / "corpus.db"
VERBATIM = Path(sys.executable).parent / "verbatim"
SENTENCE = "Magnetic flux threads the ring."
AGENT_TOOLS = {schema["name"] for schema in TOOL_SCHEMAS}
CLIENT_ONLY = {"verify_citations", "export_citations", "list_runs"}
JOB_TOOLS = {"ask", "report", "get_job", "cancel_job"}
QUESTION = "What threads the ring?"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def state(tmp_path: Path) -> Path:
    """A corpus written and closed, which the server opens for itself."""
    with chunked_corpus(tmp_path, {"2608.11111v1": [f"## Intro\n\n{SENTENCE}"]}):
        pass
    return tmp_path / "corpus.db"


def _chunk_id(state: Path) -> str:
    with closing(sqlite3.connect(state)) as database:
        return database.execute("SELECT chunk_id FROM chunks LIMIT 1").fetchone()[0]


def _text(result) -> str:
    return " ".join(block.text for block in result.content)


async def _tools(client) -> dict:
    return {tool.name: tool for tool in (await client.list_tools()).tools}


def test_a_result_keeps_the_fields_a_citation_needs() -> None:
    """A client that cites what it read needs provenance, not a summary."""
    rendered = render_result(
        [{"arxiv_id": "2608.05440v1", "page": 16, "chunk_id": "abc", "text": "words"}]
    )

    assert '"arxiv_id": "2608.05440v1"' in rendered
    assert '"page": 16' in rendered
    assert '"chunk_id": "abc"' in rendered


def test_an_empty_result_says_so_rather_than_rendering_null() -> None:
    assert render_result(None) == "No result."


@pytest.mark.anyio
async def test_the_server_offers_the_agents_tools_and_the_checks_a_client_needs(state: Path) -> None:
    """A client's model writes its own answers, so it alone needs the check as a tool."""
    async with Client(build_server(state, MODEL)) as client:
        listed = await _tools(client)

    assert set(listed) == AGENT_TOOLS | CLIENT_ONLY


@pytest.mark.anyio
async def test_a_client_can_list_and_call_tools_over_a_session(state: Path) -> None:
    async with Client(build_server(state, MODEL)) as client:
        result = await client.call_tool("list_papers", {"limit": 5})

    assert result.is_error is not True
    assert "2608.11111v1" in _text(result)


@pytest.mark.anyio
async def test_the_server_works_over_real_stdio_on_a_copy_of_the_sample(tmp_path: Path) -> None:
    """The process, the transport and a clean stdout, as a client launches it. No model is loaded."""
    copy = tmp_path / "corpus.db"
    shutil.copy(SAMPLE, copy)
    with closing(sqlite3.connect(copy)) as database:
        chunk_id, text = database.execute(
            "SELECT chunk_id, text FROM chunks WHERE source_kind = 'html' LIMIT 1"
        ).fetchone()
    quote = " ".join(text.split()[:8])
    params = StdioServerParameters(
        command=str(VERBATIM),
        args=["mcp", "--state", str(copy)],
        cwd=str(tmp_path),
        env={"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", str(tmp_path))},
    )

    async with Client(params) as client:
        listed = await _tools(client)
        papers = await client.call_tool("list_papers", {"limit": 2})
        checked = await client.call_tool(
            "verify_citations", {"citations": [{"chunk_id": chunk_id, "quote": quote}]}
        )
        refused = await client.call_tool("list_papers", {"since": "not-a-date"})

    assert set(listed) == AGENT_TOOLS | CLIENT_ONLY
    assert len(json.loads(_text(papers))) == 2
    assert checked.structured_content["all_verified"] is True
    assert refused.is_error is True and "since must be" in _text(refused)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "arguments", "complaint"),
    [
        ("list_papers", {"since": "not-a-date"}, "since must be"),
        ("list_papers", {"days": -5}, "days must be"),
        ("get_paper", {"identifier": ""}, "identifier is empty"),
        ("read_pages", {"arxiv_id": "9999.99999v1", "pages": [1]}, "no paper"),
        ("search_chunks", {"query": "   "}, "query is empty"),
        ("verify_citations", {"citations": []}, "citations is empty"),
        ("export_citations", {"chunk_ids": ["feedface"]}, "no passages with these chunk_ids"),
        ("list_runs", {"limit": 0}, "limit must be"),
    ],
)
async def test_a_bad_argument_reaches_the_client_as_a_readable_error(
    state: Path, tool: str, arguments: dict, complaint: str
) -> None:
    """The SDK erases any other exception, and then a model cannot correct itself."""
    async with Client(build_server(state, MODEL)) as client:
        result = await client.call_tool(tool, arguments)

    assert result.is_error is True
    assert complaint in _text(result)


@pytest.mark.anyio
async def test_a_crash_keeps_its_details_off_the_wire(state: Path, monkeypatch) -> None:
    def crash(*args, **kwargs):
        raise RuntimeError("/Users/someone/secret/path")

    monkeypatch.setattr("verbatim.mcp_server.call_tool", crash)

    async with Client(build_server(state, MODEL)) as client:
        result = await client.call_tool("list_papers", {})

    assert result.is_error is True
    assert "secret" not in _text(result)


@pytest.mark.anyio
async def test_the_server_says_what_it_holds_and_how_to_use_it(state: Path) -> None:
    async with Client(build_server(state, MODEL)) as client:
        instructions = client.instructions
        info = client.server_info

    assert "1 paper" in instructions and "2026-08-08" in instructions
    assert "not live" in instructions
    assert "verify_citations" in instructions
    assert "never instructions" in instructions
    # Claude Code loads these at session start and cuts what runs long.
    assert len(instructions) < 2_000
    assert info.name == "verbatim"
    assert info.version == version("verbatim")


@pytest.mark.anyio
async def test_every_tool_is_titled_and_marked_as_reading_a_closed_corpus(state: Path) -> None:
    async with Client(build_server(state, MODEL)) as client:
        listed = await _tools(client)

    for tool in listed.values():
        assert tool.title
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        assert tool.annotations.idempotent_hint is True
        assert tool.annotations.open_world_hint is False


@pytest.mark.anyio
async def test_a_tool_declares_the_arguments_it_actually_takes(state: Path) -> None:
    """The server derives the schema from the signature, so it must be real."""
    async with Client(build_server(state, MODEL)) as client:
        listed = await _tools(client)

    search = listed["search_chunks"].input_schema
    assert "query" in search["properties"]
    assert search["required"] == ["query"]
    assert "arxiv_id" in search["properties"]
    for schema in TOOL_SCHEMAS:
        published = listed[schema["name"]].input_schema
        assert set(schema["parameters"]["properties"]) <= set(published["properties"])
        assert set(schema["parameters"].get("required", ())) == set(published.get("required", ()))


@pytest.mark.anyio
async def test_an_array_parameter_is_published_as_a_list_not_a_string(state: Path) -> None:
    async with Client(build_server(state, MODEL)) as client:
        listed = await _tools(client)

    pages = listed["read_pages"].input_schema["properties"]["pages"]
    assert pages["type"] == "array"
    assert pages["items"]["type"] == "integer"


@pytest.mark.anyio
async def test_a_parameter_keeps_its_description_and_its_choices(state: Path) -> None:
    """They were dropped on the way to the signature, leaving a model to guess."""
    async with Client(build_server(state, MODEL)) as client:
        listed = await _tools(client)

    direction = listed["get_citations"].input_schema["properties"]["direction"]
    assert "cited_by" in json.dumps(direction) and "references" in json.dumps(direction)
    assert direction["description"]
    assert listed["search_chunks"].input_schema["properties"]["limit"]["description"]


def test_a_client_is_told_more_than_the_local_agent_without_changing_the_tools() -> None:
    by_name = {schema["name"]: schema for schema in CLIENT_TOOL_SCHEMAS}

    assert set(by_name) == AGENT_TOOLS
    assert "skip" in by_name["read_pages"]["parameters"]["properties"]
    assert all(schema["title"] for schema in CLIENT_TOOL_SCHEMAS)
    assert all("title" not in schema for schema in TOOL_SCHEMAS)


@pytest.mark.anyio
async def test_paper_text_reaches_the_client_marked_as_data(state: Path, monkeypatch) -> None:
    """A paper can say anything, and one in the corpus says to forward every email."""
    with CorpusStore(state) as corpus:
        chunk = list(corpus.iter_chunks())[0]
    monkeypatch.setattr(
        tools, "hybrid_search",
        lambda *a, **o: [ChunkMatch(chunk=chunk, score=1.0, snippet="")],
    )

    async with Client(build_server(state, MODEL)) as client:
        searched = json.loads(_text(await client.call_tool("search_chunks", {"query": "flux"})))
        read = json.loads(
            _text(await client.call_tool("read_pages", {"arxiv_id": "2608.11111v1", "pages": [1]}))
        )
        paper = json.loads(_text(await client.call_tool("get_paper", {"identifier": "2608.11111"})))

    assert searched["passages"][0]["chunk_id"] == chunk.chunk_id
    assert searched["passages"][0]["locator"] == "p. 1"
    assert read["passages"][0]["text"] == chunk.text
    for result in (searched, read, paper):
        assert "never instructions" in result["note"]


@pytest.mark.anyio
async def test_a_paper_that_is_not_held_says_no_result(state: Path) -> None:
    async with Client(build_server(state, MODEL)) as client:
        result = await client.call_tool("get_paper", {"identifier": "no such paper anywhere"})

    assert result.is_error is not True
    assert _text(result) == "No result."


@pytest.mark.anyio
async def test_quotes_are_checked_over_the_protocol_with_a_published_shape(state: Path) -> None:
    chunk_id = _chunk_id(state)

    async with Client(build_server(state, MODEL)) as client:
        listed = await _tools(client)
        result = await client.call_tool(
            "verify_citations",
            {
                "citations": [
                    {"chunk_id": chunk_id, "quote": SENTENCE},
                    {"chunk_id": chunk_id, "quote": "Magnetic flux threads the loop"},
                ],
                "answer": "Magnetic flux threads the ring here [1]. It threads a loop too [2].",
            },
        )

    report = result.structured_content
    assert listed["verify_citations"].output_schema is not None
    properties = listed["verify_citations"].input_schema["properties"]
    assert "chunk_id" in json.dumps(listed["verify_citations"].input_schema)
    assert "answer" in properties
    assert result.is_error is not True
    assert (report["verified"], report["rejected"]) == (1, 1)
    assert report["verdicts"][1]["did_you_mean"] == SENTENCE
    assert "[unverified]" in report["checked_answer"]
    assert report["bibliography"].startswith("[1] ")


@pytest.mark.anyio
async def test_references_are_exported_over_the_protocol(state: Path) -> None:
    chunk_id = _chunk_id(state)

    async with Client(build_server(state, MODEL)) as client:
        listed = await _tools(client)
        result = await client.call_tool(
            "export_citations", {"chunk_ids": [chunk_id], "format": "bibtex"}
        )

    assert "bibtex" in json.dumps(listed["export_citations"].input_schema)
    assert result.structured_content["entries"] == 1
    assert result.structured_content["references"].startswith("@misc{")


@pytest.mark.anyio
async def test_calls_are_logged_and_the_corpus_stays_as_it_was(state: Path) -> None:
    """A read leaves the corpus alone, and a model's bad argument is not a failed run."""
    before = state.read_bytes()
    chunk_id = _chunk_id(state)

    async with Client(build_server(state, MODEL)) as client:
        await client.call_tool("list_papers", {"limit": 1})
        await client.call_tool("list_papers", {"days": -5})
        await client.call_tool(
            "verify_citations",
            {"citations": [{"chunk_id": chunk_id, "quote": "a very private sentence " * 40}]},
        )
        history = await client.call_tool("list_runs", {})

    assert state.read_bytes() == before
    calls = history.structured_content["runs"]
    assert [call["command"] for call in calls] == ["verify_citations", "list_papers", "list_papers"]
    verified, refused, listed = calls
    assert listed["ok"] is True and listed["outputs"] == {"results": 1}
    assert refused["ok"] is False and "days must be" in refused["error"]
    assert verified["inputs"]["citations"] == 1
    assert "private" not in json.dumps(verified)
    with CorpusStore(state) as corpus:
        assert recent_runs(corpus) == []


@pytest.mark.anyio
async def test_a_query_is_logged_only_as_far_as_is_useful(state: Path) -> None:
    async with Client(build_server(state, MODEL)) as client:
        await client.call_tool("get_paper", {"identifier": "x" * 500})
        history = await client.call_tool("list_runs", {})

    assert len(history.structured_content["runs"][0]["inputs"]["identifier"]) <= 200


@pytest.mark.anyio
async def test_list_runs_reads_the_commands_too(state: Path) -> None:
    with CorpusStore(state) as corpus:
        with record_run(corpus, "sync", limit=5) as run:
            run.record(papers=5)

    async with Client(build_server(state, MODEL)) as client:
        history = await client.call_tool("list_runs", {"source": "commands"})
        filtered = await client.call_tool("list_runs", {"source": "commands", "name": "ask"})

    (sync,) = history.structured_content["runs"]
    assert sync["command"] == "sync" and sync["outputs"] == {"papers": 5}
    assert "corpus_before" not in sync
    assert filtered.structured_content["runs"] == []


def test_a_missing_corpus_stops_the_server_and_creates_nothing(tmp_path: Path) -> None:
    """It once answered every tool with nothing, from a new empty database, looking healthy."""
    missing = tmp_path / "elsewhere" / "corpus.db"

    with pytest.raises(SystemExit, match="no corpus at"):
        serve(state=missing, embedding_model=MODEL)

    assert not missing.parent.exists()


def test_an_empty_corpus_stops_the_server(tmp_path: Path) -> None:
    """The empty database that bug left behind must not pass for a corpus either."""
    empty = tmp_path / "corpus.db"
    CorpusStore(empty).close()

    with pytest.raises(SystemExit, match="holds no papers"):
        serve(state=empty, embedding_model=MODEL)


def test_started_in_the_wrong_folder_the_server_names_where_it_looked(tmp_path: Path) -> None:
    done = subprocess.run(
        [str(VERBATIM), "mcp"],
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", str(tmp_path))},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert done.returncode != 0
    assert str(tmp_path.resolve() / "data" / "corpus.db") in done.stderr
    assert "--directory" in done.stderr
    assert not (tmp_path / "data").exists()
    assert done.stdout == ""


class Local:
    """A scripted local model that answers from the one passage the corpus holds."""

    model = "scripted-local"

    def __init__(
        self,
        chunk_id: str,
        *,
        quote: str = SENTENCE,
        first_quote: str | None = None,
        subject: str = "magnetic flux through a ring",
        parts: tuple[str, ...] = (QUESTION,),
        hold: threading.Event | None = None,
    ) -> None:
        self.chunk_id, self.quote, self.first_quote = chunk_id, quote, first_quote
        self.subject, self.parts, self.hold = subject, parts, hold
        self.calls: list[str] = []
        self.usage = Usage()

    def complete(self, system, user, schema):
        self.calls.append("complete")
        if self.hold is not None:
            self.hold.wait(10)
        self.usage.record({"prompt_eval_count": 100, "eval_count": 10, "total_duration": 2_000_000_000})
        first = self.first_quote is not None and self.calls.count("complete") == 1
        return {
            "answer": "Magnetic flux threads the ring [1].",
            "citations": [{
                "marker": 1, "chunk_id": self.chunk_id,
                "quoted_text": self.first_quote if first else self.quote,
            }],
        }

    def revise(self, system, user, schema):
        self.calls.append("revise")
        return {
            "answer": "Magnetic flux threads the ring [1].",
            "citations": [{"marker": 1, "chunk_id": self.chunk_id, "quoted_text": SENTENCE}],
        }

    def complete_text(self, system, user):
        self.calls.append("complete_text")
        return "\n".join(self.parts)

    def judge(self, system, user):
        self.calls.append("judge")
        return self.subject


class Encoder:
    """Enough embedder for a job, with no weights: text is close only to text sharing its word."""

    model_id = "stub"

    def __init__(self, word: str = "ring") -> None:
        self.model = EmbeddingModel(model_id="stub", dimensions=3, normalized=True, revision="v0")
        self.word = word

    def embed_passages(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0] if self.word in text.lower() else [0.0, 1.0, 0.0]


def job_server(state: Path, model: Local):
    return build_server(state, MODEL, completion=model, encoder=Encoder())


async def _collected(client, started) -> dict:
    assert started.is_error is not True, _text(started)
    job_id = started.structured_content["job_id"]
    for _ in range(50):
        view = (await client.call_tool("get_job", {"job_id": job_id, "wait_seconds": 5})).structured_content
        if view["status"] not in ("queued", "running"):
            return view
    raise AssertionError("the job never finished")


@pytest.mark.anyio
async def test_the_model_tools_appear_only_when_the_server_has_a_model(state: Path) -> None:
    async with Client(build_server(state, MODEL)) as client:
        plain = await _tools(client)
    async with Client(job_server(state, Local(_chunk_id(state)))) as client:
        answering = await _tools(client)

    assert set(plain) == AGENT_TOOLS | CLIENT_ONLY
    assert set(answering) == AGENT_TOOLS | CLIENT_ONLY | JOB_TOOLS


@pytest.mark.anyio
async def test_the_job_tools_say_what_they_change(state: Path) -> None:
    async with Client(job_server(state, Local(_chunk_id(state)))) as client:
        listed = await _tools(client)

    for name in ("ask", "report"):
        hints = listed[name].annotations
        assert listed[name].title
        # A job writes a ledger row and spends the model, so it is not a read.
        assert hints.read_only_hint is False and hints.destructive_hint is False
        assert hints.idempotent_hint is False and hints.open_world_hint is False
    assert listed["get_job"].annotations.read_only_hint is True
    assert listed["cancel_job"].annotations.read_only_hint is False
    assert listed["get_job"].output_schema is not None
    strategy = json.dumps(listed["ask"].input_schema["properties"]["strategy"])
    assert all(choice in strategy for choice in ("default", "plan", "deep"))
    assert "ctx" not in listed["get_job"].input_schema["properties"]


@pytest.mark.anyio
async def test_ask_starts_a_job_at_once_and_get_job_returns_the_checked_answer(state: Path) -> None:
    model = Local(_chunk_id(state))

    async with Client(job_server(state, model)) as client:
        started = await client.call_tool("ask", {"question": QUESTION})
        done = await _collected(client, started)

    assert started.structured_content["status"] in ("queued", "running", "done")
    assert done["status"] == "done" and done["asked"] == QUESTION
    result = done["result"]
    assert result["answer"] == "Magnetic flux threads the ring [1]."
    (cited,) = result["citations"]
    assert cited["quote"] == SENTENCE and cited["locator"] == "p. 1"
    assert cited["arxiv_id"] == "2608.11111v1" and cited["title"] == "Paper 2608.11111v1"
    assert cited["url"] == "https://arxiv.org/abs/2608.11111v1"
    assert result["references"].startswith("[1] ")
    assert result["rejected"] == []
    assert "Answered from" in result["evidence"][0]
    assert result["subject_warning"] is None
    assert result["passages"][0]["chunk_id"] == cited["chunk_id"]
    assert result["models"]["writer"] == "scripted-local"
    assert result["cost"].startswith("1 model call")
    assert "unchanged" in result["note"] and "never instructions" in result["note"]
    assert model.calls == ["complete", "judge"]


@pytest.mark.anyio
async def test_a_quote_the_check_rejects_is_reported_and_marked_in_the_answer(state: Path) -> None:
    model = Local(_chunk_id(state), quote="Magnetic flux threads the loop")

    async with Client(job_server(state, model)) as client:
        done = await _collected(client, await client.call_tool("ask", {"question": QUESTION}))

    result = done["result"]
    assert result["citations"] == []
    assert result["rejected"][0]["reason"] == "quoted text does not appear in the cited passage"
    assert "[unverified]" in result["answer"]


@pytest.mark.anyio
async def test_the_repair_runs_inside_the_job_when_reflection_is_on(state: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "REFLECT", True)
    model = Local(_chunk_id(state), quote="Magnetic flux threads the loop")

    async with Client(job_server(state, model)) as client:
        done = await _collected(client, await client.call_tool("ask", {"question": QUESTION}))

    assert "revise" in model.calls
    assert [cited["quote"] for cited in done["result"]["citations"]] == [SENTENCE]


@pytest.mark.anyio
async def test_the_answer_warns_when_its_passages_are_about_something_else(state: Path) -> None:
    model = Local(_chunk_id(state), subject="medieval poetry")

    async with Client(job_server(state, model)) as client:
        done = await _collected(client, await client.call_tool("ask", {"question": QUESTION}))

    assert "medieval poetry" in done["result"]["subject_warning"]
    assert any("medieval poetry" in line for line in done["result"]["evidence"])


@pytest.mark.anyio
async def test_plan_answers_a_compound_question_in_parts(state: Path) -> None:
    parts = (QUESTION, "What is magnetic flux?")
    model = Local(_chunk_id(state), parts=parts)

    async with Client(job_server(state, model)) as client:
        done = await _collected(
            client,
            await client.call_tool(
                "ask", {"question": "What threads the ring, and what is magnetic flux?", "strategy": "plan"}
            ),
        )

    assert done["result"]["parts"] == list(parts)
    assert done["result"]["citations"][0]["quote"] == SENTENCE
    assert model.calls[0] == "complete_text"


@pytest.mark.anyio
async def test_deep_climbs_the_ladder_and_reports_each_rung(state: Path) -> None:
    model = Local(_chunk_id(state), first_quote="Magnetic flux never threads anything")

    async with Client(job_server(state, model)) as client:
        done = await _collected(
            client, await client.call_tool("ask", {"question": QUESTION, "strategy": "deep"})
        )

    ladder = done["result"]["ladder"]
    assert ladder["attempts"][0]["verdict"] == "retry"
    assert ladder["attempts"][-1]["verdict"] == "accepted"
    assert ladder["outcome"] == "accepted"
    assert done["result"]["citations"][0]["quote"] == SENTENCE


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool", "arguments", "complaint"),
    [
        ("ask", {"question": "  "}, "question is empty"),
        ("ask", {"question": QUESTION, "strategy": "deep", "arxiv_id": "2608.11111"}, "no effect with deep"),
        ("ask", {"question": QUESTION, "arxiv_id": "9999.99999"}, "no paper"),
        ("report", {"topic": ""}, "topic is empty"),
        ("report", {"topic": "flux", "sections": 7}, "sections must be"),
        ("get_job", {"job_id": "nope"}, "no job 'nope'"),
        ("get_job", {"job_id": "x", "wait_seconds": 99}, "wait_seconds must be"),
        ("cancel_job", {"job_id": "nope"}, "no job 'nope'"),
    ],
)
async def test_a_bad_job_argument_reaches_the_client_as_a_readable_error(
    state: Path, tool: str, arguments: dict, complaint: str
) -> None:
    async with Client(job_server(state, Local(_chunk_id(state)))) as client:
        result = await client.call_tool(tool, arguments)

    assert result.is_error is True
    assert complaint in _text(result)


@pytest.mark.anyio
async def test_a_job_writes_one_ledger_row_as_the_command_would(state: Path) -> None:
    async with Client(job_server(state, Local(_chunk_id(state)))) as client:
        await _collected(client, await client.call_tool("ask", {"question": QUESTION}))

    with CorpusStore(state) as corpus:
        (row,) = recent_runs(corpus)
    assert row["command"] == "ask" and row["ok"] is True
    assert row["inputs"]["via"] == "mcp" and row["inputs"]["strategy"] == "default"
    assert row["outputs"]["citations"] == 1


@pytest.mark.anyio
async def test_job_calls_are_logged_without_the_answer(state: Path) -> None:
    async with Client(job_server(state, Local(_chunk_id(state)))) as client:
        started = await client.call_tool("ask", {"question": f"{QUESTION} {'x' * 400}"})
        await _collected(client, started)
        history = await client.call_tool("list_runs", {})

    calls = history.structured_content["runs"]
    assert {"ask", "get_job"} <= {call["command"] for call in calls}
    asked = next(call for call in calls if call["command"] == "ask")
    assert len(asked["inputs"]["question"]) <= 200
    assert asked["outputs"]["job_id"] == started.structured_content["job_id"]
    assert "threads the ring [1]" not in json.dumps(calls)


@pytest.mark.anyio
async def test_a_report_job_returns_the_review_with_its_references(state: Path) -> None:
    headings = (QUESTION, "How does magnetic flux behave?")
    model = Local(_chunk_id(state), parts=headings)

    async with Client(job_server(state, model)) as client:
        done = await _collected(
            client, await client.call_tool("report", {"topic": "magnetic flux", "sections": 2})
        )

    result = done["result"]
    assert done["asked"] == "magnetic flux"
    assert f"## {QUESTION}" in result["markdown"] and "## References" in result["markdown"]
    assert [section["heading"] for section in result["sections"]] == list(headings)
    assert all(section["citations"] == 1 for section in result["sections"])
    with CorpusStore(state) as corpus:
        (row,) = recent_runs(corpus)
    assert row["command"] == "report" and row["inputs"]["via"] == "mcp"


@pytest.mark.anyio
async def test_the_instructions_explain_jobs_only_when_they_are_on(state: Path) -> None:
    async with Client(build_server(state, MODEL)) as client:
        plain = client.instructions
    async with Client(job_server(state, Local(_chunk_id(state)))) as client:
        answering = client.instructions

    assert "get_job" not in plain
    assert "get_job" in answering and "unchanged" in answering
    assert len(answering) < 2_000


@pytest.mark.anyio
async def test_waiting_tells_a_client_that_asks_what_the_job_is_doing(state: Path) -> None:
    gate = threading.Event()
    model = Local(_chunk_id(state), hold=gate)
    told: list[str] = []

    async def progress(value, total, message) -> None:
        told.append(message or "")
        if "writing the answer" in (message or ""):
            gate.set()

    async with Client(job_server(state, model)) as client:
        started = await client.call_tool("ask", {"question": QUESTION})
        waited = await client.call_tool(
            "get_job",
            {"job_id": started.structured_content["job_id"], "wait_seconds": 10},
            progress_callback=progress,
        )

    assert any("writing the answer" in message for message in told)
    assert waited.structured_content["status"] == "done"


@pytest.mark.anyio
async def test_a_cloud_model_is_refused_and_nothing_leaves_the_machine(state: Path, monkeypatch) -> None:
    def no_network(*args, **kwargs):
        raise AssertionError("contacted a server")

    monkeypatch.setattr("verbatim.jobs.urlopen", no_network)
    monkeypatch.setattr("verbatim.answers.urlopen", no_network)

    async with Client(build_server(state, MODEL, llm="gemma4:31b-cloud")) as client:
        listed = await _tools(client)
        refused = await client.call_tool("ask", {"question": QUESTION})

    assert JOB_TOOLS <= set(listed)
    assert refused.is_error is True and "cloud model" in _text(refused)
    with CorpusStore(state) as corpus:
        assert recent_runs(corpus) == []


@pytest.mark.anyio
async def test_ollama_not_running_is_refused_when_a_job_is_asked_for(state: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OLLAMA_HOST", "http://127.0.0.1:9")

    async with Client(build_server(state, MODEL, llm="gemma4:e4b-mlx")) as client:
        refused = await client.call_tool("report", {"topic": "magnetic flux"})

    assert refused.is_error is True and "Ollama is not answering" in _text(refused)


@pytest.mark.anyio
async def test_the_command_line_turns_the_model_tools_on(tmp_path: Path) -> None:
    """Over real stdio, so the flag, the tools and a clean stdout are checked together. No model runs."""
    copy = tmp_path / "corpus.db"
    shutil.copy(SAMPLE, copy)
    params = StdioServerParameters(
        command=str(VERBATIM),
        args=["mcp", "--state", str(copy), "--llm", "gemma4:31b-cloud"],
        cwd=str(tmp_path),
        env={"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", str(tmp_path))},
    )

    async with Client(params) as client:
        listed = await _tools(client)
        refused = await client.call_tool("ask", {"question": "what is attention"})

    assert JOB_TOOLS <= set(listed)
    assert refused.is_error is True and "cloud model" in _text(refused)
