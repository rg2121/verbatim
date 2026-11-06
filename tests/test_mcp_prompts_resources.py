"""The server's prompts and resources, driven as a client drives them."""

import json
import os
from pathlib import Path
import shutil

from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError
import pytest

from verbatim.arxiv import Paper
from verbatim.chunking import extract_pdf_chunks
from verbatim.corpus import CorpusStore
import verbatim.graph_supervisor as graph_supervisor
import verbatim.mcp_server as mcp_server
from verbatim.mcp_server import PAPER_TEXT_NOTE, build_server
from verbatim.references import Reference
from verbatim.report import MAX_SECTIONS

from helpers import chunked_corpus, downloaded
from test_mcp_server import MODEL, SAMPLE, SENTENCE, VERBATIM, Local, job_server

CC_BY = "http://creativecommons.org/licenses/by/4.0/"
NONEXCLUSIVE = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"
# Cited by the corpus and not held by it.
ABSENT = "1706.03762"
OVERVIEW = "verbatim://corpus/overview"
QUESTION = "What threads the ring?"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _paper(arxiv_id: str, license_url: str | None, categories: list[str]) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        summary=f"The abstract of {arxiv_id}.",
        authors=["Ada Lovelace"],
        categories=categories,
        published="2026-08-01T10:00:00Z",
        updated="2026-08-08T10:00:00Z",
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        license_url=license_url,
    )


@pytest.fixture
def state(tmp_path: Path) -> Path:
    """Three papers: the first cited by both others, both citing one the corpus lacks."""
    papers = [
        (
            _paper("2608.11111v1", CC_BY, ["cs.CL", "cs.IR"]),
            [f"## Intro\n\n{SENTENCE}", "## Method\n\nMore words."],
        ),
        (_paper("2608.22222v1", NONEXCLUSIVE, ["cs.CL"]), ["## Intro\n\nA second paper."]),
        (_paper("2608.33333v1", None, ["cs.LG", "cs.CL"]), ["## Intro\n\nA third paper."]),
    ]
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        for paper, pages in papers:
            record = downloaded(tmp_path, paper.arxiv_id)
            corpus.upsert_papers([paper])
            corpus.record_downloads([record])
            corpus.record_chunks(
                extract_pdf_chunks([record], extract_pages=lambda _, captured=pages: captured)
            )
        corpus.record_references(
            [
                Reference("2608.22222v1", 1, "Paper one.", cited_arxiv_id="2608.11111"),
                Reference("2608.22222v1", 2, "Attention.", cited_arxiv_id=ABSENT),
                Reference("2608.33333v1", 1, "Paper one.", cited_arxiv_id="2608.11111"),
                Reference("2608.33333v1", 2, "Paper two.", cited_arxiv_id="2608.22222"),
                Reference("2608.33333v1", 3, "Attention.", cited_arxiv_id=ABSENT),
            ],
            "a-rendering",
        )
    return tmp_path / "corpus.db"


def _passages(state: Path, arxiv_id: str | None = None) -> int:
    with CorpusStore(state) as corpus:
        if arxiv_id is None:
            return corpus.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return corpus.connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()[0]


async def _read(client, uri: str) -> dict:
    read = await client.read_resource(uri)
    (content,) = read.contents
    assert content.mime_type == "application/json"
    return json.loads(content.text)


async def _prompt_text(client, name: str, arguments: dict) -> str:
    got = await client.get_prompt(name, arguments)
    (message,) = got.messages
    assert message.role == "user"
    return message.content.text


def _call_in(text: str) -> dict:
    """The arguments a job prompt spells out, on a line of their own."""
    (line,) = [line for line in text.splitlines() if line.startswith("{")]
    return json.loads(line)


@pytest.mark.anyio
async def test_the_prompts_offered_follow_what_the_server_can_run(state: Path) -> None:
    """Without a model only the client writes, so only the prompt that checks its quotes."""
    async with Client(build_server(state, MODEL)) as client:
        plain = {prompt.name: prompt for prompt in (await client.list_prompts()).prompts}
    async with Client(job_server(state, Local("unused"))) as client:
        answering = {prompt.name: prompt for prompt in (await client.list_prompts()).prompts}

    assert set(plain) == {"cite_check"}
    assert set(answering) == {"cite_check", "verbatim_answer", "literature_review"}
    for prompt in answering.values():
        assert prompt.title and prompt.description
        assert all(argument.description for argument in prompt.arguments)
    named = {name: [a.name for a in prompt.arguments] for name, prompt in answering.items()}
    required = {name: {a.name for a in prompt.arguments if a.required} for name, prompt in answering.items()}
    assert named["verbatim_answer"] == ["question", "strategy", "supervisor", "arxiv_id"]
    assert named["literature_review"] == ["topic", "sections", "agents", "supervisor"]
    assert required == {
        "cite_check": {"question"}, "verbatim_answer": {"question"}, "literature_review": {"topic"},
    }


@pytest.mark.anyio
async def test_cite_check_walks_the_client_through_the_check_with_tools_it_has(state: Path) -> None:
    async with Client(build_server(state, MODEL)) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
        text = await _prompt_text(client, "cite_check", {"question": QUESTION})

    named = {"search_chunks", "read_pages", "verify_citations", "export_citations"}
    assert named <= tools
    assert all(name in text for name in named)
    assert QUESTION in text
    assert "word for word" in text and "from memory" in text
    assert PAPER_TEXT_NOTE in text


@pytest.mark.anyio
async def test_verbatim_answer_spells_out_the_exact_call(state: Path, monkeypatch) -> None:
    """The arguments as JSON, so the client cannot reword the question on the way."""
    monkeypatch.setattr(graph_supervisor, "installed", lambda: True)

    async with Client(job_server(state, Local("unused"))) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
        deep = await _prompt_text(
            client, "verbatim_answer",
            {"question": QUESTION, "strategy": "deep", "supervisor": "langgraph"},
        )
        scoped = await _prompt_text(
            client, "verbatim_answer", {"question": QUESTION, "arxiv_id": "2608.11111"}
        )

    assert _call_in(deep) == {"question": QUESTION, "strategy": "deep", "supervisor": "langgraph"}
    assert _call_in(scoped) == {
        "question": QUESTION, "strategy": "default", "supervisor": "builtin", "arxiv_id": "2608.11111",
    }
    assert {"ask", "get_job"} <= tools
    assert "Call ask" in deep and "get_job" in deep and "unchanged" in deep


@pytest.mark.anyio
async def test_literature_review_reads_its_arguments_from_text(state: Path) -> None:
    """Prompt arguments arrive as text, so its number and its yes or no are read here."""
    async with Client(job_server(state, Local("unused"))) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
        chosen = await _prompt_text(
            client, "literature_review", {"topic": "magnetic flux", "sections": "3", "agents": "yes"}
        )
        plain = await _prompt_text(client, "literature_review", {"topic": "magnetic flux"})

    assert _call_in(chosen) == {
        "topic": "magnetic flux", "sections": 3, "agents": True, "supervisor": "builtin",
    }
    assert _call_in(plain) == {
        "topic": "magnetic flux", "sections": 4, "agents": False, "supervisor": "builtin",
    }
    assert "report" in tools
    assert "Call report" in chosen and "markdown" in chosen and "unchanged" in chosen


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("prompt", "arguments", "complaint"),
    [
        ("cite_check", {"question": "  "}, "question is empty"),
        ("verbatim_answer", {"question": "  "}, "question is empty"),
        ("verbatim_answer", {"question": QUESTION, "strategy": "sideways"}, "strategy must be one of"),
        (
            "verbatim_answer",
            {"question": QUESTION, "supervisor": "langgraph"},
            "only applies with strategy deep",
        ),
        (
            "verbatim_answer",
            {"question": QUESTION, "strategy": "deep", "arxiv_id": "2608.11111"},
            "no effect with deep",
        ),
        (
            "verbatim_answer",
            {"question": QUESTION, "strategy": "deep", "supervisor": "crew"},
            "supervisor must be one of",
        ),
        ("literature_review", {"topic": " "}, "topic is empty"),
        ("literature_review", {"topic": "flux", "sections": "four"}, "sections must be a whole number"),
        (
            "literature_review",
            {"topic": "flux", "sections": str(MAX_SECTIONS + 1)},
            "sections must be between",
        ),
        ("literature_review", {"topic": "flux", "agents": "maybe"}, "agents must be yes or no"),
        ("literature_review", {"topic": "flux", "supervisor": "langgraph"}, "only applies with agents"),
    ],
)
async def test_a_prompt_refuses_what_its_tool_would_refuse_and_says_why(
    state: Path, prompt: str, arguments: dict, complaint: str
) -> None:
    """The SDK turns any other error in a prompt into "Internal server error"."""
    async with Client(job_server(state, Local("unused"))) as client:
        with pytest.raises(MCPError) as refused:
            await client.get_prompt(prompt, arguments)

    assert complaint in str(refused.value)


@pytest.mark.anyio
async def test_a_prompt_for_langgraph_without_its_extra_says_how_to_install_it(
    state: Path, monkeypatch
) -> None:
    monkeypatch.setattr(graph_supervisor, "installed", lambda: False)

    async with Client(job_server(state, Local("unused"))) as client:
        with pytest.raises(MCPError) as refused:
            await client.get_prompt(
                "verbatim_answer", {"question": QUESTION, "strategy": "deep", "supervisor": "langgraph"}
            )

    assert "uv sync --extra langgraph" in str(refused.value)


@pytest.mark.anyio
async def test_the_overview_says_what_the_corpus_holds_and_lacks(state: Path) -> None:
    async with Client(build_server(state, MODEL)) as client:
        (listed,) = (await client.list_resources()).resources
        overview = await _read(client, OVERVIEW)

    assert str(listed.uri) == OVERVIEW
    assert listed.title and listed.description and listed.mime_type == "application/json"
    assert overview["papers"] == 3 and overview["passages"] == _passages(state)
    assert overview["published"] == {"earliest": "2026-08-01", "latest": "2026-08-01"}
    assert overview["updated_up_to"] == "2026-08-08"
    assert overview["licences"] == {"arxiv-nonexclusive": 1, "open": 1, "unknown": 1}
    assert overview["categories"] == {"cs.CL": 3, "cs.IR": 1, "cs.LG": 1}
    assert overview["citation_graph"] == {"edges": 3, "connected_papers": 3, "bibliography_entries": 5}
    assert overview["most_cited_held"] == [
        {"arxiv_id": "2608.11111v1", "title": "Paper 2608.11111v1", "cited_by": 2},
        {"arxiv_id": "2608.22222v1", "title": "Paper 2608.22222v1", "cited_by": 1},
    ]
    assert overview["most_cited_missing"] == [{"arxiv_id": ABSENT, "cited_by": 2}]
    assert "not live arXiv" in overview["note"] and PAPER_TEXT_NOTE in overview["note"]


@pytest.mark.anyio
async def test_the_overview_is_counted_once_until_the_corpus_changes(state: Path, monkeypatch) -> None:
    """It takes about 2 s on the real corpus, on the one thread every search waits for."""
    counted = []
    real = mcp_server.corpus_overview

    def counting(corpus):
        counted.append(corpus)
        return real(corpus)

    monkeypatch.setattr(mcp_server, "corpus_overview", counting)

    async with Client(build_server(state, MODEL)) as client:
        first = await _read(client, OVERVIEW)
        again = await _read(client, OVERVIEW)
        # Another connection, as a sync or a job's ledger row would be.
        with CorpusStore(state) as corpus:
            corpus.upsert_papers([_paper("2608.44444v1", CC_BY, ["cs.CL"])])
        changed = await _read(client, OVERVIEW)

    assert len(counted) == 2
    assert first == again and first["papers"] == 3 and changed["papers"] == 4


@pytest.mark.anyio
async def test_a_paper_reads_by_id_with_what_cites_it_and_what_it_cites(state: Path) -> None:
    async with Client(build_server(state, MODEL)) as client:
        (template,) = (await client.list_resource_templates()).resource_templates
        cited = await _read(client, "verbatim://paper/2608.11111")
        citing = await _read(client, "verbatim://paper/2608.33333v1")

    assert template.uri_template == "verbatim://paper/{+arxiv_id}"
    assert template.title and template.description and template.mime_type == "application/json"
    assert cited["arxiv_id"] == "2608.11111v1" and cited["title"] == "Paper 2608.11111v1"
    assert cited["abstract"] == "The abstract of 2608.11111v1." and cited["licence"] == "open"
    assert cited["url"] == "https://arxiv.org/abs/2608.11111v1" and cited["license_url"] == CC_BY
    assert cited["cited_by"] == 2 and cited["references_held"] == 0
    assert cited["passages"] == _passages(state, "2608.11111v1")
    assert cited["positions"] == {"first": 1, "last": 2}
    assert citing["cited_by"] == 0 and citing["references_held"] == 2
    assert cited["note"] == PAPER_TEXT_NOTE


@pytest.mark.anyio
async def test_an_old_style_identifier_keeps_its_slash(tmp_path: Path) -> None:
    """Two papers in the real corpus have identifiers like hep-th/9901001."""
    with chunked_corpus(tmp_path, {"hep-th/9901001v1": [f"## Intro\n\n{SENTENCE}"]}):
        pass

    async with Client(build_server(tmp_path / "corpus.db", MODEL)) as client:
        record = await _read(client, "verbatim://paper/hep-th/9901001")

    assert record["arxiv_id"] == "hep-th/9901001v1"


@pytest.mark.anyio
@pytest.mark.parametrize("identifier", ["9999.99999", "Paper"])
async def test_a_paper_the_corpus_lacks_is_an_error_that_says_so(state: Path, identifier: str) -> None:
    """By ID only: a title search would let one address name whichever paper matched best."""
    async with Client(build_server(state, MODEL)) as client:
        with pytest.raises(MCPError) as missing:
            await client.read_resource(f"verbatim://paper/{identifier}")

    assert "no paper" in str(missing.value)


@pytest.mark.anyio
async def test_reads_are_logged_and_leave_the_corpus_as_it_was(state: Path) -> None:
    before = state.read_bytes()

    async with Client(build_server(state, MODEL)) as client:
        await _read(client, OVERVIEW)
        await _read(client, "verbatim://paper/2608.11111")
        with pytest.raises(MCPError):
            await client.read_resource("verbatim://paper/9999.99999")
        history = await client.call_tool("list_runs", {"name": "read_resource"})

    assert state.read_bytes() == before
    missing, paper, overview = history.structured_content["runs"]
    assert overview["inputs"] == {"uri": OVERVIEW} and overview["ok"] is True
    assert overview["outputs"]["characters"] > 0
    assert paper["inputs"] == {"uri": "verbatim://paper/2608.11111"} and paper["ok"] is True
    assert missing["ok"] is False and "no paper" in missing["error"]


@pytest.mark.anyio
async def test_the_instructions_point_to_the_overview(state: Path) -> None:
    async with Client(job_server(state, Local("unused"))) as client:
        instructions = client.instructions

    assert OVERVIEW in instructions and len(instructions) < 2_000


@pytest.mark.anyio
async def test_prompts_and_resources_work_over_real_stdio_on_a_copy_of_the_sample(tmp_path: Path) -> None:
    """The process a client launches, with no model and no embedder loaded."""
    copy = tmp_path / "corpus.db"
    shutil.copy(SAMPLE, copy)
    with CorpusStore(copy) as corpus:
        papers, first = corpus.connection.execute(
            "SELECT COUNT(*), MIN(arxiv_id) FROM papers WHERE is_latest = 1"
        ).fetchone()
    params = StdioServerParameters(
        command=str(VERBATIM),
        args=["mcp", "--state", str(copy)],
        cwd=str(tmp_path),
        env={"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", str(tmp_path))},
    )

    async with Client(params) as client:
        prompts = [prompt.name for prompt in (await client.list_prompts()).prompts]
        overview = await _read(client, OVERVIEW)
        record = await _read(client, f"verbatim://paper/{first}")

    assert prompts == ["cite_check"]
    assert overview["papers"] == papers
    assert record["arxiv_id"] == first
