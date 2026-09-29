"""The MCP front end onto the corpus."""

from pathlib import Path

import pytest

from verbatim.mcp_server import render_result
from verbatim.tools import TOOL_SCHEMAS


@pytest.mark.anyio
async def test_the_server_offers_exactly_the_tools_the_agent_has(tmp_path: Path) -> None:
    """Two front ends onto one set of operations, so they cannot drift apart."""
    from helpers import chunked_corpus
    from verbatim.mcp_server import build_server

    with chunked_corpus(tmp_path, {"2608.11111v1": ["Body text."]}):
        pass

    listed = await build_server(tmp_path / "corpus.db", "BAAI/bge-small-en-v1.5").list_tools()

    assert {tool.name: tool.description for tool in listed} == {
        schema["name"]: schema["description"] for schema in TOOL_SCHEMAS
    }


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
async def test_a_client_can_list_and_call_tools_over_a_real_session(tmp_path: Path) -> None:
    """Exercise the protocol itself, not just the shapes handed to it."""
    from helpers import chunked_corpus
    from verbatim.mcp_server import build_server

    with chunked_corpus(tmp_path, {"2608.11111v1": ["Superconducting qubits decohere."]}):
        pass  # the corpus file is written and closed; the server opens its own

    server = build_server(tmp_path / "corpus.db", "BAAI/bge-small-en-v1.5")

    listed = await server.list_tools()
    result = await server.call_tool("list_papers", {"limit": 5})

    assert "search_chunks" in {tool.name for tool in listed}
    # Every tool reads the corpus; a client may be told none of them writes.
    assert all(tool.annotations.read_only_hint for tool in listed)
    assert "2608.11111v1" in str(result)


@pytest.mark.anyio
async def test_a_tool_declares_the_arguments_it_actually_takes(tmp_path: Path) -> None:
    """The server derives the schema from the signature, so it must be real."""
    from helpers import chunked_corpus
    from verbatim.mcp_server import build_server

    with chunked_corpus(tmp_path, {"2608.11111v1": ["Body text."]}):
        pass

    server = build_server(tmp_path / "corpus.db", "BAAI/bge-small-en-v1.5")
    listed = {tool.name: tool for tool in await server.list_tools()}

    search = listed["search_chunks"].input_schema
    assert "query" in search["properties"]
    assert search["required"] == ["query"]
    assert "arxiv_id" in search["properties"]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_an_array_parameter_is_published_as_a_list_not_a_string(tmp_path: Path) -> None:
    """An array parameter is published as a list, not a string."""
    from helpers import chunked_corpus
    from verbatim.mcp_server import build_server

    with chunked_corpus(tmp_path, {"2608.11111v1": ["Body text."]}):
        pass

    server = build_server(tmp_path / "corpus.db", "BAAI/bge-small-en-v1.5")
    listed = {tool.name: tool for tool in await server.list_tools()}

    pages = listed["read_pages"].input_schema["properties"]["pages"]
    assert pages["type"] == "array"
    assert pages["items"]["type"] == "integer"
    # Every tool still declares exactly the arguments the agent's schema does.
    for schema in TOOL_SCHEMAS:
        published = set(listed[schema["name"]].input_schema["properties"])
        assert published == set(schema["parameters"]["properties"])
