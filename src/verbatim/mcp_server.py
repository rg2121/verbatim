"""Exposing the corpus over the Model Context Protocol.

The same tools the built-in agent uses, offered to any MCP client - a desktop
assistant, a coding agent, an editor - so the corpus can be queried from wherever
the work is happening rather than only through this CLI.

Nothing here reimplements retrieval. The tools, their schemas and their
dispatch already exist in `tools`, because an assistant that answers questions
and a server that answers a client's questions want exactly the same
operations; this is a second front end onto them, not a second copy.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

from verbatim import config
from verbatim.corpus import CorpusStore
from verbatim.embedding import FastEmbedEmbedder
from verbatim.tools import TOOL_SCHEMAS, call_tool

SERVER_NAME = "verbatim"


def as_mcp_tools() -> list[dict[str, Any]]:
    """Restate the existing tool schemas in the shape MCP asks for.

    MCP calls the parameter schema `inputSchema`; the definitions themselves
    are the ones the agent already uses, so the two front ends cannot drift
    apart in what they offer or how they describe it.
    """
    return [
        {
            "name": schema["name"],
            "description": schema["description"],
            "inputSchema": schema["parameters"],
        }
        for schema in TOOL_SCHEMAS
    ]


def build_server(state: Path, embedding_model: str) -> Any:
    """Create the MCP server, holding the corpus open for its lifetime.

    The embedder is constructed lazily: loading an ONNX model costs seconds,
    and a client that only reads metadata should not pay for it.
    """
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    server = MCPServer(SERVER_NAME)
    # Tool bodies are synchronous and hit SQLite and ONNX, so they run off the
    # event loop that serves the client. A SQLite connection may only be used
    # from the thread that made it, so the pool holds exactly one thread and
    # the corpus is opened inside it: every call reaches the same connection,
    # and calls serialise rather than racing.
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verbatim-corpus")
    resources: dict[str, Any] = {}

    def open_resources() -> tuple[CorpusStore, FastEmbedEmbedder]:
        if "corpus" not in resources:
            resources["corpus"] = CorpusStore(state)
        if "embedder" not in resources:
            # Loading an ONNX model costs seconds, so a client that only reads
            # metadata never pays for it.
            resources["embedder"] = FastEmbedEmbedder(model_name=embedding_model)
        return resources["corpus"], resources["embedder"]

    def run_tool(name: str, arguments: dict[str, Any]) -> str:
        corpus, embedder = open_resources()
        return render_result(call_tool(name, arguments, corpus, embedder))

    for schema in TOOL_SCHEMAS:
        server.add_tool(
            _tool_function(schema, worker, run_tool),
            name=schema["name"],
            description=schema["description"],
            # Every tool reads; none of them writes to the corpus.
            annotations=ToolAnnotations(read_only_hint=True),
        )
    return server


def _tool_function(schema: dict[str, Any], worker: ThreadPoolExecutor, run_tool: Any) -> Any:
    """Wrap one tool schema as the callable MCPServer derives its signature from.

    The parameters are declared explicitly rather than as **kwargs, because the
    server builds the input schema from the signature and would otherwise offer
    a tool that accepts nothing.
    """
    properties = schema["parameters"].get("properties", {})
    required = set(schema["parameters"].get("required", ()))
    order = sorted(properties, key=lambda name: (name not in required, name))
    types_by_name = {
        "string": "str", "integer": "int", "number": "float", "boolean": "bool",
    }

    parameters = ", ".join(
        f"{name}: {types_by_name.get(properties[name].get('type'), 'str')}"
        + ("" if name in required else " | None = None")
        for name in order
    )
    body = ", ".join(f"{name!r}: {name}" for name in order)
    source = (
        f"async def {schema['name']}({parameters}):\n"
        f"    arguments = {{{body}}}\n"
        "    given = {k: v for k, v in arguments.items() if v is not None}\n"
        "    loop = asyncio.get_running_loop()\n"
        "    return await loop.run_in_executor(worker, run_tool, name, given)\n"
    )
    namespace: dict[str, Any] = {
        "asyncio": asyncio, "worker": worker, "run_tool": run_tool,
        "name": schema["name"],
    }
    exec(compile(source, f"<tool {schema['name']}>", "exec"), namespace)
    function = namespace[schema["name"]]
    function.__doc__ = schema["description"]
    return function


def render_result(result: Any) -> str:
    """Serialize a tool result for a client that reads text.

    JSON rather than prose: the fields are the provenance - arXiv id, page,
    section, chunk id - and a client that wants to cite what it read needs them
    intact rather than summarised away.
    """
    if result is None:
        return "No result."
    return json.dumps(result, indent=2, ensure_ascii=False, default=str)


def serve(
    *, state: Path | None = None, embedding_model: str | None = None
) -> None:
    """Run the server on stdio, which is how MCP clients launch a local server."""
    build_server(
        state or config.STATE_PATH, embedding_model or config.EMBEDDING_MODEL
    ).run(transport="stdio")
