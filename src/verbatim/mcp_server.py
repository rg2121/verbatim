"""The corpus over the Model Context Protocol: the agent's tools, offered to any MCP client."""

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


def build_server(state: Path, embedding_model: str) -> Any:
    """Create the MCP server; the corpus and embedder are opened on first use."""
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    server = MCPServer(SERVER_NAME)
    # Tools run on one worker thread: off the event loop, and on the only thread
    # allowed to use the SQLite connection, so calls serialise.
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verbatim-corpus")
    resources: dict[str, Any] = {}

    def open_resources() -> tuple[CorpusStore, FastEmbedEmbedder]:
        if "corpus" not in resources:
            resources["corpus"] = CorpusStore(state)
        if "embedder" not in resources:
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
    """Wrap a tool schema as a function whose signature MCPServer reads.

    Parameters are declared explicitly, since **kwargs would publish a tool
    that accepts nothing.
    """
    properties = schema["parameters"].get("properties", {})
    required = set(schema["parameters"].get("required", ()))
    order = sorted(properties, key=lambda name: (name not in required, name))
    types_by_name = {
        "string": "str", "integer": "int", "number": "float", "boolean": "bool",
    }

    def annotation(spec: dict[str, Any]) -> str:
        """The Python type MCP derives this parameter from; arrays become lists."""
        if spec.get("type") == "array":
            item = (spec.get("items") or {}).get("type")
            return f"list[{types_by_name.get(item, 'str')}]"
        return types_by_name.get(spec.get("type"), "str")

    parameters = ", ".join(
        f"{name}: {annotation(properties[name])}"
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
    """Serialise a tool result as JSON, keeping the provenance fields a citation needs."""
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
