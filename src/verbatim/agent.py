"""Tool-routed answering: the model picks a tool, code runs it.

Recency questions go to the catalogue, a named paper to a lookup and scoped
search, anything else to passage retrieval. The answer itself still goes
through the verified-citation path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from verbatim.answers import Answer, Completion, answer_from_matches
from verbatim.chunking import Chunk
from verbatim.corpus import ChunkMatch, CorpusStore
from verbatim.embedding import Embedder
from verbatim.tools import TOOL_SCHEMAS, call_tool

MAX_STEPS = 4


@dataclass
class AgentTrace:
    """What the agent did, so an answer can be explained rather than trusted."""

    steps: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def describe(self) -> str:
        return " → ".join(f"{name}({_brief(args)})" for name, args in self.steps) or "no tools used"


class ToolCaller:
    """A model that can choose a tool, as distinct from one that writes prose."""

    def choose_tool(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> tuple[str, dict[str, Any]] | None:
        raise NotImplementedError


def gather(
    corpus: CorpusStore,
    embedder: Embedder,
    caller: ToolCaller,
    question: str,
    *,
    max_steps: int = MAX_STEPS,
) -> tuple[list[ChunkMatch], AgentTrace]:
    """Call tools until one returns passages or `max_steps` runs out."""
    trace = AgentTrace()
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    seen: dict[str, Chunk] = {}

    for _ in range(max_steps):
        choice = caller.choose_tool(SYSTEM_PROMPT, messages, TOOL_SCHEMAS)
        if choice is None:
            break
        name, arguments = choice
        try:
            result = call_tool(name, arguments, corpus, embedder)
        except (ValueError, TypeError) as error:
            # A malformed call goes back to the model as an error, not a crash.
            result = {"error": str(error)}
        trace.steps.append((name, arguments))

        for item in result if isinstance(result, list) else [result]:
            if isinstance(item, dict) and "chunk_id" in item:
                chunk = corpus.chunks_by_id([item["chunk_id"]])
                if chunk:
                    seen[item["chunk_id"]] = chunk[0]
            elif isinstance(item, dict) and "title" in item:
                trace.metadata.append(item)

        messages.append({"role": "assistant", "content": f"Called {name}."})
        messages.append({"role": "user", "content": _summarise(result)})
        if seen:
            break  # evidence in hand; answer from it rather than keep calling

    matches = [ChunkMatch(chunk=c, score=1.0, snippet=c.text[:160]) for c in seen.values()]
    return matches, trace


def answer_question(
    corpus: CorpusStore,
    embedder: Embedder,
    caller: ToolCaller,
    completion: Completion,
    question: str,
    *,
    max_steps: int = MAX_STEPS,
) -> tuple[Answer, AgentTrace]:
    """Route the question, then answer it with citations that are verified."""
    matches, trace = gather(corpus, embedder, caller, question, max_steps=max_steps)
    if not matches and not trace.metadata and trace.steps:
        # A tool ran but matched nothing: say which, rather than "no passages match".
        used = ", ".join(dict.fromkeys(name for name, _ in trace.steps))
        return (
            Answer(
                question=question,
                text=f"The corpus has no records matching this question (searched via {used}).",
                citations=[],
                rejected=[],
            ),
            trace,
        )
    if not matches and trace.metadata:
        return records_answer(question, trace.metadata), trace
    return answer_from_matches(corpus, completion, question, matches), trace


def records_answer(question: str, records: list[dict[str, Any]]) -> Answer:
    """Answer a metadata question, such as recency, by listing catalogue records.

    The records are listed rather than narrated by a model, so nothing unsupported
    is claimed. The orchestrator's agent rung uses this too.
    """
    return Answer(
        question=question,
        text=_format_metadata(records),
        citations=[],
        rejected=[],
    )


SYSTEM_PROMPT = (
    "You are a research assistant over a local arXiv corpus.\n"
    "Choose exactly one tool per step to gather what the question needs:\n"
    "- questions about what is new or recent -> list_papers\n"
    "- questions naming a specific paper -> get_paper, then search_chunks with its arxiv_id\n"
    "- questions about what the literature says -> search_chunks\n"
    "Never answer from memory; the corpus is the only source."
)


def _summarise(result: Any) -> str:
    """Feed a compact form of the result back, not whole passages."""
    if isinstance(result, list):
        return f"{len(result)} result(s): " + json.dumps(
            [{k: v for k, v in item.items() if k != "text"} for item in result[:10]]
        )[:1200]
    return json.dumps(result)[:1200]


def _format_metadata(records: list[dict[str, Any]]) -> str:
    lines = []
    for record in records[:25]:
        updated = str(record.get("updated", ""))[:10]
        lines.append(f"- {record.get('arxiv_id','?')}  {updated}  {record.get('title','')}")
    return "\n".join(lines) if lines else "Nothing in the corpus matches."


def _brief(arguments: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in list(arguments.items())[:2])
