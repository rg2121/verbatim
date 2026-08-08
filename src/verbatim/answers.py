"""Answering a research question from retrieved passages, with citations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol

from verbatim.citations import Citation, citation_from_chunk
from verbatim.corpus import ChunkMatch, CorpusStore

MODEL = "hosted-model"
MAX_TOKENS = 16_000

SYSTEM_PROMPT = """You answer research questions strictly from the passages provided.

Rules:
- Use only the passages. If they do not answer the question, say so plainly.
- Every factual claim must carry a marker like [1] pointing at a citation.
- quoted_text must be copied character-for-character from the passage it cites.
  Do not paraphrase, reformat, or repair it. It is checked against the source.
- Prefer the shortest quote that supports the claim.
- If passages disagree, say so and cite both rather than picking one.
"""

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer, with [n] markers pointing at citations.",
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "marker": {"type": "integer"},
                    "chunk_id": {"type": "string"},
                    "quoted_text": {"type": "string"},
                },
                "required": ["marker", "chunk_id", "quoted_text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "citations"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Answer:
    """A generated answer and the citations it rests on."""

    question: str
    text: str
    citations: list[Citation]

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "text": self.text,
            "citations": [citation.to_dict() for citation in self.citations],
        }


class Completion(Protocol):
    """A model that returns JSON matching a schema."""

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]: ...


class HostedCompletion:
    """A hosted model via its SDK, constrained to the answer schema."""

    def __init__(self, model: str = MODEL, *, max_tokens: int = MAX_TOKENS) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            import hosted_sdk
        except ImportError as error:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "Answering needs the hosted SDK package; install it with 'uv add hosted-sdk'"
            ) from error

        client = hosted_sdk.Client()
        try:
            # Credentials are resolved when the request is built, not at
            # construction, so this is the call that fails without a key.
            response = client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
        except TypeError as error:
            if "authentication" not in str(error).lower():
                raise
            raise RuntimeError(
                "No hosted credentials found. Set HOSTED_API_KEY, or run "
                "the provider's login command if you have its CLI installed."
            ) from error
        if response.stop_reason == "refusal":
            raise RuntimeError(
                "The model declined to answer this question "
                f"({getattr(response.stop_details, 'category', 'no category given')})."
            )
        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise RuntimeError(f"No text in model response (stop_reason={response.stop_reason})")
        return json.loads(text)


def build_prompt(question: str, matches: list[ChunkMatch]) -> str:
    """Lay out the passages the model may cite, each tagged with its chunk id."""
    blocks = []
    for match in matches:
        chunk = match.chunk
        blocks.append(
            f"<passage chunk_id=\"{chunk.chunk_id}\" "
            f'arxiv_id="{chunk.arxiv_id}" page="{chunk.page_number}">\n'
            f"{chunk.text}\n"
            f"</passage>"
        )
    passages = "\n\n".join(blocks)
    return f"<passages>\n{passages}\n</passages>\n\nQuestion: {question}"


def answer_from_matches(
    corpus: CorpusStore,
    completion: Completion,
    question: str,
    matches: list[ChunkMatch],
) -> Answer:
    """Generate an answer over already-retrieved passages."""
    if not matches:
        return Answer(question=question, text="No passages in the corpus match this question.", citations=[])

    payload = completion.complete(SYSTEM_PROMPT, build_prompt(question, matches), ANSWER_SCHEMA)
    chunks = {match.chunk.chunk_id: match.chunk for match in matches}
    papers = corpus.papers_by_id(sorted({match.chunk.arxiv_id for match in matches}))

    citations: list[Citation] = []
    for raw in payload.get("citations", []):
        chunk = chunks.get(raw.get("chunk_id", ""))
        paper = papers.get(chunk.arxiv_id) if chunk else None
        if chunk is None or paper is None:
            # The model named a passage that was not offered to it.
            continue
        citations.append(citation_from_chunk(chunk, paper, raw.get("quoted_text", "")))
    return Answer(question=question, text=payload.get("answer", ""), citations=citations)
