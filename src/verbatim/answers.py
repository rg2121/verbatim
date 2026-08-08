"""Answering a research question from retrieved passages, with citations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

from verbatim import config
from verbatim.citations import Citation, citation_from_chunk
from verbatim.corpus import ChunkMatch, CorpusStore

MODEL = config.HOSTED_MODEL
MAX_TOKENS = 16_000
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
ANSWER_LINE_PATTERN = re.compile(r"^ANSWER:\s*(.+?)(?=^CITE\b|\Z)", re.M | re.S)
CITE_LINE_PATTERN = re.compile(r"^CITE\s*\[(\d+)\]\s*(\S+)\s*::\s*(.+)$", re.M)
CHUNK_ID_PATTERN = re.compile(r"[0-9a-f]{16,}")
MARKER_PATTERN = re.compile(r"\[([^\[\]]{1,64})\]")

LINE_PROTOCOL = """
Reply in exactly this format and nothing else:

ANSWER: <one paragraph, using [1], [2] markers>
CITE [1] <chunk_id> :: <words copied character-for-character from that passage>
CITE [2] <chunk_id> :: <words copied character-for-character from that passage>

Emit no CITE lines if the passages do not answer the question.
"""

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
class RejectedCitation:
    """A citation whose quoted words are not in the passage it names."""

    chunk_id: str
    quoted_text: str
    reason: str


@dataclass(frozen=True)
class Answer:
    """A generated answer and the citations it rests on."""

    question: str
    text: str
    citations: list[Citation]
    rejected: list[RejectedCitation]

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "text": self.text,
            "citations": [citation.to_dict() for citation in self.citations],
            "rejected": [asdict(item) for item in self.rejected],
        }


def verify_quote(quoted_text: str, chunk_text: str) -> str | None:
    """Return why a quote fails verification, or None if it is genuine.

    Two presentation differences are forgiven, because neither changes which
    words were quoted: collapsed whitespace, and wrapping quotation marks or
    trailing sentence punctuation that a model adds when it sets a quote into
    its own prose. Measured against a local model, the wrapper alone accounted
    for a third of apparent failures.

    Nothing about the words themselves is forgiven. A paraphrase still fails,
    which is the whole point of the check.
    """
    quote = _normalize_quote(quoted_text)
    if not quote:
        return "empty quote"
    if quote not in " ".join(chunk_text.split()):
        return "quoted text does not appear in the cited passage"
    return None


def _normalize_quote(text: str) -> str:
    collapsed = " ".join(text.split()).strip()
    unwrapped = collapsed.strip("\"'“”‘’").strip()
    return unwrapped.rstrip(".,;:").strip()


class Completion(Protocol):
    """A model that returns JSON matching a schema."""

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]: ...


class OllamaCompletion:
    """A local model served by Ollama.

    Structure is elicited with a flat line protocol rather than Ollama's
    `format` parameter or tool calling. See `complete` for the measurements
    behind that choice; briefly, neither of the documented mechanisms survived
    a realistic multi-passage prompt on the model tested, and the line protocol
    parsed every time.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        host: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        # Set before model discovery, which makes a request.
        self.timeout = config.OLLAMA_TIMEOUT if timeout is None else timeout
        self.model = model or config.OLLAMA_MODEL or self._first_available_model()

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Elicit the answer as flat lines, then shape it like the JSON schema.

        `schema` is deliberately unused. Measured on this machine's model:
        tool calling produced the right structure on a one-passage prompt but
        was ignored on every realistic five-passage prompt, as were both
        `format: "json"` and a full JSON schema. The flat CITE-line protocol
        parsed correctly on every attempt, so the backend elicits that and
        converts it to the shape the caller expects.
        """
        body = self._post(
            "/api/chat",
            {
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": f"{system}\n{LINE_PROTOCOL}"},
                    {"role": "user", "content": user},
                ],
            },
        )
        content = body.get("message", {}).get("content", "")
        payload = parse_line_protocol(content)
        if payload is None:
            raise RuntimeError(
                f"Ollama model {self.model!r} did not answer in the expected "
                f"format. Response began: {content[:200]!r}"
            )
        return payload

    def _first_available_model(self) -> str:
        models = [item["name"] for item in self._get("/api/tags").get("models", [])]
        if not models:
            raise RuntimeError(
                "Ollama has no models pulled. Run 'ollama pull <model>' first."
            )
        return models[0]

    def _get(self, path: str) -> dict[str, Any]:
        return self._request(Request(f"{self.host}{path}"))

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        return self._request(request)

    def _request(self, request: Request) -> dict[str, Any]:
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - local host
                body = json.loads(response.read())
        except URLError as error:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host} ({error.reason}). "
                f"Is it running? Start it with 'ollama serve'."
            ) from error
        if "error" in body:
            raise RuntimeError(f"Ollama error: {body['error']}")
        return body


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


def parse_line_protocol(content: str) -> dict[str, Any] | None:
    """Read the flat ANSWER/CITE format back into the answer payload shape.

    Returns None when there is no ANSWER line at all, which is the signal that
    the model ignored the format rather than simply finding nothing to cite.
    """
    answer = ANSWER_LINE_PATTERN.search(content)
    if answer is None:
        return None
    citations = [
        {"marker": int(marker), "chunk_id": chunk_id, "quoted_text": quote.strip()}
        for marker, chunk_id, quote in CITE_LINE_PATTERN.findall(content)
    ]
    return {"answer": answer.group(1).strip(), "citations": citations}


def renumber_markers(
    text: str, citations: list[Citation], marker_positions: dict[int, int] | None = None
) -> str:
    """Rewrite inline markers so they match the rendered bibliography.

    Small models cite by chunk id, and their numbering does not survive a
    rejected citation: a model citing [1] and [2] where [1] failed leaves the
    prose pointing at a reference that is not printed. The bibliography is
    numbered by position, so the prose is rewritten to match, and any marker
    whose citation did not survive becomes [unverified].
    """
    positions = marker_positions or {}
    chunk_positions = {citation.chunk_id: n for n, citation in enumerate(citations, start=1)}

    def rewrite(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        if token in chunk_positions:
            return f"[{chunk_positions[token]}]"
        if token.isdigit() and int(token) in positions:
            return f"[{positions[int(token)]}]"
        if token.isdigit() or CHUNK_ID_PATTERN.fullmatch(token):
            # Points at a citation that did not survive verification. Say so
            # rather than leaving a dangling number or an opaque hash.
            return "[unverified]"
        return match.group(0)  # not a citation marker; leave it alone

    text = MARKER_PATTERN.sub(rewrite, text)
    # Collapse the runs this can produce, e.g. "[1] [1]" or "[1][1]".
    return re.sub(r"\[(\d+)\](?:\s*\[\1\])+", r"[\1]", text)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of prose, tolerating ``` fences."""
    match = JSON_OBJECT_PATTERN.search(text)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


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
        return Answer(
            question=question,
            text="No passages in the corpus match this question.",
            citations=[],
            rejected=[],
        )

    payload = completion.complete(SYSTEM_PROMPT, build_prompt(question, matches), ANSWER_SCHEMA)
    chunks = {match.chunk.chunk_id: match.chunk for match in matches}
    papers = corpus.papers_by_id(sorted({match.chunk.arxiv_id for match in matches}))

    citations: list[Citation] = []
    rejected: list[RejectedCitation] = []
    marker_positions: dict[int, int] = {}
    for raw in payload.get("citations", []):
        chunk_id = raw.get("chunk_id", "")
        quoted_text = raw.get("quoted_text", "")
        chunk = chunks.get(chunk_id)
        paper = papers.get(chunk.arxiv_id) if chunk else None
        if chunk is None or paper is None:
            rejected.append(
                RejectedCitation(chunk_id, quoted_text, "cited a passage that was not retrieved")
            )
            continue
        failure = verify_quote(quoted_text, chunk.text)
        if failure is not None:
            rejected.append(RejectedCitation(chunk_id, quoted_text, failure))
            continue
        duplicate = next(
            (n for n, existing in enumerate(citations, start=1) if existing.chunk_id == chunk_id),
            None,
        )
        if duplicate is not None:
            # A second quote from the same passage renders as an identical
            # reference, so the reader gains nothing from a duplicate entry -
            # but its marker must still resolve to the entry that is printed.
            if isinstance(raw.get("marker"), int):
                marker_positions[raw["marker"]] = duplicate
            continue
        citations.append(citation_from_chunk(chunk, paper, quoted_text))
        if isinstance(raw.get("marker"), int):
            marker_positions[raw["marker"]] = len(citations)
    return Answer(
        question=question,
        text=renumber_markers(payload.get("answer", ""), citations, marker_positions),
        citations=citations,
        rejected=rejected,
    )
