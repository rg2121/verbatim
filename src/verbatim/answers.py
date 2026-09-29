"""Answering a research question from retrieved passages, with citations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
import threading
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from verbatim import config
from verbatim.citations import Citation, citation_from_chunk
from verbatim.corpus import ChunkMatch, CorpusStore

ANSWER_LINE_PATTERN = re.compile(r"^ANSWER:\s*(.+?)(?=^CITE\b|\Z)", re.M | re.S)
CITE_LINE_PATTERN = re.compile(r"^CITE\s*\[(\d+)\]\s*(\S+)\s*::\s*(.+)$", re.M)
# The same line with the chunk id in the brackets and no number, as some models
# write it. The quote is verified just as strictly.
CITE_ID_LINE_PATTERN = re.compile(r"^CITE\s*\[([0-9a-f]{16,})\]\s*::\s*(.+)$", re.M)
CHUNK_ID_PATTERN = re.compile(r"[0-9a-f]{16,}")
MARKER_PATTERN = re.compile(r"\[([^\[\]]{1,64})\]")

LINE_PROTOCOL = """
Reply in exactly this format and nothing else:

ANSWER: <one paragraph, using [1], [2] markers>
CITE [1] <chunk_id> :: <words copied character-for-character from that passage>
CITE [2] <chunk_id> :: <words copied character-for-character from that passage>

If the passages are about a different subject, say so in ANSWER and emit no
CITE lines.
"""

SYSTEM_PROMPT = """You answer research questions strictly from the passages provided.

Rules:
- First decide whether the passages are actually about the question's subject.
  Search returns the closest passages it has, which for a subject the corpus
  does not cover are merely the least unrelated ones. Sharing a word is not
  being about the subject: a paper on tensor slicing in deep learning does not
  explain what a tensor is.
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
        "relevant": {
            "type": "boolean",
            "description": "Whether the passages are actually about the question's subject.",
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
    "required": ["relevant", "answer", "citations"],
    "additionalProperties": False,
}


@dataclass
class Usage:
    """Model calls and tokens spent, for the cost footer.

    Shared by report sections running in parallel, so updates take a lock.
    """

    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    _lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, body: dict[str, Any]) -> None:
        """Add one Ollama response's counters. Absent fields count as zero."""
        with self._lock:
            self.calls += 1
            self.prompt_tokens += int(body.get("prompt_eval_count") or 0)
            self.output_tokens += int(body.get("eval_count") or 0)
            # Ollama reports nanoseconds.
            self.seconds += float(body.get("total_duration") or 0) / 1e9

    def render(self) -> str:
        return (
            f"{self.calls} model call{'s' if self.calls != 1 else ''}, "
            f"{self.prompt_tokens:,} tokens in, {self.output_tokens:,} out, "
            f"{self.seconds:.1f}s in the model"
        )

    def snapshot(self) -> tuple[int, int, int, float]:
        """The counters as they stand, for measuring one step against."""
        with self._lock:
            return (self.calls, self.prompt_tokens, self.output_tokens, self.seconds)

    def since(self, before: tuple[int, int, int, float]) -> "Usage":
        """What has been spent since `before`, a snapshot."""
        calls, prompt, output, seconds = self.snapshot()
        return Usage(
            calls=calls - before[0], prompt_tokens=prompt - before[1],
            output_tokens=output - before[2], seconds=round(seconds - before[3], 3),
        )


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
    declined: bool = False

    @property
    def supporting_papers(self) -> int:
        """Distinct papers behind the answer.

        Reported but not warned on: measured, it did not separate good answers from bad.
        """
        return len({citation.arxiv_id for citation in self.citations})

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "text": self.text,
            "declined": self.declined,
            "supporting_papers": self.supporting_papers,
            "citations": [citation.to_dict() for citation in self.citations],
            "rejected": [asdict(item) for item in self.rejected],
        }


def verify_quote(quoted_text: str, chunk_text: str) -> str | None:
    """Return why a quote fails verification, or None if it is exact.

    Only whitespace, wrapping quotation marks and trailing punctuation are
    forgiven. Any change to the words fails.
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

    def judge(self, system: str, user: str) -> str:
        """Answer a short question with free text, for a yes/no decision."""
        ...


class OllamaToolCaller:
    """Picks a tool with Ollama's native tool calling, or None when none is chosen."""

    def __init__(self, model: str | None = None, *, host: str | None = None,
                 timeout: float | None = None) -> None:
        self._client = OllamaCompletion(model, host=host, timeout=timeout)

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def usage(self) -> "Usage":
        """This caller's model calls and tokens."""
        return self._client.usage

    def choose_tool(self, system, messages, tools):
        body = self._client._post(
            "/api/chat",
            {
                "model": self._client.model,
                "stream": False,
                "tools": [{"type": "function", "function": tool} for tool in tools],
                "messages": [{"role": "system", "content": system}, *messages],
            },
        )
        # A routing turn is a model call, so it is counted.
        self._client.usage.record(body)
        for call in body.get("message", {}).get("tool_calls") or []:
            function = call.get("function", {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if function.get("name"):
                return function["name"], (arguments or {})
        return None


class OllamaCompletion:
    """A model served by Ollama, local or `-cloud`.

    Answers use a flat line protocol, which small models follow more reliably
    than JSON output or tool calling (see `complete`).
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        host: str | None = None,
        timeout: float | None = None,
        protocol: str | None = None,
    ) -> None:
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        # Which structure the model is asked for. See `complete`.
        self.protocol = protocol or config.ANSWER_PROTOCOL
        # What this client has been asked to do, for the cost footer.
        self.usage = Usage()
        # Set before model discovery, which makes a request.
        self.timeout = config.OLLAMA_TIMEOUT if timeout is None else timeout
        self.model = model or config.OLLAMA_MODEL or self._first_available_model()

    def complete_text(self, system: str, user: str) -> str:
        """Ask for prose and return it unparsed."""
        body = self._post(
            "/api/chat",
            {
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        self.usage.record(body)
        return str(body.get("message", {}).get("content", "")).strip()

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Elicit the answer as ANSWER/CITE lines and return it in the schema's shape.

        `schema` is used only by the opt-in tools protocol. On a local model,
        tool calling and JSON output were ignored on realistic prompts; the line
        protocol always parsed.
        """
        if self.protocol == "tools":
            payload = self._complete_with_tools(system, user, schema)
            if payload is not None:
                return payload
            # The model ignored the tool: fall back to the line protocol.
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
        self.usage.record(body)
        content = body.get("message", {}).get("content", "")
        payload = parse_line_protocol(content)
        if payload is None:
            raise RuntimeError(
                f"Ollama model {self.model!r} did not answer in the expected "
                f"format. Response began: {content[:200]!r}"
            )
        return payload

    def _complete_with_tools(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Ask for the answer as a tool call, or return None if the model ignores it."""
        body = self._post(
            "/api/chat",
            {
                "model": self.model,
                "stream": False,
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "answer",
                        "description": "Answer the question with verified citations",
                        "parameters": schema,
                    },
                }],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        self.usage.record(body)
        for call in body.get("message", {}).get("tool_calls") or []:
            arguments = call.get("function", {}).get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    continue
            if isinstance(arguments, dict) and "answer" in arguments:
                arguments.setdefault("relevant", True)
                arguments.setdefault("citations", [])
                return arguments
        return None

    def judge(self, system: str, user: str) -> str:
        body = self._post(
            "/api/chat",
            {
                "model": self.model,
                "stream": False,
                "think": False,
                "options": {"num_predict": 8, "temperature": 0},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        self.usage.record(body)
        return body.get("message", {}).get("content", "")

    def stream(self, system: str, user: str):
        """Yield the reply's text as the model writes it.

        The caller parses and verifies the finished reply exactly as `complete`
        does. Failures are retried only before the first line arrives, since a
        retry after that would repeat text already shown.
        """
        request = Request(
            f"{self.host}/api/chat",
            data=json.dumps(
                {
                    "model": self.model,
                    "stream": True,
                    "messages": [
                        {"role": "system", "content": f"{system}\n{LINE_PROTOCOL}"},
                        {"role": "user", "content": user},
                    ],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        attempts = config.HTTP_RETRIES
        last: Exception | None = None
        for attempt in range(attempts + 1):
            if attempt:
                time.sleep(config.HTTP_BACKOFF * (2 ** (attempt - 1)))
            try:
                response = urlopen(request, timeout=self.timeout)  # noqa: S310 - local host
            except HTTPError as error:
                final = self._refusal(error)
                if final is not None:
                    raise final from error
                last = error
                continue
            except (URLError, TimeoutError, OSError) as error:
                last = error
                continue
            with response:
                for line in response:
                    if not line.strip():
                        continue
                    try:
                        body = json.loads(line)
                    except ValueError as error:
                        raise RuntimeError(
                            f"Ollama sent a line that is not JSON: {line[:120]!r}"
                        ) from error
                    if "error" in body:
                        raise RuntimeError(f"Ollama error: {body['error']}")
                    fragment = body.get("message", {}).get("content", "")
                    if fragment:
                        yield fragment
                    if body.get("done"):
                        self.usage.record(body)
                        return
            return
        raise self._unreachable(attempts, last)

    def _first_available_model(self) -> str:
        """The first local model pulled. A `-cloud` model is used only when named."""
        models = [item["name"] for item in self._get("/api/tags").get("models", [])]
        if not models:
            raise RuntimeError(
                "Ollama has no models pulled. Run 'ollama pull <model>' first."
            )
        local = [name for name in models if not name.endswith("-cloud")]
        if not local:
            raise RuntimeError(
                "Only cloud models are pulled, and a cloud model sends the passages "
                "to Ollama's servers. Pull a local model, or set OLLAMA_MODEL to one "
                f"of: {', '.join(models)}."
            )
        return local[0]

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
        """One call to Ollama, retrying server errors and dropped connections but never a 4xx."""
        attempts = config.HTTP_RETRIES
        last: Exception | None = None
        for attempt in range(attempts + 1):
            if attempt:
                time.sleep(config.HTTP_BACKOFF * (2 ** (attempt - 1)))
            try:
                with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                    body = json.loads(response.read())
            except HTTPError as error:
                final = self._refusal(error)
                if final is not None:
                    raise final from error
                last = error
                continue
            except (URLError, TimeoutError, OSError) as error:
                last = error
                continue
            if "error" in body:
                raise RuntimeError(f"Ollama error: {body['error']}")
            return body
        raise self._unreachable(attempts, last)

    def _refusal(self, error: HTTPError) -> RuntimeError | None:
        """The error to stop on, or None when a retry could fix it."""
        if error.code == 404:
            # Ollama answers 404 when the model is not pulled: name it and list what is.
            return RuntimeError(self._missing_model_message())
        if error.code < 500:
            return RuntimeError(
                f"Ollama at {self.host} refused the request "
                f"({error.code} {error.reason})."
            )
        return None

    def _unreachable(self, attempts: int, last: Exception | None) -> RuntimeError:
        return RuntimeError(
            f"Cannot reach Ollama at {self.host} after {attempts + 1} attempts "
            f"({last}). Is it running? Start it with 'ollama serve'."
        )

    def _missing_model_message(self) -> str:
        """Name the missing model and, if listing succeeds, the models that are pulled."""
        message = (
            f"Ollama at {self.host} has no model named {self.model!r}. "
            f"Pull it with 'ollama pull {self.model}'"
        )
        try:
            available = [
                item["name"] for item in self._get("/api/tags").get("models", [])
            ]
        except Exception:  # noqa: BLE001 - a listing failure must not mask this
            return message + "."
        if not available:
            return message + ", or pull any model - none are installed."
        return message + f", or use one of: {', '.join(sorted(available))}."


def parse_line_protocol(content: str) -> dict[str, Any] | None:
    """Parse ANSWER/CITE lines into the answer payload.

    Returns None when there is no ANSWER line, meaning the model ignored the format.
    """
    answer = ANSWER_LINE_PATTERN.search(content)
    if answer is None:
        return None
    citations = [
        {"marker": int(marker), "chunk_id": chunk_id, "quoted_text": quote.strip()}
        for marker, chunk_id, quote in CITE_LINE_PATTERN.findall(content)
    ]
    citations.extend(
        # Numbered after the other form; this form's prose cites by chunk id.
        {"marker": position, "chunk_id": chunk_id, "quoted_text": quote.strip()}
        for position, (chunk_id, quote) in enumerate(
            CITE_ID_LINE_PATTERN.findall(content), start=len(citations) + 1
        )
    )
    return {
        "answer": answer.group(1).strip(),
        "citations": citations,
    }


def renumber_markers(
    text: str, citations: list[Citation], marker_positions: dict[int, int] | None = None
) -> str:
    """Renumber inline markers to match the printed bibliography.

    Markers may be numbers or chunk ids. One whose citation was rejected
    becomes [unverified].
    """
    positions = marker_positions or {}
    chunk_positions = {citation.chunk_id: n for n, citation in enumerate(citations, start=1)}

    def rewrite_one(token: str) -> str | None:
        if token in chunk_positions:
            return f"[{chunk_positions[token]}]"
        if token.isdigit() and int(token) in positions:
            return f"[{positions[int(token)]}]"
        if token.isdigit() or CHUNK_ID_PATTERN.fullmatch(token):
            # Its citation was rejected.
            return "[unverified]"
        return None  # not a citation marker

    def is_marker(token: str) -> bool:
        return bool(token.isdigit() or CHUNK_ID_PATTERN.fullmatch(token))

    def rewrite(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        rewritten = rewrite_one(token)
        if rewritten is not None:
            return rewritten
        # Grouped markers such as "[5, 3]" are rewritten only when every part is a
        # marker, so text like "[see also, page 3]" is left alone.
        grouped = [part.strip() for part in token.split(",")]
        if len(grouped) > 1 and all(is_marker(part) for part in grouped):
            return "".join(rewrite_one(part) or "[unverified]" for part in grouped)
        return match.group(0)  # leave anything else alone

    text = MARKER_PATTERN.sub(rewrite, text)
    # Collapse the runs this can produce, e.g. "[1] [1]" or "[1][1]".
    return re.sub(r"\[(\d+)\](?:\s*\[\1\])+", r"[\1]", text)


SUBJECT_SYSTEM = (
    "Name the single subject these passages are about, in at most six words. "
    "Describe only what they discuss. Do not mention the question."
)


def describe_subject(
    completion: Completion, matches: list[ChunkMatch], *, sample: int = 5
) -> str:
    """Ask what the passages are about, without showing the question.

    Asked whether passages are relevant, models say yes; asked what they
    discuss, they answer accurately.
    """
    passages = "\n\n".join(f"<passage>{m.chunk.text[:500]}</passage>" for m in matches[:sample])
    reply = completion.judge(SUBJECT_SYSTEM, f"{passages}\n\nSubject:")
    return " ".join((reply or "").split())[:120]


def subject_similarity(embedder: Any, question: str, subject: str) -> float:
    """Compare question and subject arithmetically, so no model judges the match."""
    a, b = embedder.embed_query(question), embedder.embed_query(subject)
    return sum(x * y for x, y in zip(a, b, strict=True))


def subject_check(
    completion: Completion,
    embedder: Any,
    question: str,
    matches: list[ChunkMatch],
    *,
    sample: int = 5,
) -> tuple[str | None, float | None]:
    """What the passages are about, and its similarity to the question.

    Returns (None, None) when no subject is named, so a check that did not run
    stays silent.
    """
    subject = describe_subject(completion, matches, sample=sample)
    if not subject:
        return None, None
    return subject, subject_similarity(embedder, question, subject)


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
            declined=True,
        )

    payload = completion.complete(SYSTEM_PROMPT, build_prompt(question, matches), ANSWER_SCHEMA)
    papers = corpus.papers_by_id(sorted({match.chunk.arxiv_id for match in matches}))
    return answer_from_payload(question, matches, payload, papers)


def answer_from_payload(
    question: str,
    matches: list[ChunkMatch],
    payload: dict[str, Any],
    papers: dict[str, Any],
) -> Answer:
    """Verify a model's payload into an Answer.

    Streamed and non-streamed answers both finish here, so they are verified
    the same way. `papers` is passed in so this can run off the database thread.
    """
    chunks = {match.chunk.chunk_id: match.chunk for match in matches}

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
            # A second quote from one passage adds no reference; its marker points
            # at the existing one.
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
