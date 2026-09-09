"""Answering a research question from retrieved passages, with citations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
import threading
from typing import Any, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

from verbatim import config
from verbatim.citations import Citation, citation_from_chunk
from verbatim.corpus import ChunkMatch, CorpusStore

JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
ANSWER_LINE_PATTERN = re.compile(r"^ANSWER:\s*(.+?)(?=^CITE\b|\Z)", re.M | re.S)
RELEVANT_LINE_PATTERN = re.compile(r"^RELEVANT:\s*(yes|no)\b", re.M | re.I)
VERDICT_PATTERN = re.compile(r"\b(yes|no)\b", re.I)

RELEVANCE_SYSTEM = """You judge one thing only: are these passages about the \
subject the question asks about?

You are not answering the question. Do not try.

Search returns its closest passages even when the corpus contains nothing on the
subject, so the passages you see may simply be the least unrelated ones. Sharing
a word is not being about a subject:
- passages on tensor slicing in deep learning are NOT about what a tensor is
- passages on AI in peer review are NOT about AI in cosmology
- passages on detectable neutron stars are NOT about the farthest star

Reply with exactly one word: yes or no."""
CITE_LINE_PATTERN = re.compile(r"^CITE\s*\[(\d+)\]\s*(\S+)\s*::\s*(.+)$", re.M)
# The same line with the chunk id inside the brackets and no number:
# `CITE [dc83e954...] :: quoted words`. gemma4:31b writes it this way every
# time, and reading only the numbered form threw away four genuine citations
# per answer and marked the prose [unverified]. Accepting it verifies exactly
# as strictly - the quote is still checked character-for-character against the
# passage it names - and only the bookkeeping differs, because the inline
# markers are chunk ids too and renumber_markers already resolves those.
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
    """What the model was asked to do, counted.

    The orchestrator's whole justification is a cost/benefit argument - a
    ladder that retries costs generations - and the argument was being made
    without numbers. Ollama returns them on every call and they were being
    thrown away.

    Mutable and shared: report sections run in parallel threads against one
    client, so the increments are taken under a lock.
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
        """What has been spent since a snapshot. A run outlives a step, so a
        step's cost is a difference rather than a total."""
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

        Reported in to_dict for callers that want it, but deliberately *not*
        warned on: measured, a lone-paper warning fired on 5 of 14 good answers
        while catching only 2 of 6 bad ones, and caught nothing the subject
        check did not already catch. Corroboration reads like a quality signal
        and is not one here.
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

    def judge(self, system: str, user: str) -> str:
        """Answer a short question with free text, for a yes/no decision."""
        ...


class OllamaToolCaller:
    """Picks a tool with Ollama's tool calling.

    Tool calling was measured unreliable on long prompts on this model, but an
    agent's routing turn is short - a question plus the tool schemas - and it
    chose correctly for recency, single-paper and literature questions alike.
    Returning None when no tool is chosen lets the loop stop rather than force
    a call the model did not want to make.
    """

    def __init__(self, model: str | None = None, *, host: str | None = None,
                 timeout: float | None = None) -> None:
        self._client = OllamaCompletion(model, host=host, timeout=timeout)

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def usage(self) -> "Usage":
        """This caller's own counters.

        Routing turns are model calls too, and they were counted nowhere:
        `report --agents` climbs to a rung that calls this, so the path whose
        cost most needs reporting was the one under-reporting it.
        """
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
        # Counted here rather than in _post, which also serves calls that are
        # not generations: a routing turn is a model call and was reporting as
        # none at all.
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
        # What this client has been asked to do, for the cost footer.
        self.usage = Usage()
        # Set before model discovery, which makes a request.
        self.timeout = config.OLLAMA_TIMEOUT if timeout is None else timeout
        self.model = model or config.OLLAMA_MODEL or self._first_available_model()

    def complete_text(self, system: str, user: str) -> str:
        """Ask for prose and return it unparsed.

        Answering needs a structure and gets it through the line protocol.
        Query rewriting wants sentences back, so it asks plainly rather than
        being made to fit a protocol it has no use for.
        """
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
        self.usage.record(body)
        content = body.get("message", {}).get("content", "")
        payload = parse_line_protocol(content)
        if payload is None:
            raise RuntimeError(
                f"Ollama model {self.model!r} did not answer in the expected "
                f"format. Response began: {content[:200]!r}"
            )
        return payload

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

        The same call `complete` makes, with `stream: true`; Ollama then sends
        one JSON object per line, each carrying a fragment. The caller joins
        what it saw and parses it with parse_line_protocol, so streaming
        changes when text is shown and nothing about what is accepted:
        verification runs on the finished reply exactly as it always has.
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
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - local host
                for line in response:
                    if not line.strip():
                        continue
                    body = json.loads(line)
                    if "error" in body:
                        raise RuntimeError(f"Ollama error: {body['error']}")
                    fragment = body.get("message", {}).get("content", "")
                    if fragment:
                        yield fragment
                    if body.get("done"):
                        self.usage.record(body)
                        return
        except URLError as error:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host} ({error.reason}). "
                f"Is it running? Start it with 'ollama serve'."
            ) from error

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
    citations.extend(
        # Numbered from where the other form left off; the number is only a
        # fallback for prose that cites numerically, and this form's prose
        # cites by chunk id.
        {"marker": position, "chunk_id": chunk_id, "quoted_text": quote.strip()}
        for position, (chunk_id, quote) in enumerate(
            CITE_ID_LINE_PATTERN.findall(content), start=len(citations) + 1
        )
    )
    verdict = RELEVANT_LINE_PATTERN.search(content)
    return {
        # Absent verdict is treated as relevant, so a model that ignores the
        # line behaves as it did before rather than falling silent.
        "relevant": verdict is None or verdict.group(1).lower() == "yes",
        "answer": answer.group(1).strip(),
        "citations": citations,
    }


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

    def rewrite_one(token: str) -> str | None:
        if token in chunk_positions:
            return f"[{chunk_positions[token]}]"
        if token.isdigit() and int(token) in positions:
            return f"[{positions[int(token)]}]"
        if token.isdigit() or CHUNK_ID_PATTERN.fullmatch(token):
            # Points at a citation that did not survive verification. Say so
            # rather than leaving a dangling number or an opaque hash.
            return "[unverified]"
        return None  # not a citation marker

    def rewrite(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        rewritten = rewrite_one(token)
        if rewritten is not None:
            return rewritten
        # Models also group markers, "[5, 3]", which must not survive as a
        # stale pair of numbers when each member was renumbered.
        grouped = [part.strip() for part in token.split(",")]
        if len(grouped) > 1 and all(part.isdigit() for part in grouped):
            return "".join(rewrite_one(part) or "[unverified]" for part in grouped)
        return match.group(0)  # leave anything else alone

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


SUBJECT_SYSTEM = (
    "Name the single subject these passages are about, in at most six words. "
    "Describe only what they discuss. Do not mention the question."
)


def describe_subject(
    completion: Completion, matches: list[ChunkMatch], *, sample: int = 5
) -> str:
    """Ask what the passages are about, without mentioning the question.

    Asked whether passages are relevant, every model tested says yes: they were
    retrieved because they matched, and the model reads that as evidence. Asked
    only what they discuss, the same model answers accurately - it called the
    passages behind "what is a tensor?" multidimensional arrays, and those
    behind "the farthest star" neutron star properties.
    """
    passages = "\n\n".join(f"<passage>{m.chunk.text[:500]}</passage>" for m in matches[:sample])
    reply = completion.judge(SUBJECT_SYSTEM, f"{passages}\n\nSubject:")
    return " ".join((reply or "").split())[:120]


def subject_similarity(embedder: Any, question: str, subject: str) -> float:
    """Compare question and subject arithmetically, so no model judges the match."""
    a, b = embedder.embed_query(question), embedder.embed_query(subject)
    return sum(x * y for x, y in zip(a, b))


def judge_relevance(
    completion: Completion, question: str, matches: list[ChunkMatch]
) -> bool | None:
    """Ask, in a call of its own, whether the passages are on-subject.

    **Not wired into answering, because it does not work on the local model.**
    Measured over twenty questions it answered "yes" eighteen times, catching
    two of six unanswerable ones - the same as no check at all, at the cost of
    an extra call. An inline verdict in the answering call did worse still,
    catching one of six and refusing one real question, because a call asked to
    both judge and answer lets the answering instinct win.

    Kept because the remaining hypothesis is that this is a small-model limit
    rather than a prompt-design one, and testing that needs only a stronger
    model behind the same function.

    Returns None when the model gives no usable verdict, so a caller can carry
    on rather than decline on a parsing failure.
    """
    passages = "\n\n".join(
        f"<passage>{m.chunk.text[:600]}</passage>" for m in matches[:6]
    )
    reply = completion.judge(
        RELEVANCE_SYSTEM, f"Question: {question}\n\n{passages}\n\nOn subject? yes or no:"
    )
    verdict = VERDICT_PATTERN.search(reply or "")
    return None if verdict is None else verdict.group(1).lower() == "yes"


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
    """Verify a model's payload into an Answer, however the payload arrived.

    Split from answer_from_matches so a streamed generation - whose deltas
    were already shown to someone - finishes through exactly this same
    verification, rather than a second copy of it. `papers` is passed in
    rather than looked up so the caller can run this off the database thread.
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
