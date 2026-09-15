"""The HTTP front end.

Retrieval itself is tested elsewhere; what matters here is that a result
carries everything needed to check it, and that the page and the API are
usable without a network.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from helpers import chunked_corpus
from verbatim.web import PAGE, create_app


class StreamingScript:
    """A scripted answering model with the streaming surface Ollama has."""

    def __init__(self, fragments: list[str]) -> None:
        self.fragments = fragments

    def stream(self, system, user):
        yield from self.fragments

    def judge(self, system, user):
        return "superconducting qubit decoherence"


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


class StubEncoder:
    """Enough embedder for the ask flow, with no ONNX weights to load."""

    # The real embedder names its vectors without loading them, and the app
    # relies on that for /api/stats and /api/health.
    model_id = "stub"

    def __init__(self) -> None:
        from verbatim.corpus import EmbeddingModel

        self.model = EmbeddingModel(
            model_id="stub", dimensions=3, normalized=True, revision="v0"
        )

    def embed_passages(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def ask_client(tmp_path: Path, fragments: list[str]) -> tuple[TestClient, str]:
    with chunked_corpus(
        tmp_path, {"2608.11111v1": ["Superconducting qubits decohere quickly."]}
    ) as store:
        chunk_id = store.all_chunks()[0].chunk_id
    app = create_app(
        tmp_path / "corpus.db",
        completion=StreamingScript(fragments),
        encoder=StubEncoder(),
    )
    return TestClient(app), chunk_id


def client(tmp_path: Path) -> TestClient:
    with chunked_corpus(
        tmp_path, {"2608.11111v1": ["Superconducting qubits decohere quickly."]}
    ):
        pass
    return TestClient(create_app(tmp_path / "corpus.db"))


def test_ask_streams_a_draft_and_finishes_with_a_verified_answer(tmp_path: Path) -> None:
    fragments = ["ANSWER: They decohere quickly [1].\n", ""]
    client_app, chunk_id = ask_client(tmp_path, fragments)
    # The script cites the passage that really was retrieved, quoting exactly.
    fragments[1] = f"CITE [1] {chunk_id} :: Superconducting qubits decohere quickly."

    events = sse_events(client_app.get("/api/ask", params={"q": "qubits"}).text)

    names = [name for name, _ in events]
    assert names[0] == "passages" and names.count("delta") == 2 and names[-1] == "done"
    answer = next(payload for name, payload in events if name == "answer")
    assert answer["text"] == "They decohere quickly [1]."
    assert [c["arxiv_id"] for c in answer["citations"]] == ["2608.11111v1"]
    assert answer["rejected"] == []
    subject = next(payload for name, payload in events if name == "subject")
    assert subject["subject"] == "superconducting qubit decoherence"
    assert "warn" in subject and "closeness" in subject


def test_ask_rejects_a_quote_the_passage_does_not_contain(tmp_path: Path) -> None:
    fragments = ["ANSWER: Made up [1].\n", ""]
    client_app, chunk_id = ask_client(tmp_path, fragments)
    fragments[1] = f"CITE [1] {chunk_id} :: Qubits never decohere at all."

    events = sse_events(client_app.get("/api/ask", params={"q": "qubits"}).text)

    answer = next(payload for name, payload in events if name == "answer")
    assert answer["citations"] == []
    assert len(answer["rejected"]) == 1
    assert "[unverified]" in answer["text"]


def test_ask_declines_when_nothing_is_retrieved(tmp_path: Path) -> None:
    client_app, _ = ask_client(tmp_path, ["ANSWER: should never be asked\n"])

    events = sse_events(
        client_app.get("/api/ask", params={"q": "photosynthesis chlorophyll"}).text
    )

    names = [name for name, _ in events]
    assert "delta" not in names  # no model call for an empty retrieval
    answer = next(payload for name, payload in events if name == "answer")
    assert answer["declined"] is True


def test_ask_reports_a_model_that_ignored_the_format(tmp_path: Path) -> None:
    client_app, _ = ask_client(tmp_path, ["I refuse to follow formats."])

    events = sse_events(client_app.get("/api/ask", params={"q": "qubits"}).text)

    failed = next(payload for name, payload in events if name == "failed")
    assert "format" in failed["message"]
    assert [name for name, _ in events][-1] == "done"


def test_a_search_result_can_be_traced_back_to_its_paper(tmp_path: Path) -> None:
    response = client(tmp_path).get("/api/search", params={"q": "qubits", "mode": "keyword"})

    assert response.status_code == 200
    (hit,) = response.json()["results"]
    assert hit["arxiv_id"] == "2608.11111v1"
    assert hit["locator"] == "p. 1"
    assert hit["source_url"].endswith("2608.11111v1")
    assert "license_url" in hit and "chunk_id" in hit


def test_an_unknown_paper_is_a_404_rather_than_an_empty_answer(tmp_path: Path) -> None:
    response = client(tmp_path).get("/api/papers/9999.99999v1")

    assert response.status_code == 404
    assert "9999.99999v1" in response.json()["detail"]


def test_a_known_paper_is_returned_whole(tmp_path: Path) -> None:
    response = client(tmp_path).get("/api/papers/2608.11111v1")

    assert response.status_code == 200
    assert response.json()["arxiv_id"] == "2608.11111v1"


def test_an_empty_query_is_refused_rather_than_scanning_everything(tmp_path: Path) -> None:
    assert client(tmp_path).get("/api/search", params={"q": ""}).status_code == 422


def test_the_limit_is_capped(tmp_path: Path) -> None:
    assert client(tmp_path).get(
        "/api/search", params={"q": "qubits", "limit": 5000}
    ).status_code == 422


def test_an_unknown_mode_is_refused(tmp_path: Path) -> None:
    assert client(tmp_path).get(
        "/api/search", params={"q": "qubits", "mode": "magic"}
    ).status_code == 422


def test_stats_are_served_for_the_page_footer(tmp_path: Path) -> None:
    counts = client(tmp_path).get("/api/stats").json()

    assert counts["papers"] == 1 and counts["chunks"] >= 1


def test_the_page_is_served_and_reaches_no_external_host(tmp_path: Path) -> None:
    """The pipeline runs offline once the corpus is built; the page must too."""
    response = client(tmp_path).get("/")

    assert response.status_code == 200
    assert "verbatim" in response.text
    for scheme in ("https://", "http://"):
        assert scheme not in PAGE.split("<script>")[0].replace(
            'xmlns="http://www.w3.org/1999/xhtml"', ""
        )


def test_a_follow_up_is_resolved_before_retrieval_and_never_sent_to_the_answer(
    tmp_path,
) -> None:
    """History may decide what is looked up. It may not decide what is claimed.

    Prose from an earlier turn cannot be verified against any paper, so letting
    it reach the answering model would let a claim enter without ever having
    come from one.
    """
    import json

    from fastapi.testclient import TestClient

    from helpers import chunked_corpus
    from verbatim.web import create_app

    with chunked_corpus(tmp_path, {"2608.11111v1": ["PPO clips its objective ratio."]}):
        pass  # written and closed; the app opens its own connection

    seen: list[str] = []

    class Recording(StreamingScript):
        """Records every prompt, so the test can assert what the model saw."""

        def complete_text(self, system: str, user: str) -> str:
            seen.append(user)
            return "Why does PPO clip its objective?"

        def stream(self, system, user):
            seen.append(user)
            yield "ANSWER: It clips to bound the update.\n"

    app = create_app(
        tmp_path / "corpus.db", completion=Recording([]), encoder=StubEncoder()
    )
    history = json.dumps([{"question": "what is PPO?", "answer": "A policy method."}])
    body = TestClient(app).get(
        "/api/ask", params={"q": "why does it clip?", "history": history}
    ).text

    assert "event: resolved" in body
    assert "Why does PPO clip its objective?" in body
    # The rewrite call saw the history; the answering call never did.
    assert any("what is PPO?" in prompt for prompt in seen[:1])
    assert not any("A policy method." in prompt for prompt in seen[1:])


def test_malformed_history_gives_a_single_turn_answer_rather_than_an_error(
    tmp_path,
) -> None:
    from fastapi.testclient import TestClient

    from helpers import chunked_corpus
    from verbatim.web import create_app

    with chunked_corpus(tmp_path, {"2608.11111v1": ["PPO clips its objective."]}):
        pass  # written and closed; the app opens its own connection

    class Stub(StreamingScript):
        def complete_text(self, system: str, user: str) -> str:
            raise AssertionError("malformed history must not trigger a rewrite")

    app = create_app(
        tmp_path / "corpus.db",
        completion=Stub(["ANSWER: An answer.\n"]),
        encoder=StubEncoder(),
    )
    body = TestClient(app).get(
        "/api/ask", params={"q": "what is PPO?", "history": "{not json"}
    ).text

    assert "event: resolved" not in body


def test_requests_served_from_different_threads_each_work(tmp_path: Path) -> None:
    """FastAPI serves sync endpoints from a pool, so a shared SQLite
    connection fails as soon as a second thread is used. It presented as
    "Connection lost while answering" in the browser, intermittently.
    """
    from concurrent.futures import ThreadPoolExecutor

    from helpers import chunked_corpus
    from verbatim.web import create_app

    with chunked_corpus(tmp_path, {"2608.11111v1": ["Dark matter haloes dominate."]}):
        pass

    app = create_app(
        tmp_path / "corpus.db",
        completion=StreamingScript(["ANSWER: Something.\n"]),
        encoder=StubEncoder(),
    )
    client = TestClient(app)

    def call(_: int) -> int:
        return client.get("/api/search", params={"q": "dark matter"}).status_code

    # Enough calls to be served from more than one pool thread.
    with ThreadPoolExecutor(max_workers=4) as pool:
        codes = list(pool.map(call, range(12)))

    assert codes == [200] * 12


def test_the_page_javascript_parses() -> None:
    """A broken escape once turned a regex literal into a syntax error, which
    killed the whole script block and left the page inert - every button dead,
    with nothing in the server log because the failure was in the browser.

    Checked with node when it is available, and by direct assertion when it is
    not, so CI catches it either way.
    """
    import re
    import shutil
    import subprocess

    from verbatim.web import PAGE

    script = "\n".join(re.findall(r"<script>(.*?)</script>", PAGE, re.S))
    assert script.strip()

    # The escape that broke: `\n` inside the CITE regex must reach the browser
    # as two characters, not as an actual newline splitting the literal.
    assert "/\\n\\s*CITE" in script
    assert not re.search(r"split\(/\s*$", script, re.M)

    node = shutil.which("node")
    if node:
        checked = subprocess.run(
            [node, "--check", "-"], input=script, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stderr


def test_every_element_the_script_addresses_exists_in_the_markup() -> None:
    """A rename that misses one leaves a dead control and no error anywhere."""
    import re

    from verbatim.web import PAGE

    script = "\n".join(re.findall(r"<script>(.*?)</script>", PAGE, re.S))
    markup = re.sub(r"<script>.*?</script>", "", PAGE, flags=re.S)
    for element in sorted(set(re.findall(r'\$\("([a-zA-Z0-9_-]+)"\)', script))):
        assert f'id="{element}"' in markup, f"script uses #{element}, markup has no such id"


class ReportScript:
    """A model that plans two sections and cites the one passage that exists."""

    def __init__(self, chunk_id: str, fail_on: str | None = None) -> None:
        self.chunk_id = chunk_id
        self.fail_on = fail_on

    def complete_text(self, system, user):
        # The fixture corpus has no vectors, so only the keyword leg can
        # retrieve: the headings have to share a word with the passage.
        return "What are superconducting qubits?\nHow quickly do qubits decohere?"

    def complete(self, system, user, schema):
        if self.fail_on and self.fail_on in user:
            raise RuntimeError("Cannot reach Ollama at http://localhost:11434")
        return {
            "relevant": True,
            "answer": "They decohere quickly [1].",
            "citations": [{
                "marker": 1,
                "chunk_id": self.chunk_id,
                "quoted_text": "Superconducting qubits decohere quickly.",
            }],
        }

    def judge(self, system, user):
        return "superconducting qubit decoherence"


def report_client(tmp_path: Path, fail_on: str | None = None) -> TestClient:
    with chunked_corpus(
        tmp_path, {"2608.11111v1": ["Superconducting qubits decohere quickly."]}
    ) as store:
        chunk_id = store.all_chunks()[0].chunk_id
    return TestClient(create_app(
        tmp_path / "corpus.db",
        completion=ReportScript(chunk_id, fail_on),
        encoder=StubEncoder(),
    ))


def test_a_review_streams_each_section_as_it_lands(tmp_path: Path) -> None:
    events = sse_events(
        report_client(tmp_path).get("/api/report", params={"topic": "decoherence"}).text
    )

    names = [name for name, _ in events]
    assert names.count("working") == 2 and names.count("section") == 2
    # Sections arrive before the assembled review, which is the whole point.
    assert names.index("section") < names.index("report")
    assert names[-1] == "done"

    first = next(payload for name, payload in events if name == "section")
    assert first["heading"] == "What are superconducting qubits?"
    assert [c["arxiv_id"] for c in first["citations"]] == ["2608.11111v1"]
    assert first["evidence"] and "Answered from" in first["evidence"][0]
    # The trace the CLI ledger shows travels to the page too.
    assert first["attempts"][0]["strategy"] == "hybrid"
    assert first["shape"] == "definitional"

    assembled = next(payload for name, payload in events if name == "report")
    assert "## What are superconducting qubits?" in assembled["markdown"]
    assert "How this review was made" in assembled["ledger_markdown"]


def test_a_section_that_fails_does_not_abort_the_stream(tmp_path: Path) -> None:
    events = sse_events(
        report_client(tmp_path, fail_on="How quickly").get(
            "/api/report", params={"topic": "decoherence"}
        ).text
    )

    sections = [payload for name, payload in events if name == "section"]
    assert len(sections) == 2
    assert not sections[0]["declined"] and sections[1]["declined"]
    assert sections[1]["outcome"].startswith("failed")
    assert [name for name, _ in events][-1] == "done"


def test_the_review_endpoint_refuses_an_unbounded_request(tmp_path: Path) -> None:
    client_app = report_client(tmp_path)

    assert client_app.get("/api/report", params={"topic": ""}).status_code == 422
    assert client_app.get(
        "/api/report", params={"topic": "x", "sections": 50}
    ).status_code == 422


def test_an_injected_model_is_never_supplemented_by_a_real_one(tmp_path: Path) -> None:
    """The suite must not reach Ollama, and the route needs a tool caller.

    Building one behind an injected model's back made the suite take a minute
    and would have made it need a running Ollama.
    """
    with chunked_corpus(
        tmp_path, {"2608.22222v1": ["Superconducting qubits decohere quickly."]}
    ) as store:
        chunk_id = store.all_chunks()[0].chunk_id
    app = create_app(
        tmp_path / "corpus.db",
        completion=ReportScript(chunk_id),
        encoder=StubEncoder(),
    )

    # Reaching a real Ollama would make this take a minute instead of a
    # second; the suite's speed is the assertion.
    from time import perf_counter

    started = perf_counter()
    with TestClient(app) as client_app:
        events = sse_events(
            client_app.get("/api/report", params={"topic": "x", "sections": 1}).text
        )

    assert perf_counter() - started < 5
    # The scripted model cannot choose tools, so the agent rung stays unarmed
    # rather than an Ollama client being built behind it.
    section = next(payload for name, payload in events if name == "section")
    agent_rungs = [a for a in section["attempts"] if a["strategy"] == "agent"]
    assert all(a["route"] == "no tool-choosing model available" for a in agent_rungs)


def test_the_review_cost_is_this_review_not_the_servers_lifetime() -> None:
    from verbatim.answers import Usage
    from verbatim.web import _cost_since, _usage_totals

    class Counted:
        def __init__(self) -> None:
            self.usage = Usage()

    model, caller = Counted(), Counted()
    model.usage.record({"prompt_eval_count": 900, "eval_count": 100, "total_duration": 9_000_000_000})
    before = _usage_totals(model, caller)
    # This review: one answering call and one routing call.
    model.usage.record({"prompt_eval_count": 10, "eval_count": 4, "total_duration": 1_000_000_000})
    caller.usage.record({"prompt_eval_count": 5, "eval_count": 1, "total_duration": 500_000_000})

    assert _cost_since(before, model, caller) == (
        "2 model calls, 15 tokens in, 5 out, 1.5s in the model"
    )
    assert _cost_since(None, model) == "cost not reported"


def test_health_reports_the_corpus_and_the_last_run_of_each_command(tmp_path: Path) -> None:
    """A served page and an empty or half-ingested corpus look identical."""
    from verbatim.corpus import CorpusStore
    from verbatim.runs import record_run

    with chunked_corpus(tmp_path, {"2608.11111v1": ["Superconducting qubits decohere."]}):
        pass
    with CorpusStore(tmp_path / "corpus.db") as store:
        with record_run(store, "ask") as run:
            run.record(citations=3, rejected=1)
        with pytest.raises(RuntimeError):
            with record_run(store, "embed"):
                raise RuntimeError("Ollama unreachable")

    body = TestClient(
        create_app(tmp_path / "corpus.db", encoder=StubEncoder())
    ).get("/api/health").json()

    assert body["ok"] is False          # the last embed failed
    assert body["failing"] == ["embed"]
    assert body["corpus"]["papers"] == 1 and body["corpus"]["passages"] >= 1
    assert body["last_run"]["embed"]["ok"] is False
    assert "Ollama unreachable" in body["last_run"]["embed"]["error"]
    # The rejection rate is read from the ledger, not recomputed.
    assert body["answers"] == {
        "runs": 1, "citations_verified": 3, "citations_rejected": 1,
        "rejection_rate": 0.25,
    }


class LadderScript:
    """A model that fails one rung and answers on the next.

    `complete` is what the ladder calls; the first call quotes something the
    passage does not contain, so verification rejects it, no citation survives
    and the verdict is retry. The second quotes exactly, and is accepted.
    """

    def __init__(self, chunk_id: str) -> None:
        self.chunk_id = chunk_id
        self.calls = 0

    def complete(self, system, user, schema):
        self.calls += 1
        if self.calls == 1:
            quote = "Qubits never decohere at all."
        else:
            quote = "Superconducting qubits decohere quickly."
        return {
            "answer": "They decohere quickly [1].",
            "citations": [
                {"marker": 1, "chunk_id": self.chunk_id, "quoted_text": quote}
            ],
        }

    def stream(self, system, user):
        raise AssertionError("deep mode must not stream a draft it may discard")

    def judge(self, system, user):
        return "superconducting qubit decoherence"


def deep_client(tmp_path: Path) -> tuple[TestClient, LadderScript]:
    with chunked_corpus(
        tmp_path, {"2608.11111v1": ["Superconducting qubits decohere quickly."]}
    ) as store:
        chunk_id = store.all_chunks()[0].chunk_id
    model = LadderScript(chunk_id)
    app = create_app(
        tmp_path / "corpus.db", completion=model, encoder=StubEncoder()
    )
    return TestClient(app), model


def test_deep_streams_the_rungs_it_tries_and_the_answer_it_accepts(
    tmp_path: Path,
) -> None:
    """Rungs, not tokens.

    The ladder may write an answer, find that no citation survived, and retry.
    A draft shown to a reader and then withdrawn is worse than a slower page,
    so deep mode streams what is being tried and the answer only once a rung
    is accepted. LadderScript.stream raises if that is ever violated.
    """
    client_app, model = deep_client(tmp_path)

    events = sse_events(
        client_app.get("/api/ask", params={"q": "qubits", "deep": "true"}).text
    )

    names = [name for name, _ in events]
    assert "delta" not in names
    assert names.count("working") >= 2  # it retried rather than giving up
    assert names[-1] == "done"

    working = [payload for name, payload in events if name == "working"]
    assert working[0]["strategy"] == "hybrid" and working[0]["rung"] == 1
    assert working[1]["rung"] == 2

    answer = next(payload for name, payload in events if name == "answer")
    assert [c["arxiv_id"] for c in answer["citations"]] == ["2608.11111v1"]
    assert answer["rejected"] == []

    climb = next(payload for name, payload in events if name == "ladder")
    assert climb["shape"] == "open"
    assert climb["attempts"][0]["verdict"] == "retry"
    assert climb["attempts"][1]["verdict"] == "accepted"
    assert climb["outcome"] == "accepted"


def test_deep_does_not_warn_about_a_subject_it_already_escalated_on(
    tmp_path: Path,
) -> None:
    """Warning after escalating reports a failure that did not happen.

    Plain ask warns because it cannot act. The ladder retried on exactly that
    signal, so the accepted answer is the best it found and the subject is
    reported rather than complained about.
    """
    client_app, _ = deep_client(tmp_path)

    events = sse_events(
        client_app.get("/api/ask", params={"q": "qubits", "deep": "true"}).text
    )

    subject = next(payload for name, payload in events if name == "subject")
    assert subject["warn"] is False
    assert subject["subject"] == "superconducting qubit decoherence"


def test_plain_ask_still_streams_tokens(tmp_path: Path) -> None:
    """The toggle is off by default and changes nothing when it is."""
    fragments = ["ANSWER: They decohere quickly [1].\n", ""]
    client_app, chunk_id = ask_client(tmp_path, fragments)
    fragments[1] = f"CITE [1] {chunk_id} :: Superconducting qubits decohere quickly."

    events = sse_events(client_app.get("/api/ask", params={"q": "qubits"}).text)

    names = [name for name, _ in events]
    assert names.count("delta") == 2
    assert "working" not in names and "ladder" not in names


def test_search_runs_the_graph_leg_the_other_front_ends_run(tmp_path: Path) -> None:
    """The page's search skipped the leg the CLI, MCP and ask all default to.

    A definitional question reached it drowned in usage, so the same corpus
    answered "what is the Adam optimizer" differently depending on where it
    was asked.
    """
    from test_graph import TermEmbedder, graph_corpus
    from verbatim.embedding import embed_pending

    with graph_corpus(tmp_path) as corpus:
        embed_pending(corpus, TermEmbedder())

    hits = TestClient(
        create_app(tmp_path / "corpus.db", encoder=TermEmbedder())
    ).get("/api/search", params={"q": "what is the adam optimizer?", "limit": 2}).json()

    assert "1412.6980v1" in {hit["arxiv_id"] for hit in hits["results"]}


def test_a_follow_up_is_answered_as_the_question_it_was_resolved_to(
    tmp_path: Path,
) -> None:
    """The writer saw "why does it clip?" while retrieval saw the real question.

    Deep mode already wrote to the resolved question; plain mode handed the
    model a pronoun it had no way to resolve. The history itself still never
    reaches the writer - only the standalone question built from it, which the
    page shows the reader.
    """
    with chunked_corpus(tmp_path, {"2608.11111v1": ["PPO clips its objective ratio."]}):
        pass

    prompts: list[str] = []

    class Recording(StreamingScript):
        def complete_text(self, system: str, user: str) -> str:
            return "Why does PPO clip its objective?"

        def stream(self, system, user):
            prompts.append(user)
            yield "ANSWER: It clips to bound the update.\n"

    app = create_app(
        tmp_path / "corpus.db", completion=Recording([]), encoder=StubEncoder()
    )
    history = json.dumps([{"question": "what is PPO?", "answer": "A policy method."}])
    TestClient(app).get("/api/ask", params={"q": "why does it clip?", "history": history})

    (prompt,) = prompts
    assert "Question: Why does PPO clip its objective?" in prompt
    assert "A policy method." not in prompt
