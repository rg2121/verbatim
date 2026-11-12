"""The v1 worker makes exactly the calls the CLI makes, and records what the scorer needs.

The worker is a copy of the CLI's command paths, kept out of the CLI so v1 is not
edited before it is pinned. A copy can drift, so each arm runs here beside the
CLI on one scripted model and one fixture corpus, and both must make the same
model calls in the same order and hand the same arguments to the same library
functions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, is_dataclass
import io
import json
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
import arm_records  # noqa: E402
import v1_arms  # noqa: E402

from verbatim import config  # noqa: E402
from verbatim.answers import Usage  # noqa: E402
from verbatim.corpus import CorpusStore, EmbeddingModel  # noqa: E402
from verbatim.gaps import PROSPECTOR_SYSTEM, checkable  # noqa: E402

from helpers import chunked_corpus  # noqa: E402

PAGES = {
    "2608.11111v1": [
        "Transmon qubits lose coherence through dielectric loss in the substrate. "
        "Surface oxides add two-level systems that couple to the qubit."
    ],
    "2608.22222v1": [
        "Fluxonium qubits reach millisecond coherence with a large shunt inductance. "
        "Their spectra are strongly anharmonic, which eases control."
    ],
}
QUESTION = "how do transmon qubits lose coherence?"
CLAIM = "no paper measures transmon qubit coherence on sapphire above one second"
PASSAGE = re.compile(r'<passage chunk_id="([0-9a-f]+)"[^>]*>\n(.*?)\n</passage>', re.S)
# One reply serves every planner: team tasks, adversarial queries, report
# sections and gap claims each read only their own lines.
PLAN_REPLY = (
    "TASK: transmon coherence limits\n"
    "QUERY: dielectric loss in transmon qubits\n"
    f"GAP: {CLAIM}\n"
)


@dataclass
class Log:
    calls: list = field(default_factory=list)
    inits: list = field(default_factory=list)
    library: list = field(default_factory=list)


def payload(user: str, *, with_bad_quote: bool) -> dict:
    """Cite the first passage word for word; optionally add a quote no passage holds."""
    found = PASSAGE.search(user)
    if found is None:
        return {"answer": "Nothing here to cite.", "citations": []}
    chunk_id, text = found.groups()
    citations = [{"marker": 1, "chunk_id": chunk_id,
                  "quoted_text": " ".join(text.split()[:4])}]
    answer = "The passages describe this mechanism directly [1]."
    if with_bad_quote:
        citations.append({"marker": 2, "chunk_id": chunk_id,
                          "quoted_text": "words that appear in no passage at all"})
        answer = "The passages describe this mechanism directly [1] and more [2]."
    return {"answer": answer, "citations": citations}


def scripted(log: Log, kind: str):
    """A stand-in for one client class the CLI builds; every call lands in `log`."""

    class Scripted:
        model = "scripted"

        def __init__(self, *args, **options):
            log.inits.append((kind, args, tuple(sorted(options.items()))))
            self.usage = Usage()

        def _call(self, method: str, system: str, user: str) -> None:
            log.calls.append((method, system, user))
            self.usage.record({"prompt_eval_count": len(user), "eval_count": 7})

        def complete(self, system, user, schema):
            self._call("complete", system, user)
            return payload(user, with_bad_quote=True)

        def revise(self, system, user, schema):
            self._call("revise", system, user)
            return payload(user, with_bad_quote=False)

        def complete_text(self, system, user):
            self._call("complete_text", system, user)
            return PLAN_REPLY

        def judge(self, system, user):
            self._call("judge", system, user)
            return "superconducting qubit coherence"

        def choose_tool(self, system, messages, tools):
            self._call("choose_tool", system, json.dumps(messages))
            return None

    return Scripted


def embedder_class(log: Log):
    class Embedder:
        model_id = "stub"
        model = EmbeddingModel(model_id="stub", dimensions=4, normalized=True, revision="v0")

        def __init__(self, **options):
            log.inits.append(("embedder", (), tuple(sorted(options.items()))))

        def embed_query(self, text):
            raw = [len(text) % 7 + 1.0, text.count("q") + 1.0, text.count("o") + 1.0, 1.0]
            norm = sum(x * x for x in raw) ** 0.5
            return [x / norm for x in raw]

    return Embedder


def shape(value):
    """What an argument is, comparable across runs: values kept, objects named."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(shape(item) for item in value))
    if isinstance(value, dict):
        return tuple(sorted((key, shape(item)) for key, item in value.items()))
    if is_dataclass(value):
        return repr(value)
    if callable(value):
        return "callable"
    return type(value).__name__


def recording(log: Log, name: str, real):
    def wrapper(*args, **options):
        log.library.append((name, shape(args), shape(options)))
        return real(*args, **options)
    return wrapper


def patch_clients(monkeypatch, log: Log, module) -> None:
    """Swap the client classes `module` builds for scripted ones logging into `log`."""
    monkeypatch.setattr(module, "FastEmbedEmbedder", embedder_class(log))
    monkeypatch.setattr(module, "RoleCompletion", scripted(log, "roles"))
    monkeypatch.setattr(module, "OllamaCompletion", scripted(log, "plain"))
    monkeypatch.setattr(module, "OllamaToolCaller", scripted(log, "tools"))


@pytest.fixture
def corpus_path(tmp_path) -> Path:
    with chunked_corpus(tmp_path, PAGES):
        pass
    return tmp_path / "corpus.db"


@pytest.fixture(autouse=True)
def no_digest_lookup(monkeypatch):
    monkeypatch.setattr(v1_arms, "model_digests", lambda host: {"scripted": "sha256:feed"})


def run_cli(monkeypatch, argv: list[str]) -> None:
    from verbatim.cli import run

    monkeypatch.setattr("sys.argv", ["verbatim", *argv])
    run()


def worker(corpus: CorpusStore, manifest=None) -> v1_arms.Worker:
    return v1_arms.Worker(corpus, llm=None, manifest=manifest or {})


def test_ask_makes_the_calls_the_cli_makes_with_reflection_on(
    corpus_path, monkeypatch, capsys
) -> None:
    import verbatim.cli as cli
    from verbatim import retrieval

    # Shipped default; the suite turns it off, and the worker must not.
    monkeypatch.setattr(config, "REFLECT", True)

    via_cli = Log()
    patch_clients(monkeypatch, via_cli, cli)
    monkeypatch.setattr(cli, "hybrid_search",
                        recording(via_cli, "hybrid_search", retrieval.hybrid_search))
    run_cli(monkeypatch, ["ask", QUESTION, "--state", str(corpus_path)])
    printed = capsys.readouterr().out

    via_worker = Log()
    patch_clients(monkeypatch, via_worker, v1_arms)
    monkeypatch.setattr(v1_arms, "hybrid_search",
                        recording(via_worker, "hybrid_search", retrieval.hybrid_search))
    with CorpusStore(corpus_path) as corpus:
        result = worker(corpus).ask({"question": QUESTION})

    assert via_worker.calls == via_cli.calls
    assert [call[0] for call in via_cli.calls] == ["complete", "revise", "judge"]
    assert via_worker.library == via_cli.library
    assert Counter(via_worker.inits) == Counter(via_cli.inits)
    assert result["answer"]["text"] in printed
    # The repair was asked for; v1's keep rule decides whether it stands (here the
    # draft stays: the repair's sentence is cited but its words are not the passage's).
    assert result["answer"]["citations"]


def test_team_makes_the_calls_the_cli_makes(corpus_path, monkeypatch) -> None:
    import verbatim.cli as cli
    import verbatim.team as team

    real = team.run_team
    via_cli = Log()
    patch_clients(monkeypatch, via_cli, cli)
    monkeypatch.setattr(team, "run_team", recording(via_cli, "run_team", real))
    run_cli(monkeypatch, ["team", QUESTION, "--state", str(corpus_path)])

    via_worker = Log()
    patch_clients(monkeypatch, via_worker, v1_arms)
    monkeypatch.setattr(v1_arms, "run_team", recording(via_worker, "run_team", real))
    with CorpusStore(corpus_path) as corpus:
        result = worker(corpus).team({"question": QUESTION})

    assert via_cli.calls and via_worker.calls == via_cli.calls
    assert via_worker.library == via_cli.library
    assert Counter(via_worker.inits) == Counter(via_cli.inits)
    assert result["passages"]


def test_report_makes_the_calls_report_agents_makes(corpus_path, monkeypatch) -> None:
    import verbatim.cli as cli
    import verbatim.orchestrator as orchestrator

    real = orchestrator.orchestrate
    via_cli = Log()
    patch_clients(monkeypatch, via_cli, cli)
    monkeypatch.setattr(orchestrator, "orchestrate", recording(via_cli, "orchestrate", real))
    run_cli(monkeypatch, ["report", "transmon coherence", "--agents",
                          "--state", str(corpus_path)])

    via_worker = Log()
    patch_clients(monkeypatch, via_worker, v1_arms)
    monkeypatch.setattr(v1_arms, "orchestrate", recording(via_worker, "orchestrate", real))
    with CorpusStore(corpus_path) as corpus:
        result = worker(corpus).report({"topic": "transmon coherence"})

    assert via_cli.calls and via_worker.calls == via_cli.calls
    # The CLI prints progress through on_section and the worker stays quiet;
    # both pass a callable, which is all `shape` keeps.
    assert via_worker.library == via_cli.library
    assert Counter(via_worker.inits) == Counter(via_cli.inits)
    assert result["sections"] and all(s["attempts"] for s in result["sections"])


def test_gaps_falsifies_a_frozen_claim_as_the_cli_falsifies_a_proposed_one(
    corpus_path, monkeypatch
) -> None:
    import verbatim.cli as cli
    import verbatim.gaps as gaps

    assert checkable(CLAIM)
    real = gaps.falsify
    via_cli = Log()
    patch_clients(monkeypatch, via_cli, cli)
    monkeypatch.setattr(gaps, "falsify", recording(via_cli, "falsify", real))
    run_cli(monkeypatch, ["gaps", "transmon coherence", "--state", str(corpus_path)])

    via_worker = Log()
    patch_clients(monkeypatch, via_worker, v1_arms)
    monkeypatch.setattr(v1_arms, "falsify", recording(via_worker, "falsify", real))
    with CorpusStore(corpus_path) as corpus:
        result = worker(corpus).gaps(
            {"topic": "transmon coherence", "claim": CLAIM, "arxiv_id": "2608.11111v1"}
        )

    falsifying = [call for call in via_cli.calls if call[1] != PROSPECTOR_SYSTEM]
    assert falsifying and via_worker.calls[: len(falsifying)] == falsifying
    # The CLI proposes one gap and falsifies it with the paper visible; the
    # worker does that, then again with the planted paper hidden.
    visible, hidden = via_worker.library
    assert visible == via_cli.library[0]
    assert hidden[2] != visible[2] and "2608.11111v1" in str(hidden[2])
    assert Counter(via_worker.inits) == Counter(via_cli.inits)
    assert set(result) >= {"planted", "visible", "hidden"}


def test_the_worker_only_reads_the_store(corpus_path, monkeypatch) -> None:
    log = Log()
    patch_clients(monkeypatch, log, v1_arms)
    with CorpusStore(corpus_path) as corpus:
        before = corpus.connection.total_changes
        worker(corpus).ask({"question": QUESTION})
        assert corpus.connection.total_changes == before


def test_hidden_papers_never_reach_the_answer(corpus_path, monkeypatch) -> None:
    log = Log()
    patch_clients(monkeypatch, log, v1_arms)
    with CorpusStore(corpus_path) as corpus:
        result = worker(corpus).ask({"question": QUESTION, "hide": ["2608.11111"]})
    assert result["passages"]
    cited = {c["arxiv_id"] for c in result["answer"]["citations"]}
    assert "2608.11111v1" not in cited
    assert all("2608.11111" not in call[2] for call in log.calls)


def frozen(tmp_path: Path, name: str, kind: str, rows: list[dict]) -> dict:
    path = tmp_path / f"{name}.jsonl"
    path.write_text("# frozen\n" + "".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "manifest.json"
    sets = json.loads(manifest.read_text())["sets"] if manifest.exists() else {}
    sets[name] = {"path": path.name, "kind": kind,
                  "sha256": arm_records.sha256_of(path.read_bytes())}
    manifest.write_text(json.dumps({"sets": sets}))
    return arm_records.load_manifest(manifest)


def test_a_request_is_answered_from_its_frozen_set_with_every_record_field(
    tmp_path, corpus_path, monkeypatch
) -> None:
    patch_clients(monkeypatch, Log(), v1_arms)
    manifest = frozen(tmp_path, "dev", "questions",
                      [{"question": QUESTION, "expect_papers": ["2608.11111v1"]}])
    case = arm_records.case_id({"question": QUESTION})
    with CorpusStore(corpus_path) as corpus:
        result = worker(corpus, manifest).answer(
            {"arm": "ask", "set": "dev", "case": case, "order": 1}
        )
    assert result["status"] == arm_records.COMPLETE and result["error"] is None
    assert result["arm"] == "v1" and result["order"] == 1 and result["case"] == case
    assert result["digest"] == "sha256:feed" and result["model"] == "scripted"
    assert result["calls"] == 2 and result["tokens_in"] > 0 and result["seconds"] >= 0
    assert set(result) >= {"answer", "passages", "subject", "closeness", "evidence"}


def test_a_set_changed_after_freezing_is_refused(tmp_path, corpus_path, monkeypatch) -> None:
    manifest = frozen(tmp_path, "dev", "questions",
                      [{"question": QUESTION, "expect_papers": ["2608.11111v1"]}])
    (tmp_path / "dev.jsonl").write_text(json.dumps({"question": "edited"}) + "\n")
    with CorpusStore(corpus_path) as corpus, pytest.raises(arm_records.SetChanged):
        worker(corpus, manifest).rows("dev")


def test_a_request_for_the_wrong_kind_of_set_is_refused(tmp_path, corpus_path) -> None:
    manifest = frozen(tmp_path, "topics", "topics", [{"topic": "transmon coherence"}])
    case = arm_records.case_id({"topic": "transmon coherence"})
    with CorpusStore(corpus_path) as corpus, pytest.raises(ValueError, match="holds topics"):
        worker(corpus, manifest).answer({"arm": "ask", "set": "topics", "case": case})


def test_a_planted_claim_is_known_by_its_claim_not_its_topic() -> None:
    # A frozen claim also carries its paper's title as the topic the Falsifier gets.
    row = {"topic": "Transmon coherence", "claim": CLAIM, "arxiv_id": "2608.11111v1"}
    assert arm_records.identity(row) == CLAIM
    assert arm_records.identity({"topic": "transmon coherence"}) == "transmon coherence"


@pytest.mark.parametrize(("error", "kind"), [
    (RuntimeError("Cannot reach Ollama at http://x after 4 attempts (boom)."), "infrastructure"),
    (RuntimeError("Ollama at http://x refused the request (429 Too Many Requests)."),
     "infrastructure"),
    (ConnectionResetError("reset"), "infrastructure"),
    # Found by the first real smoke run: an exhausted cloud quota blamed the arm.
    (RuntimeError("Ollama at http://x refused the request (402 Payment Required)."),
     "infrastructure"),
    (RuntimeError("Ollama at http://x refused the request (401 Unauthorized)."),
     "infrastructure"),
    (RuntimeError("Ollama at http://x has no model named 'gone'. Pull it with ..."),
     "infrastructure"),
    (RuntimeError("Ollama at http://x refused the request (400 Bad Request)."), "arm"),
    (RuntimeError("Ollama error: model failed"), "arm"),
    (ValueError("a parser broke"), "arm"),
])
def test_failures_of_the_server_or_the_account_are_the_infrastructures(error, kind) -> None:
    assert arm_records.classify_error(error) == kind


def test_serving_stops_at_an_infrastructure_failure_and_records_it(
    tmp_path, corpus_path, monkeypatch
) -> None:
    patch_clients(monkeypatch, Log(), v1_arms)
    manifest = frozen(tmp_path, "dev", "questions", [
        {"question": "first question here", "expect_papers": ["2608.11111v1"]},
        {"question": "second question here", "expect_papers": ["2608.11111v1"]},
        {"question": "third question here", "expect_papers": ["2608.11111v1"]},
    ])
    failures = iter([ValueError("a parser broke"), RuntimeError("Cannot reach Ollama")])

    def failing(self, case):
        raise next(failures)

    monkeypatch.setattr(v1_arms.Worker, "ask", failing)
    requests = "".join(
        json.dumps({"arm": "ask", "set": "dev", "case": arm_records.case_id({"question": q})})
        + "\n"
        for q in ("first question here", "second question here", "third question here")
    )
    out = io.StringIO()
    with CorpusStore(corpus_path) as corpus:
        code = v1_arms.serve(worker(corpus, manifest), io.StringIO(requests), out)

    records = [json.loads(line) for line in out.getvalue().splitlines()]
    assert code == 3 and len(records) == 2
    assert [r["error_kind"] for r in records] == ["arm", "infrastructure"]
    assert all(r["status"] == arm_records.ERROR for r in records)


def test_the_cli_defaults_the_worker_copies_are_still_the_cli_defaults() -> None:
    """The numbers the worker passes are the parser's defaults, read from the parser."""
    import argparse
    import verbatim.cli as cli

    seen = {}

    def capture(self, args=None, namespace=None):
        seen["parser"] = self
        raise SystemExit(0)

    original = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = capture
    try:
        with pytest.raises(SystemExit):
            cli.run()
    finally:
        argparse.ArgumentParser.parse_args = original
    parse = seen["parser"].parse_args
    team = parse(["team", "q"])
    report = parse(["report", "t", "--agents"])
    gaps_args = parse(["gaps", "t"])
    ask = parse(["ask", "q"])
    assert (team.tasks, team.needs, team.max_steps) == (
        v1_arms.TEAM_TASKS, v1_arms.TEAM_NEEDS, v1_arms.TEAM_MAX_STEPS)
    assert (report.sections, report.passages, report.max_per_paper, report.workers) == (
        v1_arms.REPORT_SECTIONS, v1_arms.REPORT_PASSAGES, v1_arms.REPORT_MAX_PER_PAPER,
        v1_arms.REPORT_WORKERS)
    assert not report.critic and report.supervisor == "builtin"
    assert gaps_args.queries == v1_arms.GAPS_QUERIES and gaps_args.hide == []
    assert ask.passages == config.ASK_PASSAGES
    assert ask.max_per_paper == config.ASK_MAX_PER_PAPER
    assert ask.use_graph and not ask.use_compare and not ask.no_rerank
