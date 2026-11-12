"""The v1 arm of the v2 evaluation: every mode as its CLI runs it with no flags.

Pinned with the frozen sets before any v2 code exists, and run from that pinned
checkout. Each arm makes the calls `verbatim <command>` makes with its defaults,
reflection on as shipped (breadth_arms.py turns it off; this must not), and
tests/test_v1_arms.py holds every arm to the CLI on a scripted model. Cases come
only from sets whose bytes match the manifest's hash, and every result is the
record evals/arm_records.py defines.

  serve  one JSON request per stdin line -> one record per stdout line, so the
         v2 driver can interleave the arms question by question
  run    a whole set into a file, resuming, for a smoke test

A request is {"arm": ..., "set": ..., "case": <case id>, "order": 0 or 1}.
The worker only reads the store; point VERBATIM_STATE_PATH (or --state) at it.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib.request import urlopen

from verbatim import config
from verbatim.answers import (
    OllamaCompletion,
    OllamaToolCaller,
    RoleCompletion,
    answer_from_matches,
    subject_check,
)
from verbatim.corpus import CorpusStore, PaperFilter
from verbatim.embedding import FastEmbedEmbedder
from verbatim.evidence import describe_evidence
from verbatim.gaps import Gap, falsify
from verbatim.orchestrator import orchestrate
from verbatim.retrieval import default_search_options, hybrid_search
from verbatim.team import run_team

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_records import (  # noqa: E402
    COMPLETE,
    ERROR,
    INFRASTRUCTURE,
    MANIFEST,
    SetEntry,
    answer_fields,
    case_id,
    load_manifest,
    read_set,
    record,
)

ARM_NAME = "v1"

# The CLI's defaults, which the tests hold to the parser: `team`, `report
# --agents` and `gaps` with no other flags.
TEAM_TASKS = 3
TEAM_NEEDS = 2
TEAM_MAX_STEPS = 12
REPORT_SECTIONS = 4
REPORT_PASSAGES = 6
REPORT_MAX_PER_PAPER = 2
REPORT_WORKERS = 1
GAPS_QUERIES = 4


def search_options() -> dict[str, Any]:
    """What `ask` with no flags passes to the search: cli._search_options's defaults."""
    return {**default_search_options(), "use_graph": True, "use_compare": False}


def model_digests(host: str) -> dict[str, str]:
    """Each pulled model's digest, so a record says exactly which weights answered."""
    with urlopen(f"{host.rstrip('/')}/api/tags", timeout=config.HTTP_TIMEOUT) as response:  # noqa: S310
        body = json.loads(response.read())
    return {item["name"]: item.get("digest", "") for item in body.get("models", [])}


class Worker:
    """One v1 process: the clients the CLI would build, kept across requests."""

    def __init__(self, corpus: CorpusStore, *, llm: str | None, manifest: dict[str, SetEntry]):
        self.corpus = corpus
        self.llm = llm
        self.manifest = manifest
        self._sets: dict[str, dict[str, dict[str, Any]]] = {}
        self._clients: dict[str, Any] = {}
        self._digests: dict[str, str] | None = None

    # -- the clients, built exactly as each CLI command builds them -------------

    @property
    def embedder(self) -> Any:
        if "embedder" not in self._clients:
            self._clients["embedder"] = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
        return self._clients["embedder"]

    @property
    def completion(self) -> Any:
        """`ask`, `team` and `report` build RoleCompletion(args.llm)."""
        if "roles" not in self._clients:
            self._clients["roles"] = RoleCompletion(self.llm)
        return self._clients["roles"]

    @property
    def caller(self) -> Any:
        """`report --agents` gives the agent rung a tool caller on the planner's model."""
        if "caller" not in self._clients:
            self._clients["caller"] = OllamaToolCaller(config.role_model("planner", self.llm))
        return self._clients["caller"]

    @property
    def gaps_completion(self) -> Any:
        """`gaps` builds one plain client, not RoleCompletion."""
        if "gaps" not in self._clients:
            self._clients["gaps"] = OllamaCompletion(self.llm or config.OLLAMA_MODEL)
        return self._clients["gaps"]

    # -- the arms ----------------------------------------------------------------

    def ask(self, case: dict[str, Any]) -> dict[str, Any]:
        """`verbatim ask QUESTION`, with the case's hidden papers filtered out if it has any."""
        question = case["question"]
        search: dict[str, Any] = {
            "limit": config.ASK_PASSAGES,
            "arxiv_id": None,
            "max_per_paper": config.ASK_MAX_PER_PAPER,
            "include_references": False,
            **search_options(),
        }
        hidden = tuple(case.get("hide", ()))
        if hidden:
            # Any filter stands the title and graph legs down; both arms alike.
            search["papers"] = PaperFilter(exclude_ids=hidden)
        matches = hybrid_search(self.corpus, self.embedder, question, **search)
        answer = answer_from_matches(self.corpus, self.completion, question, matches)
        subject = closeness = None
        if matches:
            subject, closeness = subject_check(self.completion, self.embedder, question, matches)
        evidence = describe_evidence(
            self.corpus, question, matches, subject=subject, closeness=closeness
        )
        return {
            "answer": answer_fields(answer),
            "passages": [match.chunk.chunk_id for match in matches],
            "subject": subject,
            "closeness": closeness,
            "evidence": list(evidence.lines()),
        }

    def team(self, case: dict[str, Any]) -> dict[str, Any]:
        """`verbatim team QUESTION`."""
        completion = self.completion
        report = run_team(
            self.corpus, self.embedder, completion,
            lambda system, user: completion.complete_text(system, user),
            case["question"], max_tasks=TEAM_TASKS, needs=TEAM_NEEDS,
            max_steps=TEAM_MAX_STEPS,
        )
        return {
            "answer": answer_fields(report.final),
            "task_answers": [
                {"objective": task.objective,
                 "answer": answer_fields(report.answers.get(task.task_id))}
                for task in report.state.tasks
            ],
            "passages": [item.chunk_id for item in report.state.evidence],
            "decisions": len(report.state.decisions),
            "redirections": report.redirections,
        }

    def report(self, case: dict[str, Any]) -> dict[str, Any]:
        """`verbatim report TOPIC --agents`."""
        completion = self.completion
        built, ledger = orchestrate(
            self.corpus,
            self.embedder,
            completion,
            lambda system, user: completion.complete_text(system, user),
            case["topic"],
            caller=self.caller,
            max_sections=REPORT_SECTIONS,
            per_section=REPORT_PASSAGES,
            max_per_paper=REPORT_MAX_PER_PAPER,
            workers=REPORT_WORKERS,
            check_subject=True,
            budget=None,
            critic=False,
            on_section=_quiet,
            supervisor="builtin",
        )
        sections = []
        for section, kept in zip(built.sections, ledger.sections, strict=True):
            sections.append({
                "heading": section.heading,
                "answer": answer_fields(section.answer),
                "subject": section.subject,
                "closeness": section.subject_closeness,
                "evidence": list(section.evidence_lines),
                "outcome": kept.outcome,
                "attempts": [
                    {"strategy": a.strategy, "verdict": a.verdict, "error": a.error,
                     "chunks": list(a.chunks)}
                    for a in kept.attempts
                ],
            })
        return {"sections": sections, "escalations": ledger.escalations}

    def gaps(self, case: dict[str, Any]) -> dict[str, Any]:
        """`verbatim gaps`'s Falsifier on one frozen claim, with its paper visible and hidden."""
        completion = self.gaps_completion
        complete: Callable[[str, str], str] = (
            lambda system, user: completion.complete_text(system, user)
        )
        gap = Gap(topic=case["topic"], claim=case["claim"])
        findings = {}
        for condition, hidden in (("visible", []), ("hidden", [case["arxiv_id"]])):
            finding = falsify(
                self.corpus, self.embedder, completion, complete, gap,
                hidden=hidden, queries=GAPS_QUERIES,
            )
            findings[condition] = {
                "verdict": finding.verdict,
                "reason": finding.reason,
                "queries": list(finding.effort.queries),
                "papers": list(finding.effort.papers),
                "passages": finding.effort.passages,
                "citation": None if finding.citation is None else {
                    "chunk_id": finding.citation.chunk_id,
                    "arxiv_id": finding.citation.arxiv_id,
                    "quote": finding.citation.quoted_text,
                },
            }
        return {"planted": case["arxiv_id"], **findings}

    # -- requests ----------------------------------------------------------------

    def rows(self, set_name: str) -> dict[str, dict[str, Any]]:
        """A set's rows by case id, read once and only if its hash still matches."""
        if set_name not in self._sets:
            entry = self.manifest[set_name]
            self._sets[set_name] = {case_id(row): row for row in read_set(entry)}
        return self._sets[set_name]

    def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run one request and return its record, failures included."""
        set_name, case, order = request["set"], request["case"], int(request.get("order", 0))
        entry = self.manifest[set_name]
        if entry.arm != request["arm"]:
            raise ValueError(f"set {set_name!r} holds {entry.kind}, not {request['arm']} cases")
        row = self.rows(set_name)[case]
        run: Callable[[dict[str, Any]], dict[str, Any]] = getattr(self, entry.arm)
        clients = self._clients_for(entry.arm)
        marks = [usage.snapshot() for usage in _usages(clients)]
        started = time.monotonic()
        error: BaseException | None = None
        payload: dict[str, Any] = {}
        try:
            payload = run(row)
        except Exception as raised:  # noqa: BLE001 - every failure is recorded and classified
            error = raised
        spent = _spent(clients, marks, time.monotonic() - started)
        model = getattr(clients[0], "model", None) if clients else None
        return record(
            set_name=set_name, case=case, arm=ARM_NAME, order=order,
            status=ERROR if error is not None else COMPLETE, spent=spent,
            model=model, digest=self.digest(model), error=error, **payload,
        )

    def _clients_for(self, arm: str) -> list[Any]:
        if arm == "gaps":
            return [self.gaps_completion]
        if arm == "report":
            return [self.completion, self.caller]
        return [self.completion]

    def digest(self, model: str | None) -> str | None:
        if model is None:
            return None
        if self._digests is None:
            try:
                self._digests = model_digests(config.OLLAMA_HOST)
            except Exception:  # noqa: BLE001 - a missing digest is recorded as missing
                self._digests = {}
        return self._digests.get(model) or self._digests.get(f"{model}:latest")


def _quiet(heading: str, done: int, total: int) -> None:
    """The CLI prints section progress; stdout here carries records only."""


def _usages(clients: list[Any]) -> list[Any]:
    found = (getattr(client, "usage", None) for client in clients)
    return list({id(usage): usage for usage in found if usage is not None}.values())


def _spent(clients: list[Any], marks: list[tuple], seconds: float) -> dict[str, Any]:
    deltas = [usage.since(mark) for usage, mark in zip(_usages(clients), marks, strict=True)]
    return {
        "calls": sum(d.calls for d in deltas),
        "tokens_in": sum(d.prompt_tokens for d in deltas),
        "tokens_out": sum(d.output_tokens for d in deltas),
        "model_seconds": round(sum(d.seconds for d in deltas), 3),
        "seconds": round(seconds, 3),
    }


def serve(worker: Worker, lines: Any, out: Any) -> int:
    """Answer requests until stdin closes. An infrastructure failure ends the worker."""
    for line in lines:
        if not line.strip():
            continue
        result = worker.answer(json.loads(line))
        out.write(json.dumps(result) + "\n")
        out.flush()
        if result["error_kind"] == INFRASTRUCTURE:
            return 3
    return 0


def run_set(worker: Worker, set_name: str, out: Path, limit: int | None) -> int:
    """Every case of one set into `out`, skipping cases already there."""
    done = set()
    if out.exists():
        done = {json.loads(line)["case"] for line in out.read_text().splitlines() if line}
    out.parent.mkdir(parents=True, exist_ok=True)
    arm = worker.manifest[set_name].arm
    cases = list(worker.rows(set_name))[:limit]
    with out.open("a") as sink:
        for position, case in enumerate(cases, start=1):
            if case in done:
                continue
            result = worker.answer({"arm": arm, "set": set_name, "case": case, "order": 0})
            sink.write(json.dumps(result) + "\n")
            sink.flush()
            print(f"  {position}/{len(cases)} {result['status']}", file=sys.stderr, flush=True)
            if result["error_kind"] == INFRASTRUCTURE:
                print(f"STOPPED: {result['error']}", file=sys.stderr)
                return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["serve", "run"])
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    parser.add_argument("--llm", help="Ollama model for every role (default: as the CLI picks)")
    parser.add_argument("--set", help="run: the set to answer")
    parser.add_argument("--out", type=Path, help="run: where the records go")
    parser.add_argument("--limit", type=int, help="run: answer only the first N cases")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    with CorpusStore(args.state) as corpus:
        worker = Worker(corpus, llm=args.llm, manifest=manifest)
        if args.command == "serve":
            return serve(worker, sys.stdin, sys.stdout)
        if not args.set or not args.out:
            parser.error("run needs --set and --out")
        return run_set(worker, args.set, args.out, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
