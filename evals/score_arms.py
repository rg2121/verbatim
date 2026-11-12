"""Score both arms of the v2 evaluation from their records, trusting neither.

Pinned with v1 and the frozen sets before any v2 code exists, so its rules are
the plan's pre-registered ones and cannot be fitted to a result. Nothing an arm
says about itself counts:

- a citation is verified only if its quote is, word for word, in the passage the
  store holds for that chunk - v1's own check, run again here;
- claim sentences, expected papers and refutations are counted here;
- subject closeness is recomputed from the subject the arm recorded.

A record whose run failed scores 0. One the infrastructure broke must have been
rerun for both arms first; the scorer refuses to start otherwise.

  sheet  a blind sheet for your scoring, one per set: answers (P1), team (the
         breadth key's criteria) or reports. The mapping goes outside the
         repository and a manifest of its hashes must be committed before any score.
  gate   one gate's verdict, from both arms' records and, for P1, team and
         reports, your committed labels.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import html
import json
import os
from pathlib import Path
import random
import re
import secrets
from statistics import mean, median
import sys
from types import SimpleNamespace
from typing import Any

from verbatim import config
from verbatim.answers import subject_similarity, verify_quote
from verbatim.answer_eval import cites_an_expected_paper
from verbatim.corpus import CorpusStore
from verbatim.critique import MIN_CLAIM_WORDS, sentences, uncited_sentences, unbacked_sentences
from verbatim.gaps import REFUTED

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_records import ERROR, INFRASTRUCTURE, MANIFEST, case_id, load_manifest, read_set  # noqa: E402
from significance import bootstrap_ci, paired_sign_test  # noqa: E402

V1, V2 = "v1", "v2"

# Pre-registered in plan revision 6. Changing one after the pin is a new experiment.
# Every gate runs on the local model, so protocol failures and latency take the
# thresholds the plan set for that model.
DELTA = {"p1": 0.10, "p2": 0.05, "p3": 0.05, "abstention": 0.08}
BOOTSTRAP_SEEDS = 10
BOOTSTRAP_ROUNDS = 10_000
BLIND_SLACK = 4          # team and reports: v2's blind losses <= wins + 4
GAPS_SLACK = 4           # planted refutations v2 misses that v1 found, minus the reverse
REPORT_SHARE_MARGIN = 0.07
PROTOCOL_FAILURES = 0.05
BUDGET_EXHAUSTED = 0.05
CALLS_RATIO = 3.0
LATENCY_RATIO = 3.0
REPORT_TIME_RATIO = 3.0

PROTOCOL_FAILURE = "incomplete_protocol_failure"
BUDGET_STATUS = "budget_exhausted"
INCOMPLETE = (PROTOCOL_FAILURE, BUDGET_STATUS, ERROR)

MARKER_NUMBER = re.compile(r"\[(\d+)\]")
BLIND_DIR = Path.home() / ".verbatim-blind"
SHEET_MANIFESTS = Path(__file__).parent / "v2" / "blind"
NEUTRAL_DECLINE = "No answer - no passage could be cited for this question."

CRITERIA: dict[str, list[tuple[str, str, tuple[str, str, str]]]] = {
    "answers": [
        ("correct", "Correct and supported", (
            "wrong or unsupported, or declines a question the passages answer",
            "partly correct, or some claims rest on nothing quoted",
            "correct, and each claim rests on a quoted passage",
        )),
    ],
    # The breadth key's human measures, as evals/breadth/PROTOCOL.md defines them.
    "team": [
        ("coverage", "Coverage", (
            "addresses at most one of the question's two subjects",
            "both, one only in passing", "both, substantively",
        )),
        ("balance", "Balance", (
            "one-sided", "a second position mentioned without support",
            "each position backed by its own citation",
        )),
        ("limitations", "Limitations", (
            "none", "a limitation stated without a citation",
            "a limitation or counterpoint backed by a citation",
        )),
    ],
    "reports": [
        ("coverage", "Coverage", (
            "misses most of the topic", "covers part of it", "covers the topic's main lines",
        )),
        ("support", "Support", (
            "claims rest on nothing quoted", "some claims rest on quotes",
            "claims rest on quoted passages",
        )),
        ("coherence", "Coherence", (
            "sections repeat or contradict each other", "some overlap or loose ends",
            "the sections build one account",
        )),
    ],
}


# -- records -------------------------------------------------------------------


def load_records(path: Path, arm: str) -> dict[tuple[str, str], dict[str, Any]]:
    """One arm's last record per (set, case): a rerun replaces what it reran."""
    kept: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["arm"] != arm:
            raise ValueError(f"{path} holds a {row['arm']!r} record, not {arm!r}")
        kept[(row["set"], row["case"])] = row
    broken = [key for key, row in kept.items() if row.get("error_kind") == INFRASTRUCTURE]
    if broken:
        raise SystemExit(
            f"error: {len(broken)} case(s) in {path} ended on an infrastructure failure. "
            "The protocol reruns those for both arms before anything is scored."
        )
    return kept


def pairs(v1: dict, v2: dict, set_name: str) -> list[tuple[str, dict, dict]]:
    """Both arms' records for every case of one set, refusing a case only one arm ran."""
    cases = sorted({c for s, c in v1 if s == set_name} | {c for s, c in v2 if s == set_name})
    missing = [c for c in cases if (set_name, c) not in v1 or (set_name, c) not in v2]
    if missing:
        raise SystemExit(f"error: {len(missing)} case(s) of {set_name} lack one arm's record")
    if not cases:
        raise SystemExit(f"error: no records for set {set_name}")
    return [(c, v1[(set_name, c)], v2[(set_name, c)]) for c in cases]


def require_every_case(v1: dict, v2: dict, set_name: str, rows: list[dict]) -> None:
    """Both arms ran exactly the frozen set's cases, so no part of a run is ever scored."""
    expected = {case_id(row) for row in rows}
    for arm, records in ((V1, v1), (V2, v2)):
        ran = {case for s, case in records if s == set_name}
        if ran != expected:
            raise SystemExit(
                f"error: {arm} has records for {len(ran & expected)} of {set_name}'s "
                f"{len(expected)} cases and {len(ran - expected)} outside it; every sheet "
                "and gate waits until both arms ran the whole set"
            )


class Passages:
    """Passage text and paper from the store: the only source the scorer believes."""

    def __init__(self, corpus: CorpusStore) -> None:
        self.corpus = corpus
        self._cache: dict[str, tuple[str, str] | None] = {}

    def _load(self, chunk_ids: Sequence[str]) -> None:
        wanted = [c for c in dict.fromkeys(chunk_ids) if c not in self._cache]
        if wanted:
            for chunk in self.corpus.chunks_by_id(wanted):
                self._cache[chunk.chunk_id] = (chunk.text, chunk.arxiv_id)
            for chunk_id in wanted:
                self._cache.setdefault(chunk_id, None)

    def texts(self, chunk_ids: Sequence[str]) -> dict[str, str]:
        self._load(chunk_ids)
        return {c: found[0] for c in chunk_ids if (found := self._cache.get(c)) is not None}

    def paper(self, chunk_id: str) -> str | None:
        """The paper the store files this passage under, whatever a record says."""
        self._load([chunk_id])
        found = self._cache.get(chunk_id)
        return None if found is None else found[1]


# -- one answer, checked again ---------------------------------------------------


@dataclass(frozen=True)
class Checked:
    """An answer as the scorer reads it, from its text and the store alone."""

    verified: frozenset[int] = frozenset()
    papers: frozenset[str] = frozenset()
    failed: int = 0
    claims: int = 0
    grounded: int = 0
    uncited: int = 0
    unbacked: int = 0

    @property
    def grounding(self) -> bool:
        """P2: at least one verified citation, and every claim sentence carries one.

        The first clause matters: a decline has no claim sentence to fail.
        """
        return bool(self.verified) and self.grounded == self.claims


def check(answer: dict[str, Any] | None, passages: Passages) -> Checked:
    """Recheck every quote and count the claim sentences, as v1's own checks do."""
    if not answer:
        return Checked()
    text = answer["text"]
    by_marker = {int(c["marker"]): c for c in answer.get("citations", [])}
    texts = passages.texts([c["chunk_id"] for c in by_marker.values()])
    pieces = sentences(text)
    present = {int(n) for _, read in pieces for n in MARKER_NUMBER.findall(read)}
    verified: set[int] = set()
    failed = len(answer.get("rejected", []))
    for marker, citation in by_marker.items():
        passage = texts.get(citation["chunk_id"])
        if passage is not None and verify_quote(citation["quote"], passage) is None:
            if marker in present:
                verified.add(marker)
        else:
            failed += 1
    claims = grounded = 0
    for sentence, read in pieces:
        if len(sentence.strip().split()) < MIN_CLAIM_WORDS:
            continue
        claims += 1
        if {int(n) for n in MARKER_NUMBER.findall(read)} & verified:
            grounded += 1
    ordered = [
        SimpleNamespace(chunk_id=by_marker[m]["chunk_id"] if m in by_marker else "")
        for m in range(1, max(by_marker, default=0) + 1)
    ]
    unbacked = unbacked_sentences(SimpleNamespace(text=text, citations=ordered), texts)
    return Checked(
        verified=frozenset(verified),
        papers=frozenset(passages.paper(by_marker[m]["chunk_id"]) for m in verified),
        failed=failed,
        claims=claims,
        grounded=grounded,
        uncited=len(uncited_sentences(text)),
        unbacked=len(unbacked),
    )


def checked(row: dict[str, Any], passages: Passages) -> Checked:
    """A failed run scores 0; anything else is scored on what it wrote."""
    if row["status"] == ERROR:
        return Checked()
    return check(row.get("answer"), passages)


# -- statistics --------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """A non-inferiority test: passes if the interval's lower end is above -delta."""

    name: str
    cases: int
    v1: float
    v2: float
    low: float
    delta: float
    wins: int
    losses: int

    @property
    def passed(self) -> bool:
        return self.low > -self.delta

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def noninferior(name: str, v1: Sequence[float], v2: Sequence[float], delta: float,
                *, seeds: int = BOOTSTRAP_SEEDS, rounds: int = BOOTSTRAP_ROUNDS) -> Comparison:
    """The paired bootstrap's lower bound, the lowest over every seed.

    One seed's percentile can flip a verdict when differences take few values,
    so the bound that decides is the least favourable of ten.
    """
    diffs = [b - a for a, b in zip(v1, v2, strict=True)]
    low = min(bootstrap_ci(diffs, rounds=rounds, seed=seed)[0] for seed in range(seeds))
    wins, losses, _ = paired_sign_test(diffs)
    return Comparison(name, len(diffs), mean(v1), mean(v2), low, delta, wins, losses)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values, for the secondaries that decide nothing."""
    adjusted: dict[str, float] = {}
    running = 0.0
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def ratio(v2: Sequence[float], v1: Sequence[float]) -> float:
    base = median(v1)
    return float("inf") if base <= 0 else median(v2) / base


# -- the gates ---------------------------------------------------------------------


def expected_papers(entry_rows: list[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    return {case_id(row): tuple(row.get("expect_papers", ())) for row in entry_rows}


def gate_ask(v1: dict, v2: dict, set_name: str, rows: list[dict], passages: Passages,
             **stats: Any) -> dict[str, Any]:
    """P2 and P3, the cost ratios, and the secondaries, on the test set."""
    expected = expected_papers(rows)
    paired = [(c, checked(a, passages), checked(b, passages), a, b)
              for c, a, b in pairs(v1, v2, set_name)]
    p2 = noninferior("p2", [x.grounding for _, x, _, _, _ in paired],
                     [y.grounding for _, _, y, _, _ in paired], DELTA["p2"], **stats)
    p3 = noninferior(
        "p3",
        [cites_an_expected_paper(x.papers, expected[c]) for c, x, _, _, _ in paired],
        [cites_an_expected_paper(y.papers, expected[c]) for c, _, y, _, _ in paired],
        DELTA["p3"], **stats,
    )
    secondary = {}
    for field in ("failed", "unbacked", "uncited"):
        diffs = [getattr(y, field) - getattr(x, field) for _, x, y, _, _ in paired]
        wins, losses, p = paired_sign_test(diffs)
        secondary[field] = {"v1": mean(getattr(x, field) for _, x, _, _, _ in paired),
                            "v2": mean(getattr(y, field) for _, _, y, _, _ in paired),
                            "v2_higher": wins, "v2_lower": losses, "p": p}
    for name, adjusted in holm({k: v["p"] for k, v in secondary.items()}).items():
        secondary[name]["p_holm"] = adjusted
    calls = ratio([b["calls"] for _, _, _, _, b in paired], [a["calls"] for _, _, _, a, _ in paired])
    latency = ratio([b["seconds"] for _, _, _, _, b in paired],
                    [a["seconds"] for _, _, _, a, _ in paired])
    cost = {"calls_ratio": calls, "calls_passed": calls <= CALLS_RATIO,
            "latency_ratio": latency, "latency_passed": latency <= LATENCY_RATIO}
    return {
        "gate": "ask", "set": set_name, "p2": p2.to_dict(), "p3": p3.to_dict(),
        "cost": cost, "secondary": secondary, "reported": reported(v2, set_name),
        "passed": p2.passed and p3.passed and cost["calls_passed"] and cost["latency_passed"],
    }


def declines_or_warns(row: dict[str, Any], answer: Checked, question: str,
                      closeness: Callable[[str, str], float]) -> bool:
    """No verified citation, or the recomputed subject closeness is below the warning."""
    if row["status"] == ERROR:
        return False
    if not answer.verified:
        return True
    subject = row.get("subject")
    return bool(subject) and closeness(question, subject) < config.SUBJECT_WARN_BELOW


def gate_abstention(v1: dict, v2: dict, set_name: str, rows: list[dict], passages: Passages,
                    closeness: Callable[[str, str], float], **stats: Any) -> dict[str, Any]:
    questions = {case_id(row): row["question"] for row in rows}
    paired = pairs(v1, v2, set_name)
    first = [declines_or_warns(a, checked(a, passages), questions[c], closeness)
             for c, a, _ in paired]
    second = [declines_or_warns(b, checked(b, passages), questions[c], closeness)
              for c, _, b in paired]
    result = noninferior("abstention", first, second, DELTA["abstention"], **stats)
    hiding = sum(bool(row.get("hide")) for row in rows)
    note = (f"{hiding} of {len(rows)} cases hide their answering paper, so the title and graph "
            "legs are off for them in both arms")
    if hiding < len(rows):
        note += f"; the other {len(rows) - hiding} hide nothing and run with the defaults"
    return {"gate": "abstention", "set": set_name, "abstention": result.to_dict(),
            "note": note, "passed": result.passed}


def gate_p1(labels: dict[str, dict[str, int]], v1: dict, v2: dict, set_name: str,
            **stats: Any) -> dict[str, Any]:
    """Your blind scores of 2 on the sampled cases; a failed run scores 0."""
    cases = sorted(labels)
    def score(row: dict, arm: str, case: str) -> bool:
        return row["status"] != ERROR and labels[case][arm]["correct"] == 2
    first = [score(v1[(set_name, c)], V1, c) for c in cases]
    second = [score(v2[(set_name, c)], V2, c) for c in cases]
    result = noninferior("p1", first, second, DELTA["p1"], **stats)
    return {"gate": "p1", "set": set_name, "p1": result.to_dict(), "passed": result.passed}


def blind_count(labels: dict[str, dict[str, dict[str, int]]], kind: str) -> dict[str, Any]:
    """Per case, the summed criteria decide a win or a loss; ties count for neither."""
    names = [name for name, _, _ in CRITERIA[kind]]
    wins = losses = 0
    for scores in labels.values():
        first = sum(scores[V1][n] for n in names)
        second = sum(scores[V2][n] for n in names)
        wins += second > first
        losses += second < first
    return {"wins": wins, "losses": losses, "passed": losses <= wins + BLIND_SLACK}


def gate_team(labels: dict, v1: dict, v2: dict, set_name: str) -> dict[str, Any]:
    pairs(v1, v2, set_name)
    blind = blind_count(labels, "team")
    return {"gate": "team", "set": set_name, "blind": blind, "passed": blind["passed"]}


def report_share(row: dict[str, Any], passages: Passages) -> tuple[int, int]:
    """Claim sentences carrying a verified citation, and all claim sentences, over sections.

    A failed run wrote no sentence, so it adds nothing here; the incomplete
    clause is what counts it against its arm.
    """
    if row["status"] == ERROR:
        return 0, 0
    grounded = claims = 0
    for section in row.get("sections", []):
        answer = check(section.get("answer"), passages)
        grounded += answer.grounded
        claims += answer.claims
    return grounded, claims


def v1_failed_report(row: dict[str, Any]) -> bool:
    return row["status"] == ERROR or any(
        str(section.get("outcome", "")).startswith("failed") for section in row.get("sections", [])
    )


def gate_reports(labels: dict, v1: dict, v2: dict, set_name: str,
                 passages: Passages) -> dict[str, Any]:
    paired = pairs(v1, v2, set_name)
    blind = blind_count(labels, "reports")
    shares = {}
    for arm, rows in ((V1, [a for _, a, _ in paired]), (V2, [b for _, _, b in paired])):
        counted = [report_share(row, passages) for row in rows]
        shares[arm] = sum(g for g, _ in counted) / max(sum(c for _, c in counted), 1)
    share_passed = shares[V2] >= shares[V1] - REPORT_SHARE_MARGIN
    failed_v1 = sum(v1_failed_report(a) for _, a, _ in paired)
    incomplete_v2 = sum(b["status"] in INCOMPLETE for _, _, b in paired)
    time_ratio = ratio([b["seconds"] for _, _, b in paired], [a["seconds"] for _, a, _ in paired])
    return {
        "gate": "reports", "set": set_name, "blind": blind,
        "verified_share": {**shares, "passed": share_passed},
        "incomplete": {"v1_failed": failed_v1, "v2_incomplete": incomplete_v2,
                       "passed": incomplete_v2 <= failed_v1 + 1},
        "time_ratio": time_ratio, "time_passed": time_ratio <= REPORT_TIME_RATIO,
        "accepted_knowingly": "a 7-3 blind loss on 10 topics passes; the share is the backstop",
        "passed": blind["passed"] and share_passed and incomplete_v2 <= failed_v1 + 1
                  and time_ratio <= REPORT_TIME_RATIO,
    }


def refuted(row: dict[str, Any], condition: str, passages: Passages) -> bool:
    """A refutation counts only if its quote is in its passage, checked here."""
    if row["status"] == ERROR:
        return False
    finding = row.get(condition) or {}
    citation = finding.get("citation")
    if finding.get("verdict") != REFUTED or not citation:
        return False
    passage = passages.texts([citation["chunk_id"]]).get(citation["chunk_id"])
    return passage is not None and verify_quote(citation["quote"], passage) is None


def gate_gaps(v1: dict, v2: dict, set_name: str, passages: Passages) -> dict[str, Any]:
    paired = pairs(v1, v2, set_name)
    misses = sum(refuted(a, "visible", passages) and not refuted(b, "visible", passages)
                 for _, a, b in paired)
    gains = sum(refuted(b, "visible", passages) and not refuted(a, "visible", passages)
                for _, a, b in paired)
    hidden = {V1: sum(refuted(a, "hidden", passages) for _, a, _ in paired),
              V2: sum(refuted(b, "hidden", passages) for _, _, b in paired)}
    return {"gate": "gaps", "set": set_name, "v2_misses": misses, "v2_gains": gains,
            "false_refutations_hidden": hidden, "passed": misses - gains <= GAPS_SLACK}


def gate_runtime(v2_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = max(len(v2_rows), 1)
    protocol = sum(row["status"] == PROTOCOL_FAILURE for row in v2_rows) / total
    budget = sum(row["status"] == BUDGET_STATUS for row in v2_rows) / total
    return {"gate": "runtime", "runs": len(v2_rows), "protocol_failures": protocol,
            "budget_exhausted": budget,
            "passed": protocol <= PROTOCOL_FAILURES and budget <= BUDGET_EXHAUSTED}


def reported(v2: dict, set_name: str) -> dict[str, Any]:
    """What v2 records about its own choices, reported and never gated."""
    rows = [row for (s, _), row in v2.items() if s == set_name]
    inspected = [row.get("inspect", {}) for row in rows]
    used = sum(item.get("used", 0) for item in inspected)
    changed = sum(item.get("not_accept_after", 0) for item in inspected)
    decisions: dict[str, int] = {}
    for row in rows:
        for decision, count in (row.get("verifier_decisions") or {}).items():
            decisions[decision] = decisions.get(decision, 0) + count
    return {"inspect_used": used,
            "inspect_changed_share": changed / used if used else None,
            "verifier_decisions_on_failed_checks": decisions}


# -- blind sheets ------------------------------------------------------------------


def shown(answer: dict[str, Any] | None, row: dict[str, Any]) -> dict[str, Any]:
    """One answer as the sheet shows it; every decline and failure reads alike."""
    if row["status"] == ERROR or not answer or answer.get("declined"):
        return {"text": NEUTRAL_DECLINE, "refs": []}
    return {"text": answer["text"],
            "refs": [{"n": c["marker"], "id": c["arxiv_id"], "quote": c["quote"]}
                     for c in answer.get("citations", [])]}


def shown_report(row: dict[str, Any]) -> dict[str, Any]:
    if row["status"] == ERROR:
        return {"text": NEUTRAL_DECLINE, "refs": []}
    parts, refs = [], []
    for number, section in enumerate(row.get("sections", []), start=1):
        body = shown(section.get("answer"), {"status": "complete"})
        parts.append(f"{number}. {section['heading']}\n{body['text']}")
        refs += [{**ref, "n": f"{number}.{ref['n']}"} for ref in body["refs"]]
    return {"text": "\n\n".join(parts), "refs": refs}


def build_sheet(kind: str, cases: list[tuple[str, str, dict, dict]], seed: str
                ) -> tuple[list[dict], dict[str, dict[str, str]]]:
    """Items for the page, and the letter-to-arm mapping that must stay sealed."""
    items, mapping = [], {}
    for case, prompt, first, second in cases:
        order = [V1, V2]
        random.Random(f"{seed}:{case}").shuffle(order)
        letters = dict(zip("AB", order, strict=True))
        mapping[case] = letters
        by_arm = {V1: first, V2: second}
        answers = []
        for letter, arm in letters.items():
            row = by_arm[arm]
            body = shown_report(row) if kind == "reports" else shown(row.get("answer"), row)
            answers.append({"letter": letter, **body})
        items.append({"id": case, "prompt": prompt, "answers": answers})
    return items, mapping


def records_digest(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sheet_manifest(kind: str, mapping_bytes: bytes, records_sha256: str, cases: int) -> dict:
    mapping_hash = hashlib.sha256(mapping_bytes).hexdigest()
    return {
        "kind": kind,
        "mapping_sha256": mapping_hash,
        "records_sha256": records_sha256,
        "cases": cases,
        "storage_key": f"v2-{kind}-{mapping_hash[:12]}",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def unblind(manifest: dict, mapping_bytes: bytes, records_sha256: str,
            labels: dict) -> dict[str, dict[str, dict[str, int]]]:
    """Your scores by arm, once the mapping and records are shown to be the committed ones.

    Every answer needs a 0, 1 or 2 on every row: a case left out would
    otherwise drop out of the gate unnoticed.
    """
    if hashlib.sha256(mapping_bytes).hexdigest() != manifest["mapping_sha256"]:
        raise SystemExit("error: this is not the mapping whose hash was committed before scoring")
    if records_sha256 != manifest["records_sha256"]:
        raise SystemExit("error: the records changed after the sheet was built")
    scores = dict(labels)
    if scores.pop("blinding", None) != manifest["storage_key"]:
        raise SystemExit("error: these scores were made on another sheet's blinding")
    mapping = json.loads(mapping_bytes)["mapping"]
    names = [name for name, _, _ in CRITERIA[manifest["kind"]]]

    def scored(case: str, letter: str) -> bool:
        given = scores.get(case, {}).get(letter, {})
        return all(type(given.get(name)) is int and given[name] in (0, 1, 2) for name in names)

    unscored = [case for case, letters in mapping.items()
                if not all(scored(case, letter) for letter in letters)]
    if unscored:
        raise SystemExit(f"error: {len(unscored)} of the sheet's {len(mapping)} cases lack a "
                         "0, 1 or 2 on some row; finish the sheet before unblinding")
    return {case: {arm: scores[case][letter] for letter, arm in letters.items()}
            for case, letters in mapping.items()}


def sheet_paths(kind: str, set_name: str) -> tuple[Path, Path]:
    """One set's sheet manifest (committed) and its mapping (sealed, outside the repository).

    Named by set as well as kind, so the sealed second evaluation gets sheets of its own.
    """
    return (SHEET_MANIFESTS / f"{kind}-{set_name}.json",
            BLIND_DIR / f"v2-{kind}-{set_name}-mapping.json")


def render_sheet(kind: str, items: list[dict], storage_key: str) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(label)}</th>"
        + "".join(f"<td><b>{n}</b> {html.escape(text)}</td>" for n, text in enumerate(levels))
        + "</tr>"
        for _, label, levels in CRITERIA[kind]
    )
    data = json.dumps(items).replace("</", "<\\/")
    criteria = json.dumps([[name, label] for name, label, _ in CRITERIA[kind]])
    return (SHEET.replace("__TITLE__", f"v2 blind sheet: {kind}").replace("__RUBRIC__", rows)
            .replace("__DATA__", data).replace("__CRITERIA__", criteria)
            .replace("__KEY__", json.dumps(storage_key)))


SHEET = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--ground:#F4F6F8;--surface:#FFF;--ink:#14181D;--muted:#5D6772;--rule:#DAE0E6;--accent:#2E5C8A}
@media (prefers-color-scheme:dark){:root{--ground:#0F1317;--surface:#171C21;--ink:#E5E9ED;--muted:#96A1AC;--rule:#28313A;--accent:#80B1DF}}
body{margin:0;background:var(--ground);color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:1rem}
table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{border-top:1px solid var(--rule);padding:.35rem;text-align:left;vertical-align:top}
.item{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:1rem;margin:1rem 0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:1rem}@media (max-width:700px){.pair{grid-template-columns:1fr}}
.text{white-space:pre-wrap}.ref{font-size:.82rem;color:var(--muted)}button{font:inherit;padding:.4rem .8rem}
</style></head><body><main>
<h1>__TITLE__</h1><p>Score each answer on each row, 0 / 1 / 2. You are not told which system wrote which answer.</p>
<table>__RUBRIC__</table><p><button id="download" type="button">Download labels.json</button> <span id="progress"></span></p>
<div id="items"></div></main>
<script>
const ITEMS = __DATA__; const CRITERIA = __CRITERIA__; const KEY = __KEY__;
let labels = {}; try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
function save(){ try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} progress(); }
function progress(){ const done = ITEMS.filter(i => i.answers.every(a => CRITERIA.every(([c]) => Number.isInteger(labels[i.id]?.[a.letter]?.[c])))).length;
  document.getElementById("progress").textContent = `${done} of ${ITEMS.length} scored`; }
const box = document.getElementById("items");
for (const item of ITEMS) {
  const card = document.createElement("section"); card.className = "item";
  const q = document.createElement("h2"); q.textContent = item.prompt; card.appendChild(q);
  const pair = document.createElement("div"); pair.className = "pair";
  for (const a of item.answers) {
    const col = document.createElement("div");
    const h = document.createElement("h3"); h.textContent = "Answer " + a.letter; col.appendChild(h);
    const t = document.createElement("div"); t.className = "text"; t.textContent = a.text; col.appendChild(t);
    for (const r of a.refs) { const p = document.createElement("p"); p.className = "ref"; p.textContent = `[${r.n}] ${r.id}: "${r.quote}"`; col.appendChild(p); }
    for (const [c, label] of CRITERIA) {
      const row = document.createElement("p"); row.textContent = label + ": ";
      for (const v of [0, 1, 2]) {
        const l = document.createElement("label"); const i = document.createElement("input");
        i.type = "radio"; i.name = `${item.id}-${a.letter}-${c}`; i.checked = labels[item.id]?.[a.letter]?.[c] === v;
        i.onchange = () => { ((labels[item.id] ||= {})[a.letter] ||= {})[c] = v; save(); };
        l.append(i, " " + v + " "); row.appendChild(l);
      }
      col.appendChild(row);
    }
    pair.appendChild(col);
  }
  card.appendChild(pair); box.appendChild(card);
}
progress();
document.getElementById("download").onclick = () => {
  const blob = new Blob([JSON.stringify({blinding: KEY, ...labels}, null, 2) + "\n"], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "labels.json"; a.click();
};
</script></body></html>
"""


# -- command line --------------------------------------------------------------------


def sampled(rows: list[dict], size: int | None, seed: int) -> list[dict]:
    """The human sample: drawn with the seed frozen in the manifest."""
    if size is None or size >= len(rows):
        return rows
    ids = sorted(case_id(row) for row in rows)
    chosen = set(random.Random(seed).sample(ids, size))
    return [row for row in rows if case_id(row) in chosen]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sheet", "gate"])
    parser.add_argument("name", help="sheet: answers|team|reports; "
                        "gate: ask|abstention|p1|team|reports|gaps|runtime")
    parser.add_argument("--set", dest="set_name")
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--state", type=Path, default=config.STATE_PATH)
    parser.add_argument("--size", type=int, help="sheet answers: the human sample's size")
    parser.add_argument("--labels", type=Path, help="gate p1|team|reports: your committed labels")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.name != "runtime" and not args.set_name:
        parser.error("--set is required: every sheet and gate is for one set")

    v1 = load_records(args.v1, V1)
    v2 = load_records(args.v2, V2)
    sets = load_manifest(args.manifest)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = read_set(sets[args.set_name]) if args.set_name else []
    if args.set_name:
        require_every_case(v1, v2, args.set_name, rows)
    digest = records_digest(args.v1, args.v2)

    if args.command == "sheet":
        sheet_manifest_path, mapping_path = sheet_paths(args.name, args.set_name)
        if sheet_manifest_path.exists():
            raise SystemExit(f"error: {sheet_manifest_path} exists; re-blinding is a deliberate deletion")
        chosen = sampled(rows, args.size, int(payload["seeds"]["human"])) \
            if args.name == "answers" else rows
        cases = []
        for row in chosen:
            case = case_id(row)
            cases.append((case, row.get("question") or row.get("topic"),
                          v1[(args.set_name, case)], v2[(args.set_name, case)]))
        seed = secrets.token_hex(16)
        items, mapping = build_sheet(args.name, cases, seed)
        mapping_bytes = (json.dumps({"seed": seed, "mapping": mapping}, indent=2) + "\n").encode()
        manifest = sheet_manifest(args.name, mapping_bytes, digest, len(items))
        BLIND_DIR.mkdir(parents=True, exist_ok=True)
        mapping_path.write_bytes(mapping_bytes)
        os.chmod(mapping_path, 0o600)
        sheet_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        sheet_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        out = args.out or Path("evals/results") / f"v2-{args.name}-{args.set_name}-sheet.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_sheet(args.name, items, manifest["storage_key"]), encoding="utf-8")
        print(f"{len(items)} case(s) -> {out}\nmapping -> {mapping_path} (do not open)\n"
              f"commit {sheet_manifest_path} BEFORE scoring")
        return 0

    with CorpusStore(args.state) as corpus:
        passages = Passages(corpus)
        if args.name == "ask":
            result = gate_ask(v1, v2, args.set_name, rows, passages)
        elif args.name == "abstention":
            from verbatim.embedding import FastEmbedEmbedder

            embedder = FastEmbedEmbedder(model_name=config.EMBEDDING_MODEL)
            result = gate_abstention(
                v1, v2, args.set_name, rows, passages,
                lambda question, subject: subject_similarity(embedder, question, subject),
            )
        elif args.name in ("p1", "team", "reports"):
            from breadth_score import committed_before

            kind = {"p1": "answers", "team": "team", "reports": "reports"}[args.name]
            manifest_path, mapping_path = sheet_paths(kind, args.set_name)
            if not args.labels or not committed_before(manifest_path, args.labels):
                raise SystemExit("error: the labels must be committed after the sheet's manifest")
            labels = unblind(json.loads(manifest_path.read_text()), mapping_path.read_bytes(),
                             digest, json.loads(args.labels.read_text()))
            if args.name == "p1":
                result = gate_p1(labels, v1, v2, args.set_name)
            elif args.name == "team":
                result = gate_team(labels, v1, v2, args.set_name)
            else:
                result = gate_reports(labels, v1, v2, args.set_name, passages)
        elif args.name == "gaps":
            result = gate_gaps(v1, v2, args.set_name, passages)
        elif args.name == "runtime":
            result = gate_runtime(list(v2.values()))
        else:
            parser.error(f"unknown gate {args.name}")
    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
