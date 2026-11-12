"""The v2 scorer believes the store, not the arms, and applies the pre-registered rules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
import arm_records  # noqa: E402
import score_arms  # noqa: E402

from verbatim.corpus import CorpusStore  # noqa: E402

from helpers import chunked_corpus  # noqa: E402

TEXT = ("Transmon qubits lose coherence through dielectric loss in the substrate. "
        "Surface oxides add two-level systems that couple to the qubit.")


@pytest.fixture
def store(tmp_path):
    with chunked_corpus(tmp_path, {"2608.11111v1": [TEXT]}):
        pass
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        chunk = corpus.chunks_for_paper("2608.11111v1")[0]
        yield score_arms.Passages(corpus), chunk.chunk_id


def answer(text: str, *citations: tuple[int, str, str], rejected: int = 0) -> dict:
    return {
        "question": "q", "text": text, "declined": False,
        "citations": [{"marker": m, "chunk_id": c, "arxiv_id": "2608.11111v1", "quote": q}
                      for m, c, q in citations],
        "rejected": [{"chunk_id": "x", "quote": "y", "reason": "z"}] * rejected,
    }


def row(arm: str, case: str, body: dict | None = None, *, status: str = "complete",
        set_name: str = "test", **extra) -> dict:
    return {"set": set_name, "case": case, "arm": arm, "order": 0, "status": status,
            "error_kind": None, "error": None, "model": "m", "digest": None,
            "calls": 2, "tokens_in": 1, "tokens_out": 1, "model_seconds": 0.0,
            "seconds": 1.0, "answer": body, **extra}


def test_a_quote_counts_only_if_the_store_holds_it(store) -> None:
    passages, chunk = store
    honest = answer("Transmons lose coherence through the substrate itself [1].",
                    (1, chunk, "lose coherence through dielectric loss"))
    claimed = answer("Transmons keep coherence for an hour in any lab [1].",
                     (1, chunk, "keep coherence for an hour"))
    assert score_arms.check(honest, passages).verified == {1}
    # An arm's word that a citation verified is worth nothing: the scorer rechecks.
    faked = score_arms.check(claimed, passages)
    assert not faked.verified and faked.failed == 1 and not faked.grounding


def test_the_cited_paper_comes_from_the_store_not_from_the_record(store) -> None:
    # P3 credits the paper the store files the passage under, whatever the arm wrote.
    passages, chunk = store
    mislabelled = answer("Transmon qubits lose coherence in the substrate [1].",
                         (1, chunk, "Transmon qubits lose coherence"))
    mislabelled["citations"][0]["arxiv_id"] = "2608.99999v1"
    assert score_arms.check(mislabelled, passages).papers == frozenset({"2608.11111v1"})


def test_grounding_needs_every_claim_sentence_and_one_verified_citation(store) -> None:
    passages, chunk = store
    quote = (1, chunk, "Transmon qubits lose coherence")
    full = answer("Transmon qubits lose coherence in the substrate [1].", quote)
    partial = answer("Transmon qubits lose coherence in the substrate [1]. "
                     "Fluxonium qubits are always better than transmons.", quote)
    unmarked = answer("Transmon qubits lose coherence in the substrate.", quote)
    decline = {"question": "q", "text": "No.", "declined": True, "citations": [], "rejected": []}
    assert score_arms.check(full, passages).grounding
    assert not score_arms.check(partial, passages).grounding
    # A verified citation whose marker is not in the text supports nothing.
    assert not score_arms.check(unmarked, passages).verified
    # A decline has no claim sentence to fail, which is why P2 also needs a citation.
    assert score_arms.check(decline, passages).claims == 0
    assert not score_arms.check(decline, passages).grounding


def test_a_failed_run_scores_zero(store) -> None:
    passages, chunk = store
    good = answer("Transmon qubits lose coherence in the substrate [1].",
                  (1, chunk, "Transmon qubits lose coherence"))
    assert score_arms.checked(row("v2", "a", good, status="error"), passages) == score_arms.Checked()


def test_non_inferiority_passes_an_equal_arm_and_fails_a_worse_one() -> None:
    quick = {"seeds": 3, "rounds": 400}
    same = [True, False] * 50
    equal = score_arms.noninferior("p2", same, same, 0.05, **quick)
    worse = score_arms.noninferior("p2", [True] * 100, [True] * 80 + [False] * 20, 0.05, **quick)
    assert equal.passed and equal.low == 0
    assert not worse.passed and worse.losses == 20 and worse.wins == 0


def test_holm_adjusts_in_step_down_order() -> None:
    adjusted = score_arms.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_the_last_record_stands_and_a_broken_one_must_be_rerun(tmp_path) -> None:
    path = tmp_path / "v2.jsonl"
    first = row("v2", "a", None, status="error")
    path.write_text(json.dumps(first) + "\n" + json.dumps(row("v2", "a", None)) + "\n")
    assert score_arms.load_records(path, "v2")[("test", "a")]["status"] == "complete"
    broken = {**first, "error_kind": arm_records.INFRASTRUCTURE}
    path.write_text(json.dumps(broken) + "\n")
    with pytest.raises(SystemExit, match="rerun"):
        score_arms.load_records(path, "v2")
    with pytest.raises(ValueError):
        score_arms.load_records(path, "v1")


def test_a_case_only_one_arm_ran_is_refused() -> None:
    v1 = {("test", "a"): row("v1", "a"), ("test", "b"): row("v1", "b")}
    v2 = {("test", "a"): row("v2", "a")}
    with pytest.raises(SystemExit, match="lack one arm"):
        score_arms.pairs(v1, v2, "test")


def test_declining_or_warning_is_recomputed_from_the_subject(store) -> None:
    passages, chunk = store
    cited = answer("Transmon qubits lose coherence in the substrate [1].",
                   (1, chunk, "Transmon qubits lose coherence"))
    far = lambda question, subject: 0.1  # noqa: E731
    near = lambda question, subject: 0.9  # noqa: E731
    with_subject = row("v2", "a", cited, subject="qubits")
    assert score_arms.declines_or_warns(with_subject, score_arms.check(cited, passages), "q", far)
    assert not score_arms.declines_or_warns(
        with_subject, score_arms.check(cited, passages), "q", near)
    assert score_arms.declines_or_warns(row("v2", "a", None), score_arms.Checked(), "q", near)
    assert not score_arms.declines_or_warns(
        row("v2", "a", None, status="error"), score_arms.Checked(), "q", near)


def test_a_refutation_counts_only_with_a_quote_the_store_holds(store) -> None:
    passages, chunk = store

    def gap(arm, case, visible_quote):
        finding = {"verdict": "refuted", "citation": {
            "chunk_id": chunk, "arxiv_id": "2608.11111v1", "quote": visible_quote}}
        return row(arm, case, None, set_name="claims", visible=finding,
                   hidden={"verdict": "stands", "citation": None})

    v1 = {("claims", c): gap("v1", c, "Transmon qubits lose coherence") for c in "abcdef"}
    v2 = {("claims", c): gap("v2", c, "a quote no passage holds") for c in "abcdef"}
    result = score_arms.gate_gaps(v1, v2, "claims", passages)
    assert result["v2_misses"] == 6 and result["v2_gains"] == 0 and not result["passed"]
    fewer = {key: (v2[key] if key[1] in "ab" else v1[key]) for key in v1}
    fewer = {key: {**value, "arm": "v2"} for key, value in fewer.items()}
    assert score_arms.gate_gaps(v1, fewer, "claims", passages)["passed"]


def test_the_blind_count_allows_four_more_losses_than_wins() -> None:
    def scores(first, second):
        return {"v1": {"coverage": first, "balance": 0, "limitations": 0},
                "v2": {"coverage": second, "balance": 0, "limitations": 0}}
    four = {str(i): scores(2, 1) for i in range(4)}
    five = {str(i): scores(2, 1) for i in range(5)}
    assert score_arms.blind_count(four, "team")["passed"]
    assert not score_arms.blind_count(five, "team")["passed"]
    five["tie"] = scores(1, 1)
    five["win"] = scores(0, 2)
    assert score_arms.blind_count(five, "team")["passed"]


def test_a_blind_sheet_hides_the_arms_and_unblinds_only_with_its_own_mapping(store) -> None:
    passages, chunk = store
    cited = answer("Transmon qubits lose coherence in the substrate [1].",
                   (1, chunk, "Transmon qubits lose coherence"))
    declined = {**cited, "declined": True}
    cases = [(f"c{i}", f"question {i}", row("v1", f"c{i}", cited), row("v2", f"c{i}", declined))
             for i in range(6)]
    items, mapping = score_arms.build_sheet("answers", cases, seed="s")
    assert {letters["A"] for letters in mapping.values()} == {"v1", "v2"}
    # Arm names appear nowhere as values ("2608.11111v1" is a paper id, not an arm).
    assert all('"v1"' not in json.dumps(item) and '"v2"' not in json.dumps(item)
               for item in items)
    texts = {a["text"] for item in items for a in item["answers"]}
    assert score_arms.NEUTRAL_DECLINE in texts

    mapping_bytes = json.dumps({"seed": "s", "mapping": mapping}).encode()
    manifest = score_arms.sheet_manifest("answers", mapping_bytes, "records-hash", len(items))
    labels = {"blinding": manifest["storage_key"],
              **{case: {"A": {"correct": 2}, "B": {"correct": 0}} for case in mapping}}
    unblinded = score_arms.unblind(manifest, mapping_bytes, "records-hash", labels)
    for case, letters in mapping.items():
        assert unblinded[case][letters["A"]]["correct"] == 2
    with pytest.raises(SystemExit, match="records changed"):
        score_arms.unblind(manifest, mapping_bytes, "other", labels)
    with pytest.raises(SystemExit, match="another sheet"):
        score_arms.unblind(manifest, mapping_bytes, "records-hash",
                           {**labels, "blinding": "old"})
    page = score_arms.render_sheet("answers", items, manifest["storage_key"])
    assert "Correct and supported" in page and "<\\/" not in json.dumps(items)


def test_unblinding_refuses_labels_that_leave_an_answer_unscored() -> None:
    # A skipped case would otherwise drop out of P1 and the blind counts unnoticed.
    mapping = {f"c{i}": {"A": "v1", "B": "v2"} for i in range(3)}
    mapping_bytes = json.dumps({"seed": "s", "mapping": mapping}).encode()
    manifest = score_arms.sheet_manifest("team", mapping_bytes, "records-hash", 3)
    full = {"coverage": 2, "balance": 1, "limitations": 0}
    labels = {"blinding": manifest["storage_key"],
              **{case: {"A": dict(full), "B": dict(full)} for case in mapping}}
    assert len(score_arms.unblind(manifest, mapping_bytes, "records-hash", labels)) == 3

    skipped = {key: value for key, value in labels.items() if key != "c1"}
    partial = json.loads(json.dumps(labels))
    del partial["c2"]["B"]["balance"]
    out_of_range = json.loads(json.dumps(labels))
    out_of_range["c0"]["A"]["coverage"] = 3
    for broken in (skipped, partial, out_of_range):
        with pytest.raises(SystemExit, match="1 of the sheet's 3 cases"):
            score_arms.unblind(manifest, mapping_bytes, "records-hash", broken)


def test_each_set_gets_its_own_blind_sheet(tmp_path, monkeypatch) -> None:
    # The sealed second evaluation must not collide with the first one's committed sheet.
    monkeypatch.setattr(score_arms, "SHEET_MANIFESTS", tmp_path / "blind")
    monkeypatch.setattr(score_arms, "BLIND_DIR", tmp_path / "sealed")
    sets, records = {}, {"v1": [], "v2": []}
    for name in ("test-a", "test-b"):
        rows = [{"question": f"{name} question {i}", "expect_papers": ["2608.11111v1"]}
                for i in range(3)]
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(item) + "\n" for item in rows))
        sets[name] = {"path": path.name, "kind": "questions",
                      "sha256": arm_records.sha256_of(path.read_bytes())}
        for arm in records:
            records[arm] += [row(arm, arm_records.case_id(item), answer("An answer."),
                                 set_name=name) for item in rows]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sets": sets, "seeds": {"human": 7}}))
    for arm, rows in records.items():
        (tmp_path / f"{arm}.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rows))
    common = ["--v1", str(tmp_path / "v1.jsonl"), "--v2", str(tmp_path / "v2.jsonl"),
              "--manifest", str(manifest)]

    def sheet(set_name: str) -> int:
        return score_arms.main(["sheet", "answers", "--set", set_name, *common,
                                "--out", str(tmp_path / f"{set_name}.html")])

    assert sheet("test-a") == 0 and sheet("test-b") == 0
    assert sorted(p.name for p in (tmp_path / "blind").iterdir()) == [
        "answers-test-a.json", "answers-test-b.json"]
    assert sorted(p.name for p in (tmp_path / "sealed").iterdir()) == [
        "v2-answers-test-a-mapping.json", "v2-answers-test-b-mapping.json"]
    with pytest.raises(SystemExit, match="exists"):
        sheet("test-a")
    with pytest.raises(SystemExit):
        score_arms.main(["sheet", "answers", *common])


def test_nothing_is_scored_or_blinded_until_both_arms_ran_the_whole_set(tmp_path, monkeypatch) -> None:
    # Scoring a run that is still going would peek, and would score fewer cases than were sized.
    monkeypatch.setattr(score_arms, "SHEET_MANIFESTS", tmp_path / "blind")
    monkeypatch.setattr(score_arms, "BLIND_DIR", tmp_path / "sealed")
    items = [{"question": f"question {i}", "expect_papers": ["2608.11111v1"]} for i in range(3)]
    path = tmp_path / "test-a.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in items))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sets": {"test-a": {
        "path": path.name, "kind": "questions", "sha256": arm_records.sha256_of(path.read_bytes()),
    }}, "seeds": {"human": 7}}))

    def run(command: list[str], ran: dict[str, list[dict]]) -> int:
        for arm, cases in ran.items():
            (tmp_path / f"{arm}.jsonl").write_text("".join(
                json.dumps(row(arm, arm_records.case_id(item), answer("An answer."),
                               set_name="test-a")) + "\n" for item in cases))
        return score_arms.main([*command, "--set", "test-a", "--v1", str(tmp_path / "v1.jsonl"),
                                "--v2", str(tmp_path / "v2.jsonl"), "--manifest", str(manifest),
                                "--state", str(tmp_path / "absent.db"),
                                "--out", str(tmp_path / "out.html")])

    # Both arms agree on what they ran, so pairing alone would not notice the missing case.
    for command in (["sheet", "answers"], ["gate", "ask"], ["gate", "abstention"]):
        with pytest.raises(SystemExit, match="2 of test-a's 3 cases"):
            run(command, {"v1": items[:2], "v2": items[:2]})
    with pytest.raises(SystemExit, match="1 outside it"):
        run(["gate", "ask"], {"v1": items, "v2": [*items, {"question": "from another set"}]})
    # Refused before the store is opened or anything is blinded.
    assert not (tmp_path / "absent.db").exists() and not (tmp_path / "blind").exists()


def test_the_abstention_result_says_which_cases_hid_a_paper(store) -> None:
    # The written unanswerable cases hide nothing, so they keep the title and graph legs.
    passages, _ = store
    hidden = [{"question": f"q{i}", "expect_papers": ["2608.11111v1"], "hide": ["2608.11111v1"]}
              for i in range(2)]
    written = [{"question": "q2", "expect_none": True}]

    def note(rows: list[dict]) -> str:
        records = {arm: {("abstention-a", arm_records.case_id(item)): row(
            arm, arm_records.case_id(item), None, set_name="abstention-a") for item in rows}
            for arm in ("v1", "v2")}
        return score_arms.gate_abstention(records["v1"], records["v2"], "abstention-a", rows,
                                          passages, lambda question, subject: 0.0,
                                          seeds=1, rounds=100)["note"]

    assert note(hidden) == ("2 of 2 cases hide their answering paper, so the title and graph "
                            "legs are off for them in both arms")
    assert note(hidden + written) == (
        "2 of 3 cases hide their answering paper, so the title and graph legs are off for "
        "them in both arms; the other 1 hide nothing and run with the defaults")


def test_the_runtime_gate_allows_five_percent_protocol_failures() -> None:
    # Every gate runs on the local model, so the threshold set for it applies.
    rows = [row("v2", str(i)) for i in range(94)] + [
        row("v2", f"p{i}", status="incomplete_protocol_failure") for i in range(5)] + [
        row("v2", "b", status="budget_exhausted")]
    runtime = score_arms.gate_runtime(rows)
    assert runtime["protocol_failures"] == pytest.approx(0.05) and runtime["passed"]
    rows.append(row("v2", "p5", status="incomplete_protocol_failure"))
    assert not score_arms.gate_runtime(rows)["passed"]


def test_v2_may_take_three_times_v1s_median_time(store) -> None:
    passages, chunk = store
    cited = answer("Transmon qubits lose coherence in the substrate [1].",
                   (1, chunk, "Transmon qubits lose coherence"))
    rows = [{"question": f"question {i}", "expect_papers": ["2608.11111v1"]} for i in range(4)]
    cases = [arm_records.case_id(item) for item in rows]
    v1 = {("test", c): row("v1", c, cited) for c in cases}

    def latency(seconds: float) -> dict:
        v2 = {("test", c): row("v2", c, cited, seconds=seconds) for c in cases}
        result = score_arms.gate_ask(v1, v2, "test", rows, passages, seeds=2, rounds=200)
        return result["cost"]

    assert latency(3.0) == {"calls_ratio": 1.0, "calls_passed": True,
                            "latency_ratio": 3.0, "latency_passed": True}
    assert not latency(3.1)["latency_passed"]


def test_the_human_sample_is_drawn_with_the_frozen_seed() -> None:
    rows = [{"question": f"question number {i}"} for i in range(50)]
    first = score_arms.sampled(rows, 10, seed=7)
    assert len(first) == 10 and first == score_arms.sampled(rows, 10, seed=7)
    assert first != score_arms.sampled(rows, 10, seed=8)


def test_records_digest_covers_both_arms(tmp_path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("x")
    b.write_text("y")
    assert score_arms.records_digest(a, b) == hashlib.sha256(b"xy").hexdigest()
