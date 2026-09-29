"""The breadth evaluation's scoring and blinding rules, with no model at all."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
import breadth_score  # noqa: E402
import breadth_sheet  # noqa: E402


def row(question, arm, citations, require=("2608.00001v1", "2608.00002v2"),
        declined=False, text="An answer [1].", calls=1):
    return {
        "arm": arm, "question": question, "require_papers": list(require), "calls": calls,
        "answer": {
            "question": question, "text": text, "declined": declined,
            "citations": [{"arxiv_id": c, "title": f"Title of {c}", "quoted_text": "a quote"}
                          for c in citations],
            "rejected": [],
        },
    }


def test_the_sign_test_is_exact_and_symmetric() -> None:
    assert breadth_score.sign_test(0, 0) == 1.0
    assert breadth_score.sign_test(9, 1) == pytest.approx(22 / 1024)   # p < 0.05
    assert breadth_score.sign_test(8, 2) == pytest.approx(112 / 1024)  # p > 0.05
    assert breadth_score.sign_test(2, 8) == breadth_score.sign_test(8, 2)


def test_both_papers_counts_across_versions() -> None:
    """A bibliography names a paper, not a version; so does this measure."""
    both = breadth_score.mechanical(row("q", "team", ["2608.00001v1", "2608.00002v1"]))
    one = breadth_score.mechanical(row("q", "team", ["2608.00001v1", "2608.00009v1"]))

    assert both["both_papers"] == 1 and both["distinct_papers"] == 2
    assert one["both_papers"] == 0


def per_question(team_edges: dict[str, tuple[int, int]], n: int = 10) -> dict:
    """n questions where, per measure and rival, the team wins `w` and loses `l`."""
    table = {}
    for i in range(n):
        scores = {arm: {"coverage": 1, "limitations": 1, "both_papers": 1} for arm in breadth_score.ARMS}
        for key, (wins, losses) in team_edges.items():
            measure, rival = key.split(":")
            if i < wins:
                scores[rival][measure] = 0
            elif i < wins + losses:
                scores[rival][measure] = 2
        table[f"q{i}"] = scores
    return table


def test_the_team_earns_its_cost_only_by_beating_both_rivals() -> None:
    wins_both = per_question({"coverage:baseline": (9, 1), "coverage:ladder": (9, 1)})
    wins_one = per_question({"coverage:baseline": (9, 1), "coverage:ladder": (5, 5)})

    assert breadth_score.decide(wins_both)["earns_its_cost"] is True
    assert breadth_score.decide(wins_one)["earns_its_cost"] is False


def test_a_significant_loss_anywhere_vetoes_a_win() -> None:
    """Beating both on coverage does not excuse losing both papers to the ladder."""
    table = per_question({"coverage:baseline": (9, 1), "coverage:ladder": (9, 1),
                          "both_papers:ladder": (1, 9)})

    verdict = breadth_score.decide(table)

    assert verdict["earns_its_cost"] is False
    assert "both_papers vs ladder" in verdict["significantly_worse_on"]


def test_the_sheet_hides_which_arm_wrote_which_answer() -> None:
    question = "how do two approaches to transmon loss compare?"
    rows = {question: {
        "baseline": row(question, "baseline", ["2608.00001v1"]),
        "ladder": row(question, "ladder", ["2608.00002v2"]),
        "team": row(question, "team", [], declined=True,
                    text="The team found no passage it could cite for this question."),
    }}
    cases = [{"question": question, "require_papers": ["2608.00001v1", "2608.00002v2"]}]

    items, mapping = breadth_sheet.build(rows, cases, {}, seed="s")
    again, same = breadth_sheet.build(rows, cases, {}, seed="s")

    assert sorted(mapping[items[0]["id"]].values()) == sorted(breadth_score.ARMS)
    assert mapping == same  # a fixed seed gives a fixed order
    texts = [a["text"] for a in items[0]["answers"]]
    # The team's own decline sentence would unblind the sheet; it is not there.
    assert not any("team found" in t for t in texts)
    assert breadth_sheet.NEUTRAL_DECLINE in texts
    declined = next(a for a in items[0]["answers"] if a["text"] == breadth_sheet.NEUTRAL_DECLINE)
    assert declined["refs"] == []


def test_a_question_missing_an_arm_is_left_off_the_sheet() -> None:
    question = "q"
    rows = {question: {"baseline": row(question, "baseline", []), "team": row(question, "team", [])}}
    items, mapping = breadth_sheet.build(rows, [{"question": question, "require_papers": ["a1v1", "b2v1"]}], {}, "s")

    assert items == [] and mapping == {}


def test_model_text_cannot_break_out_of_the_page() -> None:
    """Answer text is untrusted; a closing script tag in it must stay inert."""
    hostile = "fine </script><script>alert(1)</script>"
    items = [{"id": "x", "question": "q", "papers": [], "answers": [
        {"letter": "A", "text": hostile, "refs": []}]}]

    page = breadth_sheet.render_page(items)

    assert "</script><script>alert" not in page
    assert page.count("</script>") == 1  # only the page's own


def test_unblinding_refuses_a_file_git_does_not_hold(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    labels.write_text("{}")

    assert breadth_score.committed_unchanged(labels) is False


# -- secondary statistics: they describe, and never decide ------------------


def test_an_effect_size_is_the_paired_difference_with_an_interval() -> None:
    """Better on 9 questions by one point, worse on 1 by one: +0.8 on average."""
    table = per_question({"coverage:baseline": (9, 1)})

    effects = breadth_score.effect_sizes(table)

    moved = effects["coverage"]["baseline"]
    assert moved["mean_difference"] == pytest.approx(0.8)
    low, high = moved["ci95"]
    assert low <= 0.8 <= high and low > 0
    # Nothing differs against the ladder, so the interval is exactly zero.
    assert effects["coverage"]["ladder"] == {"mean_difference": 0.0, "ci95": [0.0, 0.0]}


def test_secondary_statistics_never_change_the_verdict() -> None:
    """The preregistered rule is decide(); effect sizes and agreement sit
    beside it and must not be able to move it."""
    table = per_question({"coverage:baseline": (9, 1), "coverage:ladder": (9, 1)})
    before = breadth_score.decide(table)

    breadth_score.effect_sizes(table)

    assert breadth_score.decide(table) == before


def test_weighted_kappa_at_its_known_points() -> None:
    kappa = breadth_score.weighted_kappa
    assert kappa([0, 1, 2, 2, 1], [0, 1, 2, 2, 1]) == pytest.approx(1.0)
    assert kappa([0, 2], [2, 0]) == pytest.approx(-1.0)
    # Neither rater ever varied: agreement beyond chance is undefined, not 1.
    assert kappa([1, 1, 1], [1, 1, 1]) is None


def test_agreement_uses_only_answers_both_raters_scored() -> None:
    first = {"q1": {"A": {"coverage": 2, "balance": 1, "limitations": 0},
                    "B": {"coverage": 1, "balance": 1, "limitations": 1}}}
    second = {"q1": {"A": {"coverage": 2, "balance": 0, "limitations": 0}}}

    agreed = breadth_score.agreement(first, second)

    assert agreed["coverage"]["pairs"] == 1
    assert agreed["coverage"]["exact"] == pytest.approx(1.0)
    assert agreed["balance"]["exact"] == pytest.approx(0.0)


# -- blinding that can be audited ------------------------------------------


def test_each_blinding_draws_a_seed_nobody_can_read_from_the_code() -> None:
    first, second = breadth_sheet.new_seed(), breadth_sheet.new_seed()

    assert first != second and len(first) >= 32
    assert not hasattr(breadth_sheet, "SEED")


def test_the_manifest_commits_to_the_mapping_and_the_answers() -> None:
    import hashlib

    manifest = breadth_sheet.blind_manifest(b"the mapping", b"the answers", questions=24)

    mapping_hash = hashlib.sha256(b"the mapping").hexdigest()
    assert manifest["mapping_sha256"] == mapping_hash
    assert manifest["answers_sha256"] == hashlib.sha256(b"the answers").hexdigest()
    assert manifest["storage_key"].endswith(mapping_hash[:12])
    assert "fixed seed" in manifest["amendment"]


def test_the_sheet_saves_scores_under_its_own_blinding() -> None:
    """A re-blinded sheet must not pick up scores saved against the old one."""
    items = [{"id": "x", "question": "q", "papers": [], "answers": []}]

    page = breadth_sheet.render_page(items, storage_key="breadth-labels-abc123")

    assert '"breadth-labels-abc123"' in page
    assert "breadth-labels-v1" not in page


def test_downloaded_scores_name_the_sheet_they_were_made_on() -> None:
    items = [{"id": "x", "question": "q", "papers": [], "answers": []}]

    page = breadth_sheet.render_page(items, storage_key="breadth-labels-abc123")

    assert "JSON.stringify({blinding: KEY, ...labels}" in page


def test_scores_from_another_blindings_sheet_are_refused() -> None:
    """An older sheet orders the answers differently, so its scores unblind to nonsense."""
    manifest = {"storage_key": "breadth-labels-new"}
    scores = {"q1": {"A": {"coverage": 2}}}

    kept = breadth_score.scores_for_this_blinding(
        {"blinding": "breadth-labels-new", **scores}, manifest
    )

    assert kept == scores
    for stale in ({"blinding": "breadth-labels-old", **scores}, scores):
        with pytest.raises(SystemExit, match="sheet"):
            breadth_score.scores_for_this_blinding(stale, manifest)


def test_scoring_refuses_a_mapping_or_answers_other_than_the_committed_ones() -> None:
    import json

    mapping = json.dumps({"seed": "s", "mapping": {"q1": {"A": "team"}}}).encode()
    answers = b'{"arm": "team"}\n'
    manifest = breadth_sheet.blind_manifest(mapping, answers, questions=1)

    assert breadth_score.verify_blinding(manifest, mapping, answers) == {"q1": {"A": "team"}}
    swapped = json.dumps({"seed": "s", "mapping": {"q1": {"A": "baseline"}}}).encode()
    with pytest.raises(SystemExit, match="mapping"):
        breadth_score.verify_blinding(manifest, swapped, answers)
    with pytest.raises(SystemExit, match="answers"):
        breadth_score.verify_blinding(manifest, mapping, answers + b"edited\n")


def test_scores_must_be_committed_after_the_manifest(tmp_path: Path, monkeypatch) -> None:
    """Committed together, or the manifest touched afterwards, proves nothing."""
    import subprocess

    monkeypatch.chdir(tmp_path)
    git = lambda *a: subprocess.run(["git", *a], check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    manifest, labels = Path("manifest.json"), Path("labels.json")

    manifest.write_text("{}")
    labels.write_text("{}")
    git("add", "-A")
    git("commit", "-qm", "both at once")
    assert breadth_score.committed_before(manifest, labels) is False

    labels.write_text('{"scored": 1}')
    git("commit", "-qam", "scores after the manifest")
    assert breadth_score.committed_before(manifest, labels) is True

    manifest.write_text('{"changed": 1}')
    git("commit", "-qam", "manifest touched after scoring")
    assert breadth_score.committed_before(manifest, labels) is False
