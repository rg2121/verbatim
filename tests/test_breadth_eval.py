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
