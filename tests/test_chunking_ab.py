"""The chunking experiment's guards, with no model and no corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
import chunking_ab  # noqa: E402


def test_every_paper_a_key_names_is_held_including_compound_alternatives(tmp_path: Path) -> None:
    key = tmp_path / "key.jsonl"
    key.write_text("\n".join(json.dumps(case) for case in [
        {"question": "what is flux?", "expect_papers": ["2608.11111v1"]},
        {"question": "compare A and B", "require_papers": ["2608.22222v1|2608.33333v1", "2608.44444v1"]},
    ]) + "\n")

    assert chunking_ab.named_papers([key]) == [
        "2608.11111v1", "2608.22222v1", "2608.33333v1", "2608.44444v1",
    ]


@pytest.mark.parametrize(("arm", "chunkers"), [
    ("sentences", ["8:1200:200"]),
    ("sentences", ["8:1200:200:sentences", "8:1200:200"]),
    ("windows", ["8:1200:200:semantic"]),
    ("semantic", []),
])
def test_an_arm_not_yet_cut_its_way_is_refused(arm: str, chunkers: list[str]) -> None:
    """Scoring a copy before `chunk` ran would report windows under another name."""
    with pytest.raises(SystemExit, match="not built"):
        chunking_ab.check_built(arm, chunkers)


def test_an_arm_cut_its_way_is_scored() -> None:
    chunking_ab.check_built("windows", ["8:1200:200"])
    chunking_ab.check_built("semantic", ["8:1200:200:semantic"])


def test_arms_holding_different_papers_are_not_compared() -> None:
    rows = [{"question": "q", "source": "hand", "rank": 1, "seconds": 0.1}]
    first = {"arm": "windows", "papers_sha": "a", "rows": rows}
    second = {"arm": "sentences", "papers_sha": "b", "rows": rows}

    with pytest.raises(SystemExit, match="different papers"):
        chunking_ab.report(first, second)


def test_a_passage_ends_a_sentence_even_inside_a_bracket() -> None:
    assert chunking_ab.ends_a_sentence("It decoheres (as shown).")
    assert chunking_ab.ends_a_sentence('He said "stop."')
    assert not chunking_ab.ends_a_sentence("It decoheres because the")
