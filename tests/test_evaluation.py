from pathlib import Path

import pytest

from verbatim.corpus import ChunkMatch
from verbatim.chunking import Chunk
from verbatim.evaluation import (
    EvalCase,
    load_cases,
    rank_of_first_hit,
    render_report,
    score_mode,
)


def match(arxiv_id: str) -> ChunkMatch:
    chunk = Chunk(
        chunk_id=f"c-{arxiv_id}", arxiv_id=arxiv_id, source_url="u", license_url=None,
        pdf_sha256="s", page_number=1, chunk_index=0, char_start=0, char_end=1,
        text="t", chunker_version="6:1200:200",
    )
    return ChunkMatch(chunk=chunk, score=1.0, snippet="")


def test_rank_is_the_position_of_the_first_expected_paper() -> None:
    results = [match("A"), match("B"), match("C")]

    assert rank_of_first_hit(results, ["C"]) == 3
    assert rank_of_first_hit(results, ["B", "C"]) == 2
    assert rank_of_first_hit(results, ["Z"]) is None


def test_recall_and_mrr_reward_ranking_the_answer_higher() -> None:
    cases = [EvalCase("q1", ("A",)), EvalCase("q2", ("B",))]
    top = score_mode("top", cases, lambda q: [match("A" if q == "q1" else "B")])
    deep = score_mode(
        "deep", cases, lambda q: [match("X")] * 4 + [match("A" if q == "q1" else "B")]
    )

    assert top.recall_at[1] == 1.0 and top.mrr == 1.0
    assert deep.recall_at[1] == 0.0      # nothing correct at rank 1
    assert deep.recall_at[5] == 1.0      # but found by rank 5
    assert deep.mrr == pytest.approx(0.2)
    assert top.mrr > deep.mrr


def test_a_mode_that_finds_nothing_records_the_misses() -> None:
    cases = [EvalCase("q1", ("A",)), EvalCase("q2", ("B",))]

    score = score_mode("blind", cases, lambda q: [match("Z")])

    assert score.mrr == 0.0
    assert all(value == 0.0 for value in score.recall_at.values())
    assert score.misses == ("q1", "q2")


def test_cases_load_from_jsonl_ignoring_comments(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '# a comment\n\n'
        '{"question": "what is flux?", "expect_papers": ["2608.05440v1"], "note": "n"}\n'
        '{"question": "second", "expect_papers": ["a", "b"]}\n',
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert [c.question for c in cases] == ["what is flux?", "second"]
    assert cases[1].expect_papers == ("a", "b")


def test_a_case_with_no_expected_paper_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"question": "q", "expect_papers": []}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="no expected papers"):
        load_cases(path)


def test_an_empty_answer_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text("# only comments\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no eval cases"):
        load_cases(path)


def test_report_names_the_best_mode() -> None:
    cases = [EvalCase("q", ("A",))]
    good = score_mode("good", cases, lambda q: [match("A")])
    bad = score_mode("bad", cases, lambda q: [match("Z")])

    report = render_report([bad, good])

    assert "best by mrr: good" in report
    assert report.index("bad") < report.index("best by mrr")  # sorted worst first


def test_unanswerable_cases_are_excluded_from_retrieval_scoring() -> None:
    """Scoring recall against a question with no answer would be meaningless."""
    cases = [
        EvalCase("answerable", ("A",)),
        EvalCase("unanswerable", (), expect_none=True),
    ]

    score = score_mode("m", cases, lambda q: [match("A")])

    assert score.cases == 1        # only the answerable one counted
    assert score.recall_at[1] == 1.0
    assert score.mrr == 1.0


def test_a_case_must_declare_either_papers_or_that_none_exist(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"question": "q", "expect_papers": []}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="set expect_none"):
        load_cases(path)


def test_a_case_cannot_both_expect_papers_and_expect_none(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"question": "q", "expect_papers": ["A"], "expect_none": true}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="cannot both"):
        load_cases(path)


def test_an_unanswerable_case_loads(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"question": "q", "expect_none": true}\n', encoding="utf-8")

    (case,) = load_cases(path)

    assert case.expect_none and case.expect_papers == ()
