import collections
from pathlib import Path

import pytest

from verbatim.corpus import ChunkMatch
from verbatim.chunking import Chunk
from verbatim.evaluation import (
    EvalCase,
    group_rank,
    load_cases,
    rank_of_first_hit,
    render_compound_report,
    render_report,
    score_compound_mode,
    score_mode,
)


def match(arxiv_id: str) -> ChunkMatch:
    chunk = Chunk(
        chunk_id=f"c-{arxiv_id}", arxiv_id=arxiv_id, source_url="u", license_url=None,
        source_sha256="s", page_number=1, chunk_index=0, char_start=0, char_end=1,
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


def test_a_compound_case_loads_and_a_half_specified_one_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"question": "compare A and B", "require_papers": ["A", "B1|B2"]}\n'
    )
    case = load_cases(path)[0]
    assert case.is_compound and case.require_papers == ("A", "B1|B2")

    for bad in (
        '{"question": "q", "require_papers": ["A"]}',
        '{"question": "q", "require_papers": ["A", "B"], "expect_papers": ["C"]}',
        '{"question": "q", "require_papers": ["A", "B"], "expect_none": true}',
    ):
        path.write_text(bad + "\n")
        with pytest.raises(ValueError):
            load_cases(path)


def test_group_rank_accepts_any_member_of_an_alternative_group() -> None:
    results = [match("X"), match("B2"), match("A")]

    assert group_rank(results, "A") == 3
    assert group_rank(results, "B1|B2") == 2
    assert group_rank(results, "Z1|Z2") is None


def test_compound_scoring_counts_parts_covered_not_first_hits() -> None:
    cases = [EvalCase("compare A and B", require_papers=("A", "B"))]
    # A at rank 1, B at rank 7: perfect by first-hit, half an answer at k=5.
    deep = score_compound_mode(
        "deep", cases, lambda q: [match("A")] + [match("X")] * 5 + [match("B")]
    )
    blind = score_compound_mode("blind", cases, lambda q: [match("A")] * 10)

    assert deep.coverage_at[5] == 0.5 and deep.complete_at[5] == 0.0
    assert deep.coverage_at[10] == 1.0 and deep.complete_at[10] == 1.0
    assert deep.incomplete == ()
    assert blind.coverage_at[10] == 0.5 and blind.complete_at[10] == 0.0
    assert blind.incomplete == ("compare A and B",)


def test_compound_cases_stay_out_of_first_hit_scoring() -> None:
    cases = [
        EvalCase("q1", ("A",)),
        EvalCase("compare A and B", require_papers=("A", "B")),
    ]
    score = score_mode("top", cases, lambda q: [match("A")])

    # Were the compound case counted, ten passages about A alone would score
    # a perfect hit on a question they answer half of.
    assert score.cases == 1 and score.mrr == 1.0


def test_compound_report_names_the_best_mode_by_completion() -> None:
    cases = [EvalCase("compare A and B", require_papers=("A", "B"))]
    both = score_compound_mode("both", cases, lambda q: [match("A"), match("B")])
    half = score_compound_mode("half", cases, lambda q: [match("A")] * 10)

    report = render_compound_report([half, both])

    assert "best by all@10: both" in report
    assert "did not fully cover" not in report.split("best")[0]


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
    path.write_text(
        '{"question": "q", "expect_none": true, "unanswerable_kind": "no_coverage"}\n',
        encoding="utf-8",
    )

    (case,) = load_cases(path)

    assert case.expect_none and case.expect_papers == ()
    assert case.unanswerable_kind == "no_coverage"


def test_an_unanswerable_case_must_say_which_kind_it_is(tmp_path: Path) -> None:
    """The two kinds are different problems and only one is detectable."""
    path = tmp_path / "cases.jsonl"
    path.write_text('{"question": "q", "expect_none": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="must say why it is unanswerable"):
        load_cases(path)


def test_a_covered_but_unanswered_case_is_not_a_coverage_gap(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"question": "what is a tensor?", "expect_none": true,'
        ' "unanswerable_kind": "no_answer"}\n',
        encoding="utf-8",
    )

    (case,) = load_cases(path)

    assert case.expect_none and case.unanswerable_kind == "no_answer"


def test_the_report_separates_the_two_unanswerable_kinds() -> None:
    """A single count hides which problem a change actually moved."""
    from verbatim.evaluation import EvalCase, describe_case_mix

    mix = describe_case_mix([
        EvalCase(question="a", expect_papers=("2608.11111v1",)),
        EvalCase(question="b", expect_none=True, unanswerable_kind="no_coverage"),
        EvalCase(question="c", expect_none=True, unanswerable_kind="no_coverage"),
        EvalCase(question="d", expect_none=True, unanswerable_kind="no_answer"),
    ])

    assert "4 cases" in mix
    assert "1 answerable" in mix
    assert "2 with no coverage" in mix
    assert "1 covered but unanswered" in mix


def test_load_key_reads_several_files_as_one(tmp_path: Path) -> None:
    """Two keys authored differently, scored together but separable."""
    from verbatim.evaluation import load_key, source_of

    hand = tmp_path / "heldout.jsonl"
    hand.write_text(
        '# a header\n'
        '{"question": "why do qubits decohere", "expect_papers": ["a1v1"],'
        ' "note": "source=heldout; shape=plain"}\n'
    )
    made = tmp_path / "generated.jsonl"
    made.write_text(
        '{"question": "what limits coherence time", "expect_papers": ["b2v1"],'
        ' "note": "source=heldout-generated; shape=plain; shared_run=2"}\n'
    )

    cases = load_key([hand, made])

    assert len(cases) == 2
    assert [source_of(c) for c in cases] == ["heldout", "heldout-generated"]


def test_a_question_in_two_keys_is_refused(tmp_path: Path) -> None:
    """Scoring a case twice weights it twice and distorts every mean."""
    from verbatim.evaluation import load_key

    first = tmp_path / "one.jsonl"
    first.write_text('{"question": "same", "expect_papers": ["a1v1"]}\n')
    second = tmp_path / "two.jsonl"
    second.write_text('{"question": "same", "expect_papers": ["b2v1"]}\n')

    with pytest.raises(ValueError, match="already in"):
        load_key([first, second])


def test_source_of_falls_back_when_a_case_is_unlabelled(tmp_path: Path) -> None:
    from verbatim.evaluation import EvalCase, source_of

    assert source_of(EvalCase(question="q", expect_papers=("a",))) == "unknown"
    assert source_of(
        EvalCase(question="q", expect_papers=("a",), note="shape=plain")
    ) == "unknown"


def test_stratified_sample_draws_evenly_across_batches(tmp_path: Path) -> None:
    """File order would let the biggest batch decide the result."""
    from verbatim.evaluation import EvalCase, source_of, stratified_sample

    cases = (
        [EvalCase(question=f"small {n}", expect_papers=("a",), note="source=small")
         for n in range(5)]
        + [EvalCase(question=f"big {n}", expect_papers=("b",), note="source=big")
           for n in range(200)]
    )

    drawn = stratified_sample(cases, size=8)

    counts = collections.Counter(source_of(c) for c in drawn)
    assert counts["small"] == 4 and counts["big"] == 4


def test_stratified_sample_exhausts_a_small_batch_before_padding(
    tmp_path: Path,
) -> None:
    """A batch smaller than its share contributes all it has, not a crash."""
    from verbatim.evaluation import EvalCase, source_of, stratified_sample

    cases = (
        [EvalCase(question="only one", expect_papers=("a",), note="source=tiny")]
        + [EvalCase(question=f"big {n}", expect_papers=("b",), note="source=big")
           for n in range(50)]
    )

    drawn = stratified_sample(cases, size=10)

    counts = collections.Counter(source_of(c) for c in drawn)
    assert len(drawn) == 10
    assert counts["tiny"] == 1 and counts["big"] == 9


def test_stratified_sample_is_reproducible(tmp_path: Path) -> None:
    """The same sample must be drawable twice, to score it twice."""
    from verbatim.evaluation import EvalCase, stratified_sample

    cases = [EvalCase(question=f"q {n}", expect_papers=("a",), note="source=s")
             for n in range(40)]

    first = [c.question for c in stratified_sample(cases, size=10, seed=3)]
    again = [c.question for c in stratified_sample(cases, size=10, seed=3)]
    other = [c.question for c in stratified_sample(cases, size=10, seed=4)]

    assert first == again
    assert first != other
