"""The retrieval A/B harness and the statistics every eval script shares, with no model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
import retrieval_ab  # noqa: E402
import significance  # noqa: E402


def test_the_sign_test_is_exact_and_symmetric() -> None:
    assert significance.sign_test(0, 0) == 1.0
    assert significance.sign_test(9, 1) == pytest.approx(22 / 1024)
    assert significance.sign_test(2, 8) == significance.sign_test(8, 2)


def test_a_paired_sign_test_ignores_ties() -> None:
    assert significance.paired_sign_test([0.5, -0.25, 0.0, 0.0, 1.0]) == (
        2, 1, pytest.approx(1.0)
    )


def test_the_bootstrap_reports_the_share_at_or_below_zero() -> None:
    """The share, not the interval's edge, decides: the edge can sit on zero."""
    low, high, at_or_below = significance.bootstrap_ci([1.0] * 9 + [-1.0])

    assert low <= 0.8 <= high
    assert 0 < at_or_below < 0.05
    assert significance.bootstrap_ci([]) == (0.0, 0.0, 1.0)


def test_the_eval_scripts_share_one_sign_test() -> None:
    """Five copies of one formula is how a fix reaches four of them."""
    import conversation_arms
    import paired_bootstrap
    import tables_ab
    import three_arm

    assert tables_ab.sign_test is significance.sign_test
    assert conversation_arms.sign_test is significance.sign_test
    assert paired_bootstrap.sign_test is significance.paired_sign_test
    assert three_arm.sign_test is significance.paired_sign_test
    assert paired_bootstrap.ci is significance.bootstrap_ci
    assert three_arm.ci is significance.bootstrap_ci


def rows(ranks: list[int | None], prefix: str = "q") -> list[dict]:
    return [{"question": f"{prefix}{n}", "source": "hand", "rank": rank, "seconds": 0.1}
            for n, rank in enumerate(ranks)]


def test_an_arm_is_summarised_by_recall_and_mrr() -> None:
    summary = retrieval_ab.summarise(rows([1, 2, None, 6]))

    assert summary["cases"] == 4
    assert summary["r@1"] == pytest.approx(0.25)
    assert summary["r@5"] == pytest.approx(0.5)
    assert summary["r@10"] == pytest.approx(0.75)
    assert summary["mrr"] == pytest.approx((1 + 0.5 + 0 + 1 / 6) / 4)


def test_two_arms_are_compared_case_by_case() -> None:
    before = rows([1, 2, None, 3])
    after = rows([1, 1, 1, 4])

    verdict = retrieval_ab.compare(before, after)

    assert (verdict["wins"], verdict["losses"]) == (2, 1)
    assert verdict["mean_difference"] == pytest.approx(
        (0 + 0.5 + 1 + (0.25 - 1 / 3)) / 4
    )
    assert 0 <= verdict["at_or_below_zero"] <= 1


def test_arms_run_on_different_cases_are_not_compared() -> None:
    with pytest.raises(SystemExit, match="different cases"):
        retrieval_ab.compare(rows([1, 2]), rows([1, 2], prefix="other"))


def test_an_arm_ranks_each_case_and_times_it(tmp_path: Path) -> None:
    from helpers import chunked_corpus
    from verbatim.corpus import EmbeddingModel
    from verbatim.evaluation import EvalCase

    class KeywordOnly:
        model = EmbeddingModel(model_id="stub", dimensions=4, normalized=True, revision="v0")

        def embed_query(self, text):
            return [0.0, 0.0, 0.0, 0.0]

    cases = [
        EvalCase.from_dict({"question": "superconducting transmon qubits",
                            "expect_papers": ["2608.22222v1"]}),
        EvalCase.from_dict({"question": "magnetic monopole detectors",
                            "expect_papers": ["2608.11111v1"]}),
    ]
    with chunked_corpus(tmp_path, {
        "2608.11111v1": ["A study of protein folding."],
        "2608.22222v1": ["Superconducting transmon qubits lose coherence."],
    }) as corpus:
        ranked = retrieval_ab.rank_cases(corpus, KeywordOnly(), cases, {})

    assert [row["rank"] for row in ranked] == [1, None]
    assert all(row["seconds"] >= 0 for row in ranked)


def test_another_key_gets_its_own_results_file() -> None:
    """The development key's ranks must not overwrite the held-out key's."""
    held_out = retrieval_ab.results_path("hybrid", retrieval_ab.KEYS)
    development = retrieval_ab.results_path("hybrid", [Path("evals/retrieval.jsonl")])

    assert held_out.name == "ab-hybrid.json"
    assert development.name == "ab-hybrid@retrieval.json"


def test_an_unknown_arm_is_refused_with_the_known_ones() -> None:
    with pytest.raises(SystemExit, match="hybrid\\+graph"):
        retrieval_ab.options_for("no-such-arm")


def test_every_arm_is_a_search_configuration() -> None:
    """A misspelt option would otherwise fail only after a model has loaded."""
    import inspect

    from verbatim.reranking import RERANKERS
    from verbatim.retrieval import hybrid_search

    accepted = set(inspect.signature(hybrid_search).parameters)
    for arm, options in retrieval_ab.ARMS.items():
        assert set(options) <= accepted, arm
        assert options.get("reranker", "cross") in RERANKERS, arm


def test_each_case_records_whether_the_legs_disagree(tmp_path: Path) -> None:
    """The gated reranker fires only there, so the signal is measured on its own."""
    from helpers import chunked_corpus
    from verbatim.corpus import EmbeddingModel
    from verbatim.evaluation import EvalCase

    class KeywordOnly:
        model = EmbeddingModel(model_id="stub", dimensions=4, normalized=True, revision="v0")

        def embed_query(self, text):
            return [0.0, 0.0, 0.0, 0.0]

    case = EvalCase.from_dict({"question": "transmon qubits", "expect_papers": ["2608.22222v1"]})
    with chunked_corpus(tmp_path, {"2608.22222v1": ["Transmon qubits lose coherence."]}) as corpus:
        [row] = retrieval_ab.rank_cases(corpus, KeywordOnly(), [case], {})

    # Nothing is embedded, so the vector leg finds nothing: that counts as disagreeing.
    assert row["legs_disagree"] is True


def test_named_rerankers_are_built_once_per_run(tmp_path: Path, monkeypatch) -> None:
    from helpers import chunked_corpus
    from verbatim.corpus import EmbeddingModel
    from verbatim.evaluation import EvalCase

    class KeywordOnly:
        model = EmbeddingModel(model_id="stub", dimensions=4, normalized=True, revision="v0")

        def embed_query(self, text):
            return [0.0, 0.0, 0.0, 0.0]

    class Flat:
        def score(self, query, passages):
            return [1.0] * len(passages)

    built: list[str] = []
    monkeypatch.setattr(retrieval_ab, "make_reranker", lambda name: built.append(name) or Flat())
    cases = [EvalCase.from_dict({"question": q, "expect_papers": ["2608.22222v1"]})
             for q in ("transmon qubits", "qubit coherence")]
    with chunked_corpus(tmp_path, {"2608.22222v1": ["Transmon qubits lose coherence."]}) as corpus:
        retrieval_ab.rank_cases(corpus, KeywordOnly(), cases, {"reranker": "colbert"})

    assert built == ["colbert"]
