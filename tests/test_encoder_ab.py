"""The encoder experiment's guards, with no model and no corpus."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
import encoder_ab  # noqa: E402

from helpers import chunked_corpus  # noqa: E402


def test_an_encoder_that_has_not_embedded_every_passage_is_counted_short(tmp_path: Path) -> None:
    """Scoring mid-embed would search a fraction of the corpus under the model's name."""
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Flux decoheres the qubit."]}) as corpus:
        assert encoder_ab.unembedded(corpus, "never-run-model") > 0


def test_results_are_named_by_arm_search_and_key() -> None:
    assert encoder_ab.results_path("arctic-s", encoder_ab.retrieval_ab.KEYS).name == "encoder-arctic-s.json"
    assert encoder_ab.results_path(
        "arctic-s", [Path("evals/retrieval.jsonl")], "hybrid"
    ).name == "encoder-arctic-s.hybrid@retrieval.json"


def _rows(ranks: list[int | None]) -> list[dict]:
    return [{"question": f"q{i}", "source": "s", "rank": r, "seconds": 0.1} for i, r in enumerate(ranks)]


@pytest.mark.parametrize(("after", "clears"), [
    ([1] * 30, True),          # 30 of 30 moved up from second place
    ([1] * 3 + [2] * 27, False),
])
def test_an_arm_clears_only_the_split_threshold(after: list[int], clears: bool) -> None:
    first = {"arm": "bge-small", "search": encoder_ab.SEARCH, "rows": _rows([2] * 30)}
    second = {"arm": "arctic-s", "search": encoder_ab.SEARCH, "rows": _rows(after)}

    text = encoder_ab.report(first, second)

    assert ("does not clear" not in text) is clears
