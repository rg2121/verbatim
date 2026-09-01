from pathlib import Path

from verbatim.corpus import CorpusStore, EmbeddingModel
from verbatim.embedding import embed_pending
from verbatim.retrieval import comparison_shaped, hybrid_search

from helpers import corpus_with


def test_comparison_shaped_matches_questions_that_set_things_against_each_other() -> None:
    assert comparison_shaped("compare the post-training pipelines of Qwen3 and Llama 3")
    assert comparison_shaped("how does GRPO differ from the PPO algorithm it modifies?")
    assert comparison_shaped("MOND versus dark matter in galaxy clusters")
    assert comparison_shaped("how is masked pretraining different from next-token prediction?")
    assert comparison_shaped("what is the difference between fusion and fission?")


def test_comparison_shaped_rejects_single_subject_questions() -> None:
    assert not comparison_shaped("what is the Adam optimizer?")
    assert not comparison_shaped("how does a changing field induce current")
    # "different" without "from" is a single-subject phrasing.
    assert not comparison_shaped("what are the different approaches to regularization?")


class FluxChargeEmbedder:
    """Vectors keyed to the fixture, so ranking is checkable by hand."""

    def __init__(self) -> None:
        self.model = EmbeddingModel(
            model_id="stub", dimensions=3, normalized=True, revision="v0"
        )

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        raw = [
            float("flux" in lowered),
            float("charge" in lowered),
            float("plant" in lowered),
        ]
        norm = sum(value * value for value in raw) ** 0.5 or 1.0
        return [value / norm for value in raw]

    def embed_passages(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def comparison_corpus(tmp_path: Path) -> CorpusStore:
    """Both legs prefer the wordy passages; only a title names the third paper."""
    corpus = corpus_with(
        tmp_path,
        {
            "2608.11111v1": (
                "Ring Measurements",
                ["Magnetic flux transport threads the charge wire."],
            ),
            "2608.22222v1": (
                "Wire Measurements",
                ["Charge and flux transport move along the wire."],
            ),
            "2608.33333v1": (
                "Flux Transport Compared",
                ["Transport of the ring current."],
            ),
        },
    )
    embed_pending(corpus, FluxChargeEmbedder())
    return corpus


def test_use_compare_adds_the_title_named_paper_on_a_comparison_question(
    tmp_path: Path,
) -> None:
    question = "how does flux transport differ from charge transport?"
    with comparison_corpus(tmp_path) as corpus:
        plain = hybrid_search(corpus, FluxChargeEmbedder(), question, limit=2)
        compared = hybrid_search(
            corpus, FluxChargeEmbedder(), question, limit=2, use_compare=True
        )

    assert "2608.33333v1" not in {m.chunk.arxiv_id for m in plain}
    assert "2608.33333v1" in {m.chunk.arxiv_id for m in compared}


def test_use_compare_leaves_non_comparison_questions_untouched(tmp_path: Path) -> None:
    question = "how does flux thread the ring"
    with comparison_corpus(tmp_path) as corpus:
        plain = hybrid_search(corpus, FluxChargeEmbedder(), question, limit=3)
        compared = hybrid_search(
            corpus, FluxChargeEmbedder(), question, limit=3, use_compare=True
        )

    assert [(m.chunk.chunk_id, m.score) for m in plain] == [
        (m.chunk.chunk_id, m.score) for m in compared
    ]


def test_title_legs_respect_a_search_scoped_to_one_paper(tmp_path: Path) -> None:
    """Neither title leg may smuggle its paper past the caller's scope."""
    question = "how does flux transport differ from charge transport?"
    with comparison_corpus(tmp_path) as corpus:
        for options in ({"use_titles": True}, {"use_compare": True}):
            scoped = hybrid_search(
                corpus,
                FluxChargeEmbedder(),
                question,
                limit=5,
                arxiv_id="2608.11111v1",
                **options,
            )
            assert {m.chunk.arxiv_id for m in scoped} == {"2608.11111v1"}
