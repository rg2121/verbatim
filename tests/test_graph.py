from pathlib import Path


from verbatim.corpus import CorpusStore, EmbeddingModel
from verbatim.embedding import embed_pending
from verbatim.graph import anchor_papers, anchored_matches, definitional_subject
from verbatim.references import Reference
from verbatim.retrieval import hybrid_search
from verbatim.tools import search_chunks

from helpers import corpus_with


def cite(corpus: CorpusStore, citing_arxiv_id: str, cited_base_ids: list[str]) -> None:
    corpus.record_references(
        [
            Reference(
                citing_arxiv_id=citing_arxiv_id,
                ordinal=ordinal,
                raw_text=f"[{ordinal}] arXiv:{cited}",
                cited_arxiv_id=cited,
            )
            for ordinal, cited in enumerate(cited_base_ids, start=1)
        ],
        source_sha256="abc123",
    )


def test_definitional_subject_matches_definition_shaped_questions() -> None:
    # The article is grammar, not part of what a paper is titled after: no
    # paper is called "the Adam optimizer" or "a tensor".
    assert definitional_subject("what is the Adam optimizer?") == "Adam optimizer"
    assert definitional_subject("what is a tensor?") == "tensor"
    assert definitional_subject("What are fiber bundles") == "fiber bundles"
    assert definitional_subject("define entropy") == "entropy"
    assert definitional_subject("definition of masked language modelling") == (
        "masked language modelling"
    )


def test_definitional_subject_rejects_every_other_question_shape() -> None:
    # The gate is the feature: always-on title matching was measured worse
    # precisely on questions like the PPO one, where "clip" matches CLIP.
    assert definitional_subject("why does PPO clip its objective") is None
    assert definitional_subject("how does a changing field induce current") is None
    assert definitional_subject("compare MOND and dark matter in clusters") is None
    assert definitional_subject("what came out this week in cs.CL") is None
    assert definitional_subject("what is") is None


def graph_corpus(tmp_path: Path) -> CorpusStore:
    """Adam's situation in miniature: one defining paper, many that use it."""
    corpus = corpus_with(
        tmp_path,
        {
            "1412.6980v1": (
                "Adam: A Method for Stochastic Optimization",
                ["Adam is an algorithm for stochastic gradient optimization."],
            ),
            "2608.11111v1": (
                "Scaling Vision Transformers",
                ["We trained with the adam optimizer, learning rate 3e-4."],
            ),
            "2608.22222v1": (
                "Language Models for Code",
                ["The adam optimizer was used with default settings."],
            ),
            "2608.33333v1": (
                "Adam Smith in the Age of Machines",  # title match, never cited
                ["An economics survey mentioning the adam optimizer once."],
            ),
        },
    )
    cite(corpus, "2608.11111v1", ["1412.6980"])
    cite(corpus, "2608.22222v1", ["1412.6980"])
    return corpus


def test_anchor_papers_needs_the_corpus_to_testify(tmp_path: Path) -> None:
    """A title match anchors only when held papers cite it."""
    with graph_corpus(tmp_path) as corpus:
        anchors = anchor_papers(corpus, "the adam optimizer")

    assert [(p.arxiv_id, cited) for p, cited in anchors] == [("1412.6980v1", 2)]
    # The uncited economics paper also matches "adam" by title and is dropped.


def test_anchor_papers_returns_nothing_for_an_uncited_subject(tmp_path: Path) -> None:
    with corpus_with(
        tmp_path,
        {"2608.44444v1": ("A Tensor Slicing Library", ["We slice tensors quickly."])},
    ) as corpus:
        assert anchor_papers(corpus, "tensor slicing") == []


def test_anchored_matches_ignores_non_definitional_questions(tmp_path: Path) -> None:
    with graph_corpus(tmp_path) as corpus:
        assert anchored_matches(corpus, "why does adam converge quickly") == []


def test_anchored_matches_returns_passages_from_the_defining_paper(tmp_path: Path) -> None:
    with graph_corpus(tmp_path) as corpus:
        matches = anchored_matches(corpus, "what is the adam optimizer?")

    assert matches
    assert {m.chunk.arxiv_id for m in matches} == {"1412.6980v1"}


def test_papers_citing_counts_a_paper_once_across_its_versions(tmp_path: Path) -> None:
    with corpus_with(
        tmp_path,
        {
            "1412.6980v1": ("Adam: A Method for Stochastic Optimization", []),
            "2608.11111v1": ("Scaling Vision Transformers", []),
            "2608.11111v2": ("Scaling Vision Transformers", []),
        },
    ) as corpus:
        # Both versions were read, so both hold a stored bibliography.
        cite(corpus, "2608.11111v1", ["1412.6980"])
        cite(corpus, "2608.11111v2", ["1412.6980"])

        citers = corpus.papers_citing("1412.6980")
        counts = corpus.citation_counts(["1412.6980"])

    assert [p.arxiv_id for p in citers] == ["2608.11111v2"]
    assert counts == {"1412.6980": 1}


def test_self_citations_do_not_count_as_influence(tmp_path: Path) -> None:
    with corpus_with(
        tmp_path,
        {"2608.55555v2": ("A Paper Citing Its Own Preprint", [])},
    ) as corpus:
        cite(corpus, "2608.55555v2", ["2608.55555"])

        assert corpus.papers_citing("2608.55555") == []
        assert corpus.citation_counts(["2608.55555"]) == {}
        assert corpus.citation_graph_stats()["edges"] == 0


def test_papers_cited_by_lists_held_references_in_bibliography_order(tmp_path: Path) -> None:
    with corpus_with(
        tmp_path,
        {
            "1412.6980v1": ("Adam: A Method for Stochastic Optimization", []),
            "1810.04805v1": ("BERT: Pre-training of Deep Bidirectional Transformers", []),
            "2608.11111v1": ("Scaling Vision Transformers", []),
        },
    ) as corpus:
        # The bibliography also cites a paper the corpus does not hold.
        cite(corpus, "2608.11111v1", ["1810.04805", "9901.00001", "1412.6980"])

        cited = corpus.papers_cited_by("2608.11111v1")

    assert [p.arxiv_id for p in cited] == ["1810.04805v1", "1412.6980v1"]


def test_most_cited_held_papers_ranks_by_distinct_citers(tmp_path: Path) -> None:
    with corpus_with(
        tmp_path,
        {
            "1412.6980v1": ("Adam: A Method for Stochastic Optimization", []),
            "1810.04805v1": ("BERT: Pre-training of Deep Bidirectional Transformers", []),
            "2608.11111v1": ("Scaling Vision Transformers", []),
            "2608.22222v1": ("Language Models for Code", []),
        },
    ) as corpus:
        cite(corpus, "2608.11111v1", ["1412.6980", "1810.04805"])
        cite(corpus, "2608.22222v1", ["1412.6980"])

        ranked = corpus.most_cited_held_papers(limit=5)
        stats = corpus.citation_graph_stats()

    assert [(p.arxiv_id, n) for p, n in ranked] == [
        ("1412.6980v1", 2),
        ("1810.04805v1", 1),
    ]
    assert stats["edges"] == 3
    assert stats["connected_papers"] == 4
    assert stats["papers"] == 4


class TermEmbedder:
    """Vectors keyed to the Adam fixture, so drowning is reproducible."""

    def __init__(self) -> None:
        self.model = EmbeddingModel(
            model_id="stub", dimensions=3, normalized=True, revision="v0"
        )

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        raw = [
            float("adam" in lowered),
            float("optimizer" in lowered),
            float("stochastic" in lowered),
        ]
        norm = sum(value * value for value in raw) ** 0.5 or 1.0
        return [value / norm for value in raw]

    def embed_passages(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def test_use_graph_surfaces_the_defining_paper_a_definitional_question_needs(
    tmp_path: Path,
) -> None:
    with graph_corpus(tmp_path) as corpus:
        embed_pending(corpus, TermEmbedder())
        question = "what is the adam optimizer?"

        plain = hybrid_search(corpus, TermEmbedder(), question, limit=2)
        graphed = hybrid_search(corpus, TermEmbedder(), question, limit=2, use_graph=True)

    # Both legs prefer the passages that merely use Adam, so the paper that
    # defines it is not in the short results - until the graph leg adds it.
    assert "1412.6980v1" not in {m.chunk.arxiv_id for m in plain}
    assert "1412.6980v1" in {m.chunk.arxiv_id for m in graphed}


def test_use_graph_respects_a_search_scoped_to_one_paper(tmp_path: Path) -> None:
    """The anchor leg must not smuggle its paper past the caller's scope."""
    with graph_corpus(tmp_path) as corpus:
        embed_pending(corpus, TermEmbedder())
        scoped = hybrid_search(
            corpus,
            TermEmbedder(),
            "what is the adam optimizer?",
            limit=5,
            arxiv_id="2608.11111v1",
            use_graph=True,
        )

    assert {m.chunk.arxiv_id for m in scoped} == {"2608.11111v1"}


def test_use_graph_leaves_every_other_question_untouched(tmp_path: Path) -> None:
    with graph_corpus(tmp_path) as corpus:
        embed_pending(corpus, TermEmbedder())
        question = "how is the adam optimizer configured in practice"

        plain = hybrid_search(corpus, TermEmbedder(), question, limit=5)
        graphed = hybrid_search(corpus, TermEmbedder(), question, limit=5, use_graph=True)

    assert [(m.chunk.chunk_id, m.score) for m in plain] == [
        (m.chunk.chunk_id, m.score) for m in graphed
    ]


def test_the_tool_every_other_front_end_calls_uses_the_graph_gate(tmp_path: Path) -> None:
    """MCP clients, `ask --tools` and the orchestrator's agent rung go here."""
    with graph_corpus(tmp_path) as corpus:
        embed_pending(corpus, TermEmbedder())

        defining = search_chunks(
            corpus, TermEmbedder(), "what is the adam optimizer?", limit=2
        )
        ordinary = search_chunks(
            corpus, TermEmbedder(), "how is the adam optimizer configured", limit=2
        )

    assert "1412.6980v1" in {hit["arxiv_id"] for hit in defining}
    assert "1412.6980v1" not in {hit["arxiv_id"] for hit in ordinary}
