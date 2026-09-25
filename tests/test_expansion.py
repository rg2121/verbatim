"""Rewriting a question before retrieval sees it.

Both treatments are optimisations, so the tests are mostly about what happens
when they go wrong: retrieval must still answer, and must never lose the words
the user actually typed.
"""

import pytest

from verbatim.expansion import (
    dedupe_queries,
    fuse_rankings,
    hypothetical_answer,
    rewrite_query,
)


def reply(text: str):
    return lambda system, user: text


def test_the_users_own_words_are_always_kept_and_come_first() -> None:
    """A rewrite can drop the term that mattered; fusion must not lose it."""
    queries = rewrite_query(reply("regularisation against memorisation\nmembership inference"),
                            "how do you stop a model memorising its training data?")

    assert queries[0] == "how do you stop a model memorising its training data?"
    assert "membership inference" in queries


def test_list_markers_are_stripped_from_a_models_reply() -> None:
    queries = rewrite_query(reply("- Shannon entropy\n2) information content\n\n"), "entropy?")

    assert queries == ["entropy?", "Shannon entropy", "information content"]


def test_a_rewrite_that_starts_with_a_number_keeps_it() -> None:
    queries = rewrite_query(reply("- 3D point cloud segmentation\n2) 8-bit optimizers"), "q")

    assert queries == ["q", "3D point cloud segmentation", "8-bit optimizers"]


def test_a_repeated_phrasing_is_not_retrieved_for_twice() -> None:
    assert dedupe_queries(["what is entropy?", "What is  entropy?", "other"]) == [
        "what is entropy?",
        "other",
    ]


def test_retrieval_still_works_when_the_model_is_unreachable() -> None:
    """Expansion is an optimisation and must never be able to break search."""
    def broken(system: str, user: str) -> str:
        raise RuntimeError("Cannot reach Ollama")

    assert rewrite_query(broken, "what is entropy?") == ["what is entropy?"]
    assert hypothetical_answer(broken, "what is entropy?") == "what is entropy?"


def test_an_empty_hypothetical_answer_falls_back_to_the_question() -> None:
    assert hypothetical_answer(reply("   \n  "), "what is entropy?") == "what is entropy?"


def test_hyde_returns_the_sentence_a_paper_would_contain() -> None:
    answer = hypothetical_answer(
        reply("Entropy measures the average information content of a source."),
        "what is entropy?",
    )

    assert answer == "Entropy measures the average information content of a source."


def test_variants_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        rewrite_query(reply("x"), "q", variants=0)


def test_a_passage_found_by_several_phrasings_outranks_one_found_by_a_single(
) -> None:
    """This is the whole point of retrieving for more than one phrasing."""
    from verbatim.chunking import Chunk
    from verbatim.corpus import ChunkMatch

    def match(chunk_id: str, rank_text: str) -> ChunkMatch:
        return ChunkMatch(
            chunk=Chunk(
                chunk_id=chunk_id, arxiv_id="2608.11111v1", source_url="", license_url=None,
                source_sha256="f" * 64, page_number=1, chunk_index=0, char_start=0,
                char_end=1, text=rank_text, chunker_version="8:1200:200",
            ),
            score=1.0,
            snippet=rank_text,
        )

    # "shared" is second in both rankings; "top" is first in only one.
    fused = fuse_rankings([
        [match("top", "a"), match("shared", "b")],
        [match("other", "c"), match("shared", "b")],
    ])

    assert fused[0].chunk.chunk_id == "shared"
