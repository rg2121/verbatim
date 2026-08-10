from pathlib import Path

import pytest

from verbatim.tools import call_tool, get_paper, list_papers, read_pages

from helpers import chunked_corpus, paper


def corpus_with_dates(tmp_path: Path):
    c = chunked_corpus(tmp_path, {"2608.11111v1": ["## Intro\n\nMagnetic flux threads the ring."]})
    c.upsert_papers([
        paper("2608.22222v1", updated="2026-08-09T10:00:00Z"),
        paper("2608.33333v1", updated="2026-06-01T10:00:00Z"),
    ])
    return c


def test_list_papers_answers_recency_which_search_cannot(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        recent = list_papers(corpus, since="2026-08-01T00:00:00Z")
        older = list_papers(corpus, since="2026-01-01T00:00:00Z")

    ids = [p["arxiv_id"] for p in recent]
    assert "2608.33333v1" not in ids          # June paper excluded
    assert "2608.22222v1" in ids
    assert len(older) > len(recent)
    assert [p["updated"] for p in older] == sorted((p["updated"] for p in older), reverse=True)


def test_list_papers_filters_by_category(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        assert list_papers(corpus, category="physics.gen-ph")   # helpers use this category
        assert list_papers(corpus, category="cs.NONEXISTENT") == []


def test_get_paper_resolves_by_id_and_by_approximate_title(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        by_id = get_paper(corpus, "2608.22222v1")
        by_title = get_paper(corpus, "Paper 2608.22222v1")
        missing = get_paper(corpus, "no such paper anywhere")

    assert by_id is not None and by_id["arxiv_id"] == "2608.22222v1"
    assert by_title is not None and by_title["arxiv_id"] == "2608.22222v1"
    assert missing is None


def test_read_pages_returns_only_the_requested_pages(tmp_path: Path) -> None:
    with chunked_corpus(
        tmp_path,
        {"2608.11111v1": ["## A\n\nPage one content.", "## B\n\nPage two content.",
                          "## C\n\nPage three content."]},
    ) as corpus:
        page_two = read_pages(corpus, "2608.11111v1", [2])
        two_and_three = read_pages(corpus, "2608.11111v1", [2, 3])

    assert [c["page"] for c in page_two] == [2]
    assert "Page two" in page_two[0]["text"]
    assert sorted({c["page"] for c in two_and_three}) == [2, 3]


def test_an_unknown_tool_name_is_rejected(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        with pytest.raises(ValueError, match="Unknown tool"):
            call_tool("drop_everything", {}, corpus, embedder=None)


def test_call_tool_dispatches_without_a_model(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        found = call_tool("get_paper", {"identifier": "2608.22222v1"}, corpus, embedder=None)
        listed = call_tool("list_papers", {"limit": 2}, corpus, embedder=None)

    assert found["arxiv_id"] == "2608.22222v1"
    assert len(listed) == 2


def test_get_paper_accepts_the_version_less_id_people_actually_write(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        versioned = get_paper(corpus, "2608.22222v1")
        bare = get_paper(corpus, "2608.22222")
        prefixed = get_paper(corpus, "arXiv:2608.22222")

    assert versioned is not None
    assert bare == versioned
    assert prefixed == versioned


def test_get_paper_resolves_a_base_id_to_the_current_version(tmp_path: Path) -> None:
    from helpers import paper as make

    with corpus_with_dates(tmp_path) as corpus:
        corpus.upsert_papers([make("2608.44444v1"), make("2608.44444v3", updated="2026-08-09T00:00:00Z")])
        found = get_paper(corpus, "2608.44444")

    assert found is not None and found["arxiv_id"] == "2608.44444v3"
