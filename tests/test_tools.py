from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest

from verbatim.answers import verify_quote
from verbatim.corpus import ChunkMatch
from verbatim.references import Reference
import verbatim.tools as tools
from verbatim.tools import (
    MAX_QUOTES,
    MAX_RESULTS,
    TOOL_SCHEMAS,
    call_tool,
    export_citations,
    get_citations,
    get_paper,
    list_papers,
    read_pages,
    search_chunks,
    verify_citations,
)

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
        page_two = read_pages(corpus, "2608.11111v1", [2])["passages"]
        two_and_three = read_pages(corpus, "2608.11111v1", [2, 3])["passages"]

    assert [c["page"] for c in page_two] == [2]
    assert "Page two" in page_two[0]["text"]
    assert sorted({c["page"] for c in two_and_three}) == [2, 3]


def test_get_citations_walks_both_directions_of_the_graph(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        corpus.record_references(
            [
                Reference(
                    citing_arxiv_id="2608.22222v1",
                    ordinal=1,
                    raw_text="[1] arXiv:2608.11111",
                    cited_arxiv_id="2608.11111",
                )
            ],
            source_sha256="abc123",
        )
        cited_by = get_citations(corpus, "2608.11111v1", direction="cited_by")
        # Resolution goes through get_paper, so a bare base id works too.
        references = get_citations(corpus, "2608.22222", direction="references")
        missing = get_citations(corpus, "no such paper anywhere")

    assert cited_by is not None
    assert [p["arxiv_id"] for p in cited_by["papers"]] == ["2608.22222v1"]
    assert references is not None
    assert [p["arxiv_id"] for p in references["papers"]] == ["2608.11111v1"]
    assert missing is None


def test_get_citations_rejects_a_direction_it_never_offered(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        with pytest.raises(ValueError, match="direction"):
            get_citations(corpus, "2608.11111v1", direction="sideways")


def test_get_citations_dispatches_without_a_model(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        result = call_tool(
            "get_citations", {"identifier": "2608.11111v1"}, corpus, embedder=None
        )

    assert result is not None and result["direction"] == "cited_by"


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


def test_search_chunks_searches_as_every_front_end_does(monkeypatch) -> None:
    """The graph leg once reached the CLI and not MCP; one default keeps them together."""
    import verbatim.tools as tools
    from verbatim import config
    from verbatim.retrieval import default_search_options

    searched: dict = {}
    monkeypatch.setattr(config, "RERANKER", "colbert")
    monkeypatch.setattr(tools, "hybrid_search", lambda *args, **options: searched.update(options) or [])

    tools.search_chunks(None, None, "how do transmons decohere?")

    shared = default_search_options()
    assert {name: searched[name] for name in shared} == shared


# A schema is hand-written beside its function, so nothing else would notice a drift.
@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda schema: schema["name"])
def test_each_tool_schema_matches_its_function(schema) -> None:
    parameters = inspect.signature(getattr(tools, schema["name"])).parameters
    takes = {name: p for name, p in parameters.items() if name not in ("corpus", "embedder")}
    required = {name for name, p in takes.items() if p.default is inspect.Parameter.empty}

    assert set(schema["parameters"]["properties"]) <= set(takes)
    assert set(schema["parameters"].get("required", ())) == required


def _first_chunk(corpus):
    return list(corpus.iter_chunks())[0]


def _search_finding(monkeypatch, chunk) -> None:
    """Search returns this passage, with no model to load."""
    monkeypatch.setattr(
        tools, "hybrid_search",
        lambda *args, **options: [ChunkMatch(chunk=chunk, score=1.0, snippet="")],
    )


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_a_blank_search_is_refused_rather_than_answered(tmp_path: Path, query: str) -> None:
    """It once returned an unrelated table, which reads as an answer."""
    with corpus_with_dates(tmp_path) as corpus:
        with pytest.raises(ValueError, match="query is empty"):
            search_chunks(corpus, None, query)


@pytest.mark.parametrize("identifier", ["", "   ", "arXiv:", "?!"])
def test_a_blank_identifier_is_refused_rather_than_matched_to_any_title(
    tmp_path: Path, identifier: str
) -> None:
    """A blank title search matched every paper, and the shortest title won."""
    with corpus_with_dates(tmp_path) as corpus:
        with pytest.raises(ValueError, match="identifier"):
            get_paper(corpus, identifier)
        with pytest.raises(ValueError, match="identifier"):
            get_citations(corpus, identifier)


@pytest.mark.parametrize(
    ("arguments", "complaint"),
    [
        ({"since": "not-a-date"}, "since must be"),
        ({"days": -5}, "days must be"),
        ({"days": 0}, "days must be"),
        ({"limit": 0}, "limit must be"),
        ({"limit": -1}, "limit must be"),
    ],
)
def test_list_papers_refuses_what_would_look_like_no_results(
    tmp_path: Path, arguments: dict, complaint: str
) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        with pytest.raises(ValueError, match=complaint):
            list_papers(corpus, **arguments)


def test_a_negative_limit_cannot_unbound_a_list(tmp_path: Path) -> None:
    """SQLite reads LIMIT -1 as no limit at all."""
    with corpus_with_dates(tmp_path) as corpus:
        with pytest.raises(ValueError, match="limit must be"):
            get_citations(corpus, "2608.11111v1", limit=-1)
        with pytest.raises(ValueError, match="limit must be"):
            search_chunks(corpus, None, "flux", limit=0)


def test_since_is_read_as_a_moment_not_compared_as_text(tmp_path: Path) -> None:
    """12:00 at +02:00 is 10:00 UTC, when this paper was updated; as text, 10 sorts before 12."""
    with corpus_with_dates(tmp_path) as corpus:
        offset = list_papers(corpus, since="2026-08-09T12:00:00+02:00")
        date_only = list_papers(corpus, since="2026-08-01")

    assert "2608.22222v1" in [p["arxiv_id"] for p in offset]
    assert "2608.22222v1" in [p["arxiv_id"] for p in date_only]


def test_days_counts_back_from_now(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        corpus.upsert_papers([paper("2608.55555v1", updated="2999-01-01T00:00:00Z")])
        recent = list_papers(corpus, days=1)

    assert [p["arxiv_id"] for p in recent] == ["2608.55555v1"]


def test_a_paper_the_corpus_does_not_hold_is_refused_rather_than_empty(tmp_path: Path) -> None:
    """An empty list would read as a paper with nothing in it."""
    with corpus_with_dates(tmp_path) as corpus:
        with pytest.raises(ValueError, match="no paper"):
            read_pages(corpus, "9999.99999v1", [1])
        with pytest.raises(ValueError, match="no paper"):
            search_chunks(corpus, None, "flux", arxiv_id="9999.99999v1")


def test_scoped_tools_accept_the_version_less_id_people_write(tmp_path: Path, monkeypatch) -> None:
    searched: dict = {}
    monkeypatch.setattr(tools, "hybrid_search", lambda *a, **o: searched.update(o) or [])

    with corpus_with_dates(tmp_path) as corpus:
        read = read_pages(corpus, "2608.11111", [1])
        search_chunks(corpus, None, "flux", arxiv_id="arXiv:2608.11111")

    assert read["arxiv_id"] == "2608.11111v1" and read["passages"]
    assert searched["arxiv_id"] == "2608.11111v1"


def test_a_blank_scope_means_no_scope(tmp_path: Path, monkeypatch) -> None:
    searched: dict = {}
    monkeypatch.setattr(tools, "hybrid_search", lambda *a, **o: searched.update(o) or [])

    with corpus_with_dates(tmp_path) as corpus:
        search_chunks(corpus, None, "flux", arxiv_id="  ")

    assert searched["arxiv_id"] is None and searched["max_per_paper"] == 3


def test_reading_no_pages_is_refused(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        with pytest.raises(ValueError, match="pages is empty"):
            read_pages(corpus, "2608.11111v1", [])


def test_an_html_passage_names_its_section_and_claims_no_page(tmp_path: Path, monkeypatch) -> None:
    """94% of the corpus is HTML, where page_number only counts sections."""
    with corpus_with_dates(tmp_path) as corpus:
        html = replace(
            _first_chunk(corpus), source_kind="html", page_number=57,
            section_path="3 Model > 3.3 YaRN",
        )
        _search_finding(monkeypatch, html)
        (row,) = search_chunks(corpus, None, "flux")

    assert row["locator"] == "section 3.3 YaRN"
    assert row["source_kind"] == "html"
    assert row["position"] == 57
    assert row["section"] == "3.3 YaRN"
    assert "page" not in row


def test_a_pdf_passage_cites_its_page(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        (row,) = read_pages(corpus, "2608.11111v1", [1])["passages"]

    assert row["locator"] == "p. 1"
    assert row["source_kind"] == "pdf"
    assert row["page"] == row["position"] == 1


def test_a_passage_carries_what_a_citation_needs(tmp_path: Path, monkeypatch) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        chunk = _first_chunk(corpus)
        corpus.record_license("2608.11111", "http://creativecommons.org/licenses/by/4.0/")
        _search_finding(monkeypatch, chunk)
        (row,) = search_chunks(corpus, None, "flux")

    assert row["chunk_id"] == chunk.chunk_id
    assert row["arxiv_id"] == "2608.11111v1"
    assert row["title"] == "Paper 2608.11111v1"
    assert row["url"] == chunk.source_url
    assert row["license_url"] == "http://creativecommons.org/licenses/by/4.0/"
    assert row["text"] == chunk.text


def _long_paper(tmp_path: Path):
    return chunked_corpus(tmp_path, {"2608.11111v1": ["## Long\n\n" + "word " * 2000]})


def test_read_pages_stops_at_its_budget_and_says_how_to_go_on(tmp_path: Path, monkeypatch) -> None:
    """One HTML section runs to 1.8 million characters, and a client warns far below that.

    The budget is on the reply: with its fields, a row is much larger than its text.
    """
    monkeypatch.setattr(tools, "READ_CHARACTER_BUDGET", 5_000)

    with _long_paper(tmp_path) as corpus:
        everything = [chunk.chunk_id for chunk in corpus.iter_chunks()]
        read, skip = [], 0
        while skip is not None:
            part = read_pages(corpus, "2608.11111v1", [1], skip=skip)
            sizes = [len(json.dumps(row, indent=2, ensure_ascii=False)) for row in part["passages"]]
            assert sum(sizes) <= 5_000
            # Only the last reply may hold a single passage: the budget has room for more.
            assert len(sizes) > 1 or part["next_skip"] is None
            read.extend(row["chunk_id"] for row in part["passages"])
            skip = part["next_skip"]

    assert len(everything) > 2
    assert part["passages_total"] == len(everything)
    assert read == everything


def test_read_pages_returns_a_passage_larger_than_its_budget(tmp_path: Path, monkeypatch) -> None:
    """Otherwise that passage could never be read, and `next_skip` would never move."""
    monkeypatch.setattr(tools, "READ_CHARACTER_BUDGET", 10)

    with _long_paper(tmp_path) as corpus:
        part = read_pages(corpus, "2608.11111v1", [1])

    assert len(part["passages"]) == 1 and part["next_skip"] == 1


def test_read_pages_says_which_positions_it_did_not_read(tmp_path: Path) -> None:
    """It once cut the list at 20 and said nothing."""
    with chunked_corpus(
        tmp_path, {"2608.11111v1": ["## A\n\nOne.", "## B\n\nTwo.", "## C\n\nThree."]}
    ) as corpus:
        part = read_pages(corpus, "2608.11111v1", list(range(1, 26)))

    assert part["positions_not_read"] == [21, 22, 23, 24, 25]
    assert part["positions_without_passages"] == list(range(4, 21))
    assert part["next_skip"] is None


def test_skip_cannot_be_negative(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        with pytest.raises(ValueError, match="skip"):
            read_pages(corpus, "2608.11111v1", [1], skip=-1)


def test_call_tool_passes_skip_through(tmp_path: Path) -> None:
    with _long_paper(tmp_path) as corpus:
        first = call_tool("read_pages", {"arxiv_id": "2608.11111v1", "pages": [1]}, corpus, None)
        later = call_tool(
            "read_pages", {"arxiv_id": "2608.11111v1", "pages": [1], "skip": 1}, corpus, None
        )

    assert later["passages"][0]["chunk_id"] == first["passages"][1]["chunk_id"]


def _cited_by_many(corpus, count: int) -> None:
    for n in range(count):
        citer = f"2608.2{n:04d}v1"
        corpus.upsert_papers([paper(citer)])
        corpus.record_references(
            [Reference(citing_arxiv_id=citer, ordinal=1, raw_text="[1]", cited_arxiv_id="2608.11111")],
            source_sha256=f"sha{n}",
        )


def test_get_citations_reports_the_true_total_past_its_cap(tmp_path: Path) -> None:
    """A list that stops at 25 reads as 25 citations."""
    with corpus_with_dates(tmp_path) as corpus:
        _cited_by_many(corpus, MAX_RESULTS + 5)
        found = get_citations(corpus, "2608.11111v1", limit=100)

    assert len(found["papers"]) == MAX_RESULTS
    assert found["total"] == MAX_RESULTS + 5


def test_get_citations_reports_the_true_total_of_references_too(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        held = [f"2608.3{n:04d}" for n in range(MAX_RESULTS + 5)]
        corpus.upsert_papers([paper(f"{base}v1") for base in held])
        corpus.record_references(
            [
                Reference(citing_arxiv_id="2608.11111v1", ordinal=n, raw_text=f"[{n}]", cited_arxiv_id=base)
                for n, base in enumerate(held, start=1)
            ],
            source_sha256="abc",
        )
        found = get_citations(corpus, "2608.11111v1", direction="references", limit=3)

    assert len(found["papers"]) == 3
    assert found["total"] == MAX_RESULTS + 5


SENTENCE = "Magnetic flux threads the ring."


def _quote(chunk_id: str, quote: str) -> dict[str, str]:
    return {"chunk_id": chunk_id, "quote": quote}


def test_an_exact_quote_is_verified_and_comes_back_citable(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        chunk = _first_chunk(corpus)
        report = verify_citations(corpus, [_quote(chunk.chunk_id, "flux threads the ring")])

    (verdict,) = report["verdicts"]
    assert report["all_verified"] is True
    assert (report["verified"], report["rejected"]) == (1, 0)
    assert verdict["verified"] is True and verdict["reason"] is None
    assert verdict["citation"]["arxiv_id"] == "2608.11111v1"
    assert verdict["citation"]["title"] == "Paper 2608.11111v1"
    assert verdict["citation"]["locator"] == "p. 1"
    assert verdict["citation"]["source_sha256"] == chunk.source_sha256


def test_one_changed_word_fails_and_the_real_words_are_offered(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        chunk = _first_chunk(corpus)
        report = verify_citations(corpus, [_quote(chunk.chunk_id, "Magnetic flux threads the loop")])

    (verdict,) = report["verdicts"]
    assert report["all_verified"] is False
    assert (report["verified"], report["rejected"]) == (0, 1)
    assert verdict["verified"] is False
    assert verdict["reason"] == "quoted text does not appear in the cited passage"
    assert verdict["citation"] is None
    # Copied back, the hint has to pass.
    assert verdict["did_you_mean"] == SENTENCE
    assert verify_quote(verdict["did_you_mean"], chunk.text) is None


@pytest.mark.parametrize("quote", [f'"{SENTENCE}"', f"  Magnetic   flux\nthreads the ring.  ", SENTENCE[:-1]])
def test_a_client_is_forgiven_what_ask_is_forgiven(tmp_path: Path, quote: str) -> None:
    """Quote marks, whitespace and trailing punctuation: the same check, no looser and no tighter."""
    with corpus_with_dates(tmp_path) as corpus:
        report = verify_citations(corpus, [_quote(_first_chunk(corpus).chunk_id, quote)])

    assert report["all_verified"] is True


def test_nothing_close_gets_no_hint(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        report = verify_citations(
            corpus, [_quote(_first_chunk(corpus).chunk_id, "entirely unrelated words about cooking")]
        )

    assert report["verdicts"][0]["did_you_mean"] is None


def test_a_passage_that_does_not_exist_is_rejected_with_the_way_forward(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        report = verify_citations(corpus, [_quote("feedface", SENTENCE)])

    (verdict,) = report["verdicts"]
    assert report["all_verified"] is False
    assert "no passage with this chunk_id" in verdict["reason"]
    assert "search_chunks" in verdict["reason"]


def test_an_empty_quote_fails(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        report = verify_citations(corpus, [_quote(_first_chunk(corpus).chunk_id, "  ")])

    assert report["verdicts"][0]["reason"] == "empty quote"


def test_a_quote_from_a_neighbouring_passage_is_pointed_at_it(tmp_path: Path) -> None:
    with chunked_corpus(
        tmp_path,
        {"2608.11111v1": [f"## A\n\n{SENTENCE}", "## B\n\nCooper pairs tunnel through the junction."]},
    ) as corpus:
        first, second = [chunk.chunk_id for chunk in corpus.iter_chunks()]
        report = verify_citations(corpus, [_quote(first, "Cooper pairs tunnel through the junction")])

    assert report["verdicts"][0]["verified"] is False
    assert report["verdicts"][0]["found_in"] == second


def test_an_empty_batch_and_an_oversized_one_are_refused(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        chunk_id = _first_chunk(corpus).chunk_id
        with pytest.raises(ValueError, match="citations is empty"):
            verify_citations(corpus, [])
        with pytest.raises(ValueError, match=f"at most {MAX_QUOTES}"):
            verify_citations(corpus, [_quote(chunk_id, SENTENCE)] * (MAX_QUOTES + 1))


def test_the_report_says_what_verified_does_not_mean(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        report = verify_citations(corpus, [_quote(_first_chunk(corpus).chunk_id, SENTENCE)])

    assert "not" in report["note"] and "support" in report["note"]


def test_a_full_answer_is_read_the_way_ask_reads_one(tmp_path: Path) -> None:
    """Rejected markers, unmarked sentences and markers on sentences the passage does not share."""
    backed = "Magnetic flux threads the ring here [1]."
    unbacked = "Superconducting trains levitate across Europe quite quickly [1]."
    rejected = "The junction was invented by this sentence today [2]."
    unmarked = "And this sentence carries no marker at all."

    with corpus_with_dates(tmp_path) as corpus:
        chunk_id = _first_chunk(corpus).chunk_id
        report = verify_citations(
            corpus,
            [_quote(chunk_id, SENTENCE), _quote(chunk_id, "words that are not in it")],
            answer=f"{backed} {unbacked} {rejected} {unmarked}",
        )

    assert "today [unverified]." in report["checked_answer"]
    assert report["unmarked_sentences"] == [unmarked]
    assert report["unbacked_sentences"] == [unbacked]


def test_the_bibliography_numbers_what_the_checked_answer_marks(tmp_path: Path) -> None:
    """A rejected first quote moves the second to [1], in the text and in the references."""
    with corpus_with_dates(tmp_path) as corpus:
        chunk_id = _first_chunk(corpus).chunk_id
        report = verify_citations(
            corpus,
            [_quote(chunk_id, "words that are not in it"), _quote(chunk_id, SENTENCE)],
            answer="Magnetic flux threads the ring here [2].",
        )

    assert report["checked_answer"] == "Magnetic flux threads the ring here [1]."
    assert report["bibliography"].startswith("[1] ")
    assert "arXiv:2608.11111v1" in report["bibliography"]
    assert "Licence:" in report["bibliography"]


def test_quotes_alone_get_no_sentence_findings(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        report = verify_citations(corpus, [_quote(_first_chunk(corpus).chunk_id, SENTENCE)])

    assert report["checked_answer"] is None
    assert report["unmarked_sentences"] is None
    assert report["unbacked_sentences"] is None
    assert report["quotation_problems"] is None


def test_quotation_marks_in_a_clients_answer_are_checked_too(tmp_path: Path, monkeypatch) -> None:
    """Claude quotes papers in its own prose; those quotes were never checked."""
    # Whatever Verbatim's own answers do: this is a report on someone else's.
    monkeypatch.setattr("verbatim.config.CHECK_QUOTATIONS", False)
    with corpus_with_dates(tmp_path) as corpus:
        chunk_id = _first_chunk(corpus).chunk_id
        report = verify_citations(
            corpus,
            [_quote(chunk_id, SENTENCE)],
            answer='As written, "flux threads the ring" [1], but "flux threads every ring it meets" [1].',
        )

    assert report["checked_answer"] == (
        'As written, "flux threads the ring" [1], but flux threads every ring it meets [1].'
    )
    (problem,) = report["quotation_problems"]
    assert problem["quote"] == "flux threads every ring it meets" and problem["found_in"] is None


def test_export_renders_each_format_in_code(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        chunk_id = _first_chunk(corpus).chunk_id
        markdown = export_citations(corpus, [chunk_id])
        bibtex = export_citations(corpus, [chunk_id], format="bibtex")
        csl = export_citations(corpus, [chunk_id], format="csl-json")

    assert markdown["format"] == "markdown" and markdown["entries"] == 1
    assert "arXiv:2608.11111v1, p. 1." in markdown["references"]
    assert bibtex["references"].startswith("@misc{")
    assert json.loads(csl["references"])[0]["title"] == "Paper 2608.11111v1"


def test_export_lists_a_passage_once(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        chunk_id = _first_chunk(corpus).chunk_id
        exported = export_citations(corpus, [chunk_id, chunk_id])

    assert exported["entries"] == 1


def test_export_refuses_what_it_cannot_cite(tmp_path: Path) -> None:
    with corpus_with_dates(tmp_path) as corpus:
        chunk_id = _first_chunk(corpus).chunk_id
        with pytest.raises(ValueError, match="chunk_ids is empty"):
            export_citations(corpus, [])
        with pytest.raises(ValueError, match="no passages with these chunk_ids: feedface"):
            export_citations(corpus, [chunk_id, "feedface"])
        with pytest.raises(ValueError, match="format must be"):
            export_citations(corpus, [chunk_id], format="word")
        with pytest.raises(ValueError, match=f"at most {MAX_QUOTES}"):
            export_citations(corpus, [f"id{n}" for n in range(MAX_QUOTES + 1)])
