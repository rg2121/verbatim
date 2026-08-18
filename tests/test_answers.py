from pathlib import Path

from verbatim.answers import Answer, answer_from_matches, build_prompt
from verbatim.corpus import ChunkMatch

from helpers import chunked_corpus


class ScriptedCompletion:
    """Returns a canned payload, and records what it was asked."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.system: str | None = None
        self.user: str | None = None

    def complete(self, system, user, schema):
        self.system, self.user = system, user
        return self.payload


def matches_for(corpus) -> list[ChunkMatch]:
    return [
        ChunkMatch(chunk=chunk, score=1.0, snippet=chunk.text) for chunk in corpus.all_chunks()
    ]


def test_prompt_offers_each_passage_with_its_identifier(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux threads the ring."]}) as corpus:
        matches = matches_for(corpus)

    prompt = build_prompt("what threads the ring?", matches)

    assert f'chunk_id="{matches[0].chunk.chunk_id}"' in prompt
    assert 'arxiv_id="2608.11111v1"' in prompt
    assert 'page="1"' in prompt
    assert "Magnetic flux threads the ring." in prompt
    assert "what threads the ring?" in prompt


def test_answer_attaches_a_full_citation_to_each_cited_passage(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux threads the ring."]}) as corpus:
        matches = matches_for(corpus)
        chunk_id = matches[0].chunk.chunk_id
        completion = ScriptedCompletion(
            {
                "answer": "Magnetic flux threads it [1].",
                "citations": [
                    {"marker": 1, "chunk_id": chunk_id, "quoted_text": "Magnetic flux threads"}
                ],
            }
        )
        answer = answer_from_matches(corpus, completion, "what threads the ring?", matches)

    assert isinstance(answer, Answer)
    assert answer.text == "Magnetic flux threads it [1]."
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.arxiv_id == "2608.11111v1"
    assert citation.version == 1
    assert citation.page_number == 1
    assert citation.quoted_text == "Magnetic flux threads"


def test_answer_rejects_citations_naming_a_passage_that_was_never_offered(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux threads the ring."]}) as corpus:
        matches = matches_for(corpus)
        completion = ScriptedCompletion(
            {
                "answer": "Something [1].",
                "citations": [
                    {"marker": 1, "chunk_id": "not-a-real-chunk", "quoted_text": "invented"}
                ],
            }
        )
        answer = answer_from_matches(corpus, completion, "?", matches)

    assert answer.citations == []
    assert answer.rejected[0].reason == "cited a passage that was not retrieved"


def test_answer_without_passages_does_not_call_the_model(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux."]}) as corpus:
        completion = ScriptedCompletion({"answer": "should not be used", "citations": []})
        answer = answer_from_matches(corpus, completion, "unrelated question", [])

    assert completion.user is None
    assert answer.citations == []
    assert answer.rejected == []
    assert "No passages" in answer.text


def test_a_fabricated_quote_is_rejected_not_cited(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux threads the ring."]}) as corpus:
        matches = matches_for(corpus)
        completion = ScriptedCompletion(
            {
                "answer": "The ring emits neutrinos [1].",
                "citations": [
                    {
                        "marker": 1,
                        "chunk_id": matches[0].chunk.chunk_id,
                        "quoted_text": "the ring emits neutrinos",
                    }
                ],
            }
        )
        answer = answer_from_matches(corpus, completion, "?", matches)

    assert answer.citations == []
    assert len(answer.rejected) == 1
    assert "does not appear" in answer.rejected[0].reason


def test_a_quote_differing_only_in_whitespace_still_verifies(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux threads the ring."]}) as corpus:
        matches = matches_for(corpus)
        completion = ScriptedCompletion(
            {
                "answer": "Flux threads it [1].",
                "citations": [
                    {
                        "marker": 1,
                        "chunk_id": matches[0].chunk.chunk_id,
                        "quoted_text": "Magnetic   flux\n threads",
                    }
                ],
            }
        )
        answer = answer_from_matches(corpus, completion, "?", matches)

    assert len(answer.citations) == 1
    assert answer.rejected == []


def test_genuine_and_fabricated_citations_are_separated(tmp_path: Path) -> None:
    with chunked_corpus(
        tmp_path, {"2608.11111v1": ["Magnetic flux threads the ring.", "Charge moves along wires."]}
    ) as corpus:
        matches = matches_for(corpus)
        completion = ScriptedCompletion(
            {
                "answer": "Flux threads [1] and charge moves [2], and pigs fly [3].",
                "citations": [
                    {"marker": 1, "chunk_id": matches[0].chunk.chunk_id, "quoted_text": "flux threads"},
                    {"marker": 2, "chunk_id": matches[1].chunk.chunk_id, "quoted_text": "Charge moves"},
                    {"marker": 3, "chunk_id": matches[0].chunk.chunk_id, "quoted_text": "pigs fly"},
                ],
            }
        )
        answer = answer_from_matches(corpus, completion, "?", matches)

    assert [c.quoted_text for c in answer.citations] == ["flux threads", "Charge moves"]
    assert [r.quoted_text for r in answer.rejected] == ["pigs fly"]


def test_an_empty_quote_is_rejected(tmp_path: Path) -> None:
    from verbatim.answers import verify_quote

    assert verify_quote("", "some text") == "empty quote"
    assert verify_quote("   ", "some text") == "empty quote"
    assert verify_quote("some", "some text") is None


def test_markers_written_as_chunk_ids_are_renumbered_to_match_the_bibliography(
    tmp_path: Path,
) -> None:
    """Small local models cite by chunk id; the bibliography numbers by position."""
    with chunked_corpus(
        tmp_path, {"2608.11111v1": ["Magnetic flux threads the ring.", "Charge moves along wires."]}
    ) as corpus:
        matches = matches_for(corpus)
        first, second = matches[0].chunk.chunk_id, matches[1].chunk.chunk_id
        completion = ScriptedCompletion(
            {
                "answer": f"Flux threads [{first}] and charge moves [{second}].",
                "citations": [
                    {"marker": 1, "chunk_id": first, "quoted_text": "flux threads"},
                    {"marker": 2, "chunk_id": second, "quoted_text": "Charge moves"},
                ],
            }
        )
        answer = answer_from_matches(corpus, completion, "?", matches)

    assert answer.text == "Flux threads [1] and charge moves [2]."


def test_a_numeric_marker_resolves_to_its_citations_final_position() -> None:
    """The model's second citation becomes [1] when its first was rejected."""
    from verbatim.answers import renumber_markers

    class FakeCitation:
        chunk_id = "aaa"

    assert renumber_markers("Claim [2] holds.", [FakeCitation()], {2: 1}) == "Claim [1] holds."


def test_a_marker_with_no_surviving_citation_is_flagged() -> None:
    from verbatim.answers import renumber_markers

    assert renumber_markers("Already [1] and [2].", []) == "Already [unverified] and [unverified]."


def test_renumbering_leaves_non_citation_brackets_alone() -> None:
    from verbatim.answers import renumber_markers

    assert renumber_markers("The set [a, b] is closed.", []) == "The set [a, b] is closed."


def test_renumbering_collapses_duplicate_markers() -> None:
    from verbatim.answers import renumber_markers

    class FakeCitation:
        chunk_id = "aaa"

    assert renumber_markers("Claim [1] [1] holds.", [FakeCitation()], {1: 1}) == "Claim [1] holds."
    assert (
        renumber_markers("Claim [1][1][1] holds.", [FakeCitation()], {1: 1}) == "Claim [1] holds."
    )


def test_line_protocol_parses_into_the_answer_payload() -> None:
    from verbatim.answers import parse_line_protocol

    payload = parse_line_protocol(
        "ANSWER: Flux moves [1] and charge moves [2].\n"
        "CITE [1] aaa111 :: flux can move\n"
        "CITE [2] bbb222 :: charge moves along\n"
    )

    assert payload == {
        "relevant": True,
        "answer": "Flux moves [1] and charge moves [2].",
        "citations": [
            {"marker": 1, "chunk_id": "aaa111", "quoted_text": "flux can move"},
            {"marker": 2, "chunk_id": "bbb222", "quoted_text": "charge moves along"},
        ],
    }


def test_line_protocol_accepts_an_answer_with_no_citations() -> None:
    from verbatim.answers import parse_line_protocol

    payload = parse_line_protocol("ANSWER: The passages do not address this question.")

    assert payload["citations"] == []
    assert "do not address" in payload["answer"]


def test_line_protocol_keeps_a_multi_line_answer_out_of_the_citations() -> None:
    from verbatim.answers import parse_line_protocol

    payload = parse_line_protocol(
        "ANSWER: First sentence.\nStill the answer.\nCITE [1] aaa111 :: quoted words\n"
    )

    assert payload["answer"] == "First sentence.\nStill the answer."
    assert len(payload["citations"]) == 1


def test_line_protocol_rejects_prose_that_ignored_the_format() -> None:
    from verbatim.answers import parse_line_protocol

    assert parse_line_protocol("Sure! Hybrid retrieval combines sparse and dense.") is None


def test_verification_forgives_quote_wrappers_but_not_paraphrase() -> None:
    from verbatim.answers import verify_quote

    source = "Magnetic flux can move through a region of zero magnetic field."

    # Presentation differences: the words are unchanged.
    assert verify_quote('"Magnetic flux can move"', source) is None
    assert verify_quote("Magnetic flux can move.", source) is None
    assert verify_quote('"Magnetic flux can move through a region of zero magnetic field."', source) is None
    assert verify_quote("Magnetic   flux\n can move", source) is None
    assert verify_quote("“Magnetic flux can move”", source) is None

    # Different words: still rejected.
    assert verify_quote("Magnetic flux cannot move", source) is not None
    assert verify_quote("flux moves through zero fields", source) is not None
    assert verify_quote('"the ability of models to handle complex tasks"', source) is not None
    assert verify_quote('"."', source) == "empty quote"


def test_two_quotes_from_one_passage_render_as_a_single_reference(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux threads the ring."]}) as corpus:
        matches = matches_for(corpus)
        chunk_id = matches[0].chunk.chunk_id
        completion = ScriptedCompletion(
            {
                "answer": f"Flux [{chunk_id}] threads the ring [{chunk_id}].",
                "citations": [
                    {"marker": 1, "chunk_id": chunk_id, "quoted_text": "Magnetic flux"},
                    {"marker": 2, "chunk_id": chunk_id, "quoted_text": "threads the ring"},
                ],
            }
        )
        answer = answer_from_matches(corpus, completion, "?", matches)

    assert len(answer.citations) == 1
    assert answer.text == "Flux [1] threads the ring [1]."
    assert answer.rejected == []


def test_a_marker_for_a_rejected_citation_is_labelled_unverified(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux threads the ring."]}) as corpus:
        matches = matches_for(corpus)
        chunk_id = matches[0].chunk.chunk_id
        completion = ScriptedCompletion(
            {
                "answer": f"Flux threads [{chunk_id}] the ring.",
                "citations": [
                    {"marker": 1, "chunk_id": chunk_id, "quoted_text": "invented wording"}
                ],
            }
        )
        answer = answer_from_matches(corpus, completion, "?", matches)

    assert answer.citations == []
    assert len(answer.rejected) == 1
    assert answer.text == "Flux threads [unverified] the ring."
    assert chunk_id not in answer.text


def test_an_inline_relevance_verdict_does_not_suppress_the_answer(tmp_path: Path) -> None:
    """Measured worse than no check: it declined real questions. Verdict is advisory."""
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux threads the ring."]}) as corpus:
        matches = matches_for(corpus)
        completion = ScriptedCompletion(
            {
                "relevant": True,
                "answer": "Flux threads it [1].",
                "citations": [
                    {"marker": 1, "chunk_id": matches[0].chunk.chunk_id, "quoted_text": "flux threads"}
                ],
            }
        )
        answer = answer_from_matches(corpus, completion, "?", matches)

    assert not answer.declined
    assert len(answer.citations) == 1


def test_a_missing_verdict_is_treated_as_relevant(tmp_path: Path) -> None:
    """A model that ignores the field must behave as it did before, not fall silent."""
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux threads the ring."]}) as corpus:
        matches = matches_for(corpus)
        completion = ScriptedCompletion(
            {
                "answer": "Flux threads it [1].",
                "citations": [
                    {"marker": 1, "chunk_id": matches[0].chunk.chunk_id, "quoted_text": "flux threads"}
                ],
            }
        )
        answer = answer_from_matches(corpus, completion, "?", matches)

    assert not answer.declined
    assert len(answer.citations) == 1


def test_supporting_papers_counts_distinct_sources(tmp_path: Path) -> None:
    """A lonely citation is the signature of a coincidental match."""
    with chunked_corpus(
        tmp_path,
        {"2608.11111v1": ["Magnetic flux threads.", "Flux moves onward."], "2608.22222v1": ["Flux again here."]},
    ) as corpus:
        matches = matches_for(corpus)
        by_paper = {}
        for m in matches:
            by_paper.setdefault(m.chunk.arxiv_id, []).append(m.chunk.chunk_id)
        one_paper = ScriptedCompletion({
            "relevant": True, "answer": "a [1] b [2].",
            "citations": [
                {"marker": 1, "chunk_id": by_paper["2608.11111v1"][0], "quoted_text": "Magnetic flux"},
                {"marker": 2, "chunk_id": by_paper["2608.11111v1"][1], "quoted_text": "Flux moves"},
            ]})
        two_papers = ScriptedCompletion({
            "relevant": True, "answer": "a [1] b [2].",
            "citations": [
                {"marker": 1, "chunk_id": by_paper["2608.11111v1"][0], "quoted_text": "Magnetic flux"},
                {"marker": 2, "chunk_id": by_paper["2608.22222v1"][0], "quoted_text": "Flux again"},
            ]})
        lonely = answer_from_matches(corpus, one_paper, "?", matches)
        corroborated = answer_from_matches(corpus, two_papers, "?", matches)

    assert lonely.supporting_papers == 1
    assert corroborated.supporting_papers == 2


def test_the_line_protocol_still_reads_a_verdict_when_one_is_given() -> None:
    from verbatim.answers import parse_line_protocol

    payload = parse_line_protocol("RELEVANT: no\nANSWER: The corpus is about something else.")

    assert payload["relevant"] is False
    assert payload["citations"] == []


class SubjectCompletion(ScriptedCompletion):
    """A completion whose judge() returns a fixed subject description."""

    def __init__(self, payload: dict, subject: str) -> None:
        super().__init__(payload)
        self.subject = subject
        self.judged: str | None = None

    def judge(self, system, user):
        self.judged = user
        return self.subject


def test_the_subject_call_never_mentions_the_question(tmp_path: Path) -> None:
    """Asked whether passages are relevant, every model tested said yes."""
    from verbatim.answers import describe_subject

    with chunked_corpus(tmp_path, {"2608.11111v1": ["Multidimensional arrays in deep learning."]}) as corpus:
        matches = matches_for(corpus)
        completion = SubjectCompletion({}, "Multidimensional arrays")

        subject = describe_subject(completion, matches)

    assert subject == "Multidimensional arrays"
    assert "Multidimensional arrays in deep learning." in completion.judged
    assert "?" not in completion.judged  # no question leaked into the prompt


def test_subject_similarity_is_arithmetic_not_judgement() -> None:
    from verbatim.answers import subject_similarity

    class UnitEmbedder:
        def embed_query(self, text):
            return [1.0, 0.0] if "flux" in text else [0.0, 1.0]

    assert subject_similarity(UnitEmbedder(), "flux", "flux physics") == 1.0
    assert subject_similarity(UnitEmbedder(), "flux", "neutron stars") == 0.0


def test_a_long_subject_is_truncated_and_whitespace_collapsed(tmp_path: Path) -> None:
    from verbatim.answers import describe_subject

    with chunked_corpus(tmp_path, {"2608.11111v1": ["Body text here."]}) as corpus:
        completion = SubjectCompletion({}, "  a   very\nlong " + "x" * 300)
        subject = describe_subject(completion, matches_for(corpus))

    assert len(subject) <= 120
    assert "\n" not in subject and "  " not in subject
