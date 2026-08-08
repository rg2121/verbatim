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
