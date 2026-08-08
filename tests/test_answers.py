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


def test_answer_drops_citations_naming_a_passage_that_was_never_offered(tmp_path: Path) -> None:
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


def test_answer_without_passages_does_not_call_the_model(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux."]}) as corpus:
        completion = ScriptedCompletion({"answer": "should not be used", "citations": []})
        answer = answer_from_matches(corpus, completion, "unrelated question", [])

    assert completion.user is None
    assert answer.citations == []
    assert "No passages" in answer.text
