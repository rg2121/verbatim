"""The critic: describes, then has everything it says checked."""

from verbatim.answers import Answer
from verbatim.chunking import Chunk
from verbatim.contracts import Gathered
from verbatim.corpus import ChunkMatch
from verbatim.critique import critique, unbacked_sentences, uncited_sentences


def match(text: str, chunk_id: str = "aaa", arxiv_id: str = "2608.11111v1") -> ChunkMatch:
    chunk = Chunk(
        chunk_id=chunk_id, arxiv_id=arxiv_id, source_url="u", license_url=None,
        source_sha256="s", page_number=1, chunk_index=0, char_start=0,
        char_end=len(text), text=text, chunker_version="v",
    )
    return ChunkMatch(chunk=chunk, score=1.0, snippet=text)


def answer_with(text: str, citations: int = 1) -> Answer:
    class Citation:
        chunk_id = "aaa"

    return Answer(question="q", text=text, citations=[Citation()] * citations, rejected=[])


def test_a_claim_with_no_citation_is_found_without_any_model() -> None:
    """The prompt has always required a marker per claim; nothing checked."""
    text = (
        "Adam adapts the learning rate per parameter [1]. "
        "It became the default optimizer for almost all of deep learning. "
        "In summary."
    )

    uncited = uncited_sentences(text)

    assert uncited == ("It became the default optimizer for almost all of deep learning.",)
    # Too short to carry a claim, so not evidence of anything missing.
    assert "In summary." not in uncited


PASSAGE = "The model reaches 28.4 BLEU on the English-to-German translation task."


def test_a_marked_sentence_its_passage_does_not_share_is_found_without_any_model() -> None:
    """The quote check passes a real quote behind the wrong claim; the words give it away."""
    text = "Larger ensembles would generalise better across unseen languages [1]."

    assert unbacked_sentences(answer_with(text), {"aaa": PASSAGE}) == (text,)


def test_half_of_a_sentences_words_in_its_passage_is_enough() -> None:
    text = "The model reaches 28.4 BLEU, beating every larger ensemble [1]."
    other = "The model scores far worse than a larger ensemble tried elsewhere [1]."

    assert unbacked_sentences(answer_with(text), {"aaa": PASSAGE}) == ()
    # model is all this one shares.
    assert unbacked_sentences(answer_with(other), {"aaa": PASSAGE}) == (other,)


def test_a_sentence_is_measured_against_every_passage_it_cites() -> None:
    class Cited:
        def __init__(self, chunk_id):
            self.chunk_id = chunk_id

    texts = {"aaa": "Attention replaces recurrence here.", "bbb": "Dropout helps."}
    both = "Attention replaces recurrence and dropout helps generalisation [1][2]."
    one = "Attention replaces recurrence and dropout helps generalisation [2]."
    citations = [Cited("aaa"), Cited("bbb")]

    assert unbacked_sentences(Answer("q", both, citations, []), texts) == ()
    assert unbacked_sentences(Answer("q", one, citations, []), texts) == (one,)


def test_words_any_sentence_would_have_do_not_count_as_shared() -> None:
    """Otherwise a passage backs a sentence by containing "the" and "of"."""
    text = "The effect of the method can differ [1]."
    passage = "The effect of training can be seen in the tables of this section."

    # effect is all it shares once "the", "of" and "can" are set aside.
    assert unbacked_sentences(answer_with(text), {"aaa": passage}) == (text,)


def test_a_marked_sentence_too_short_to_be_a_claim_is_not_measured() -> None:
    assert unbacked_sentences(answer_with("It does so here [1]."), {"aaa": PASSAGE}) == ()


def test_an_unverified_sentence_is_a_problem_already_and_is_not_measured_again() -> None:
    text = "Larger ensembles would generalise better across unseen languages [unverified]."

    assert unbacked_sentences(answer_with(text), {"aaa": PASSAGE}) == ()


def test_a_subject_the_passages_never_mention_is_reported() -> None:
    gathered = Gathered(
        "how does MOND compare with dark matter?",
        (match("MOND modifies the acceleration law in galaxies."),),
        strategy="hybrid",
    )

    def describe(system, user):
        return "MOND\ndark matter"

    found = critique(gathered.question, answer_with("MOND modifies gravity [1]."),
                     gathered, complete_text=describe)

    assert found.unaddressed == ("dark matter",)
    assert found.action.startswith("retrieve again")


def test_a_subject_that_starts_with_a_number_is_reported_whole() -> None:
    gathered = Gathered(
        "how is 3D reconstruction evaluated?",
        (match("Evaluation uses held-out views of each scene."),),
        strategy="hybrid",
    )

    found = critique(gathered.question, answer_with("Held-out views are used [1]."),
                     gathered, complete_text=lambda s, u: "1. 3D reconstruction")

    assert found.unaddressed == ("3D reconstruction",)


def test_a_subject_the_model_invents_cannot_survive_the_check() -> None:
    """The whole reason this critic can be trusted at all."""
    gathered = Gathered(
        "what is MOND?",
        (match("MOND modifies the acceleration law. Dark matter is the alternative."),),
        strategy="hybrid",
    )

    def hallucinate(system, user):
        # Every one of these is present in the passage, or unusable.
        return "MOND\ndark matter\nacceleration\n\n   \na phrase far too long to be a subject"

    found = critique("what is MOND?", answer_with("MOND modifies gravity [1]."),
                     gathered, complete_text=hallucinate)

    assert found.unaddressed == ()
    assert found.action == "accept"


def test_a_conflict_must_name_passages_that_were_really_cited() -> None:
    gathered = Gathered(
        "q", (match("Value is 5.", "aaa"), match("Value is 9.", "bbb")), strategy="hybrid"
    )

    class Cited:
        def __init__(self, chunk_id):
            self.chunk_id = chunk_id

    answer = Answer(question="q", text="It is 5 [1] or 9 [2].",
                    citations=[Cited("aaa"), Cited("bbb")], rejected=[])

    def flag(system, user):
        # The first pair is real; the second names a passage nobody retrieved.
        return "aaa || bbb\naaa || ccc"

    found = critique("q", answer, gathered, complete_text=flag)

    assert found.conflicts == (("aaa", "bbb"),)


def test_the_critic_works_with_no_model_at_all() -> None:
    """The half that fires most often is arithmetic."""
    gathered = Gathered("q", (match("Some text."),), strategy="hybrid")

    found = critique("q", answer_with("This sentence makes a claim with no marker."),
                     gathered)

    assert found.unsupported and found.unaddressed == () and found.conflicts == ()
    assert found.action.startswith("rewrite")
    assert found.clean is False


def test_a_declined_section_is_not_criticised_for_being_uncited() -> None:
    """"The passages do not answer this" is the honest outcome, not a defect."""
    gathered = Gathered("q", (), strategy="hybrid")
    declined = Answer(question="q", text="No passages in the corpus match this question.",
                      citations=[], rejected=[], declined=True)

    found = critique("q", declined, gathered, complete_text=lambda s, u: "anything")

    assert found.clean is True and found.action == "accept"


def test_an_unreachable_model_degrades_to_the_mechanical_half() -> None:
    gathered = Gathered("q", (match("Some text."),), strategy="hybrid")

    def broken(system, user):
        raise RuntimeError("Cannot reach Ollama")

    found = critique("q", answer_with("A claim with a marker [1]."), gathered,
                     complete_text=broken)

    assert found.unaddressed == () and found.conflicts == ()
    assert found.action == "accept"
