"""Generating training examples that the licence policy actually permits."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import chunked_corpus
from verbatim.answers import Answer, RejectedCitation
from verbatim.citations import Citation
from verbatim.corpus import ChunkMatch, CorpusStore, PaperFilter
from verbatim.distill import (
    Tally,
    attributions,
    source_papers,
    example_id,
    keep,
    questions_for,
    seed_passages,
    standalone,
)

SUBJECTS = ["dielectric loss", "flux noise", "quasiparticle poisoning",
            "two-level systems", "purcell decay", "thermal photons"]


def passage(n: int) -> str:
    """A passage about its own subject, so questions about it differ.

    Every paper saying the same thing would make every generated example
    identical, which the generator would then dedupe - and a fixture that
    hides the behaviour under test is worse than no fixture.
    """
    subject = SUBJECTS[n % len(SUBJECTS)]
    return " ".join([f"superconducting qubits decohere through {subject}"] * 20)


def seeded(tmp_path: Path, classes: dict[str, str]) -> CorpusStore:
    """A corpus whose papers carry the licence classes named."""
    corpus = chunked_corpus(
        tmp_path,
        {arxiv_id: [passage(n)] for n, arxiv_id in enumerate(classes)},
    )
    for arxiv_id, license_class in classes.items():
        corpus.connection.execute(
            "UPDATE papers SET license_class = ?, license_url = ? WHERE arxiv_id = ?",
            (license_class, f"https://example.invalid/{license_class}", arxiv_id),
        )
    corpus.connection.commit()
    return corpus


def cited(corpus: CorpusStore, arxiv_id: str) -> tuple[Citation, ChunkMatch]:
    """A citation whose quote really is in the passage it names."""
    chunk = corpus.chunks_for_paper(arxiv_id)[0]
    quote = chunk.text[:60]
    citation = Citation(
        arxiv_id=arxiv_id, base_arxiv_id=arxiv_id.split("v")[0], version=1,
        title="T", authors=["A"], page_number=1, char_start=0, char_end=60,
        quoted_text=quote, chunk_id=chunk.chunk_id, source_sha256=chunk.source_sha256,
        source_url=chunk.source_url, license_url=chunk.license_url,
        retrieved_at="2026-09-10T00:00:00Z",
    )
    return citation, ChunkMatch(chunk=chunk, score=1.0, snippet=quote)


def answer_with(citations, rejected=(), declined=False) -> Answer:
    return Answer(
        question="what causes decoherence in superconducting qubits",
        text="Dielectric loss does [1].",
        citations=list(citations),
        rejected=list(rejected),
        declined=declined,
    )


def test_only_eligible_papers_are_offered_as_seeds(tmp_path: Path) -> None:
    """The gate is on the way in, not a filter on the way out."""
    with seeded(tmp_path, {"2608.00001v1": "open",
                           "2608.00002v1": "arxiv-nonexclusive",
                           "2608.00003v1": "no-derivatives"}) as corpus:
        seeds = seed_passages(corpus, limit=10)

    assert {chunk.arxiv_id for chunk in seeds} == {"2608.00001v1"}


def test_seeds_are_reproducible_and_one_per_paper(tmp_path: Path) -> None:
    """A resumed run must draw the same passages, and not restudy one paper."""
    classes = {f"2608.0000{n}v1": "open" for n in range(1, 6)}
    with seeded(tmp_path, classes) as corpus:
        first = [c.chunk_id for c in seed_passages(corpus, limit=5, seed=7)]
        again = [c.chunk_id for c in seed_passages(corpus, limit=5, seed=7)]
        papers = [c.arxiv_id for c in seed_passages(corpus, limit=5, seed=7)]

    assert first == again
    assert len(papers) == len(set(papers))


def test_the_licence_filter_reaches_the_vector_leg_too(tmp_path: Path) -> None:
    """Both legs apply the filter, because both can return a passage.

    The predicate used to be written out separately for each leg. A licence
    gate on one of them is not a licence gate.
    """
    with seeded(tmp_path, {"2608.00001v1": "open",
                           "2608.00002v1": "arxiv-nonexclusive"}) as corpus:
        allowed = corpus.papers_matching(
            PaperFilter(license_classes=("open",)),
            ["2608.00001v1", "2608.00002v1"],
        )

    assert allowed == {"2608.00001v1"}


@pytest.mark.parametrize(
    "question, ok",
    [
        ("what does FlashAttention do about memory bandwidth", True),
        ("how does contrastive pretraining affect retrieval quality", True),
        ("what does this paper propose", False),
        ("what did the authors measure", False),
        ("what is the proposed method", False),
        ("why", False),
    ],
)
def test_a_question_must_name_its_subject(question: str, ok: bool) -> None:
    """A question pointing at its source is useless as a retrieval query."""
    assert standalone(question) is ok


def test_questions_that_point_at_the_passage_are_dropped(tmp_path: Path) -> None:
    """The model is told the rule; this is the check that it followed it."""
    with seeded(tmp_path, {"2608.00001v1": "open"}) as corpus:
        chunk = corpus.chunks_for_paper("2608.00001v1")[0]
        reply = (
            "Q: what causes decoherence in superconducting qubits\n"
            "Q: what does this paper conclude\n"
        )
        questions = questions_for(lambda system, user: reply, chunk, count=5)

    assert questions == ["what causes decoherence in superconducting qubits"]


def test_an_unverified_quote_rejects_the_whole_example(tmp_path: Path) -> None:
    """The reader-facing pipeline marks and continues. Training data may not.

    An example carrying [unverified] would teach a model to write it.
    """
    with seeded(tmp_path, {"2608.00001v1": "open"}) as corpus:
        citation, match = cited(corpus, "2608.00001v1")
        tally = Tally()
        example = keep(
            answer_with(
                [citation],
                rejected=[RejectedCitation("x", "not in the passage", "missing")],
            ),
            [match], corpus, generator_model="m", tally=tally,
        )

    assert example is None
    assert tally.rejected == {"a quote failed verification": 1}


def test_an_answer_with_no_citation_is_not_an_example(tmp_path: Path) -> None:
    with seeded(tmp_path, {"2608.00001v1": "open"}) as corpus:
        _citation, match = cited(corpus, "2608.00001v1")
        tally = Tally()
        assert keep(answer_with([]), [match], corpus,
                    generator_model="m", tally=tally) is None
        assert keep(answer_with([], declined=True), [match], corpus,
                    generator_model="m", tally=tally) is None

    assert tally.rejected == {"no citation": 1, "writer declined": 1}


def test_an_ineligible_citation_is_a_fault_and_not_a_rejection(
    tmp_path: Path,
) -> None:
    """If the gate leaks, saying so quietly is the wrong thing to do.

    Every other rejection is ordinary: a model wrote a bad quote, retrieval
    found nothing. This one means an ineligible paper reached the prompt an
    answer was written from, which is the failure the policy exists to
    prevent, so it is counted apart and labelled.
    """
    with seeded(tmp_path, {"2608.00002v1": "arxiv-nonexclusive"}) as corpus:
        citation, match = cited(corpus, "2608.00002v1")
        tally = Tally()
        example = keep(answer_with([citation]), [match], corpus,
                       generator_model="m", tally=tally)

    assert example is None
    assert tally.rejected == {}
    assert tally.faults == {"cited arxiv-nonexclusive": 1}
    assert "FAULT" in tally.render() and "licence gate leaked" in tally.render()


def test_a_kept_example_carries_what_is_needed_to_delete_it(tmp_path: Path) -> None:
    """Provenance is the policy's requirement, not a nicety."""
    with seeded(tmp_path, {"2608.00001v1": "open"}) as corpus:
        citation, match = cited(corpus, "2608.00001v1")
        tally = Tally()
        example = keep(answer_with([citation]), [match], corpus,
                       generator_model="gemma4:31b-cloud", tally=tally)

    assert example is not None
    record = example.to_dict()
    assert record["chunk_ids"] == [match.chunk.chunk_id]
    assert record["papers"] == [{
        "arxiv_id": "2608.00001v1",
        "title": "Paper 2608.00001v1",
        "authors": ["Ada Lovelace"],
        "license_url": "https://example.invalid/open",
        "license_class": "open",
    }]
    assert record["source_sha256s"] == [match.chunk.source_sha256]
    assert record["generator_model"] == "gemma4:31b-cloud"
    assert tally.kept == 1


def test_the_same_question_and_passages_get_the_same_id() -> None:
    """So a resumed run recognises what it already made."""
    assert example_id("What is X?", ["b", "a"]) == example_id("what is x? ", ["a", "b"])
    assert example_id("What is X?", ["a"]) != example_id("What is Y?", ["a"])


def test_attributions_lists_every_paper_once(tmp_path: Path) -> None:
    """CC-BY's attribution condition, satisfied for the dataset with certainty."""
    with seeded(tmp_path, {"2608.00001v1": "open"}) as corpus:
        citation, match = cited(corpus, "2608.00001v1")
        tally = Tally()
        example = keep(answer_with([citation]), [match], corpus,
                       generator_model="m", tally=tally)

    text = attributions(source_papers(
        paper for e in (example, example) for paper in e.to_dict()["papers"]
    ))

    assert text.count("arXiv:2608.00001v1") == 1
    assert "Ada Lovelace" in text
    assert "https://example.invalid/open" in text


class StubEmbedder:
    """Deterministic stand-in, so the generator runs with no model download."""

    def __init__(self) -> None:
        from verbatim.corpus import EmbeddingModel

        self.model = EmbeddingModel(
            model_id="stub", dimensions=4, normalized=True, revision="v0"
        )

    def embed_passages(self, texts):
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 0.0, 0.0, 0.0]


class QuotingCompletion:
    """Answers by quoting the first passage it was given, verifiably."""

    def __init__(self, corpus: CorpusStore) -> None:
        self.corpus = corpus
        self.asked: list[str] = []

    def complete(self, system, user, schema):
        chunk_id = user.split('chunk_id="')[1].split('"')[0]
        chunk = self.corpus.chunks_by_id([chunk_id])[0]
        return {
            "answer": "Dielectric loss dominates [1].",
            "citations": [
                {"marker": 1, "chunk_id": chunk_id, "quoted_text": chunk.text[:40]}
            ],
        }

    def complete_text(self, system, user):
        self.asked.append(user)
        subject = next(s for s in SUBJECTS if s in user)
        return f"Q: how does {subject} cause decoherence in superconducting qubits\n"


def test_generate_writes_examples_and_never_reaches_an_ineligible_paper(
    tmp_path: Path,
) -> None:
    """End to end with stubs: the gate holds, and the tally says so."""
    from verbatim.distill import generate

    classes = {"2608.00001v1": "open", "2608.00002v1": "open",
               "2608.00009v1": "arxiv-nonexclusive"}
    with seeded(tmp_path, classes) as corpus:
        completion = QuotingCompletion(corpus)
        tally = Tally()
        made = list(generate(
            corpus, StubEmbedder(), completion,
            complete_text=completion.complete_text,
            generator_model="stub", examples=2, questions_per_passage=1,
            tally=tally,
        ))

    assert len(made) == 2
    assert tally.kept == 2 and tally.faults == {}
    cited_papers = {p.arxiv_id for example in made for p in example.papers}
    assert cited_papers <= {"2608.00001v1", "2608.00002v1"}
    assert all(example.answer and example.passages for example in made)


def test_generate_does_not_remake_an_example_a_previous_run_wrote(
    tmp_path: Path,
) -> None:
    """The same seed draws the same passages, so a resume must skip by id."""
    from verbatim.distill import generate

    with seeded(tmp_path, {"2608.00001v1": "open", "2608.00002v1": "open"}) as corpus:
        completion = QuotingCompletion(corpus)
        first = list(generate(
            corpus, StubEmbedder(), completion,
            complete_text=completion.complete_text,
            generator_model="stub", examples=1, questions_per_passage=1,
            seed=3,
        ))
        tally = Tally()
        again = list(generate(
            corpus, StubEmbedder(), completion,
            complete_text=completion.complete_text,
            generator_model="stub", examples=1, questions_per_passage=1,
            seed=3, tally=tally, skip=[first[0].example_id],
        ))

    assert first and again
    assert again[0].example_id != first[0].example_id
    assert tally.rejected.get("already generated") == 1


def test_the_seed_draw_is_not_sized_to_an_assumed_keep_rate(tmp_path: Path) -> None:
    """A strict acceptance rule needs more seeds than the arithmetic suggests.

    Sizing the draw to an assumed keep rate makes a run quietly return fewer
    examples than it was asked for. Seeds cost one query; generations cost
    minutes. This asserts the generator keeps looking rather than giving up
    once its first few passages are rejected.
    """
    from verbatim.distill import generate

    classes = {f"2608.0000{n}v1": "open" for n in range(1, 7)}
    with seeded(tmp_path, classes) as corpus:
        completion = QuotingCompletion(corpus)
        refusals = {"left": 3}

        answering = completion.complete

        def sometimes(system, user, schema):
            if refusals["left"]:
                refusals["left"] -= 1
                return {"answer": "I cannot answer this.", "citations": []}
            return answering(system, user, schema)

        completion.complete = sometimes  # type: ignore[method-assign]
        tally = Tally()
        made = list(generate(
            corpus, StubEmbedder(), completion,
            complete_text=completion.complete_text,
            generator_model="stub", examples=2, questions_per_passage=1,
            tally=tally,
        ))

    assert len(made) == 2  # reached the target despite three rejections
    assert tally.rejected.get("no citation") == 3


@pytest.mark.parametrize(
    "question",
    [
        "What is the affiliation of Hakan Inan?",
        "What is the email address of Jinchuan Tian?",
        "Which institution is the Language Technologies Institute affiliated with?",
        "Who wrote the sparse attention analysis?",
    ],
)
def test_questions_about_the_papers_cover_are_dropped(question: str) -> None:
    """Front matter is a passage like any other, and it is a trap.

    A paper's first chunk carries its title, authors and affiliations, and a
    question generator handed one asks who works where. On the first 300
    examples, 83 came out bibliographic. The check is on the question because
    the passage has no clean signal: front matter shares chunk_index 0 with
    half the corpus and with most of the good seeds.
    """
    from verbatim.distill import bibliographic, usable

    assert bibliographic(question)
    assert not usable(question)


def test_a_research_question_is_kept() -> None:
    from verbatim.distill import usable

    assert usable("how does contrastive pretraining affect retrieval quality")
    assert usable("what sample complexity does projected subgradient descent reach")


def test_an_unverified_marker_rejects_the_example_even_with_nothing_rejected(
    tmp_path: Path,
) -> None:
    """An empty `rejected` list is not proof the text is clean.

    A model that writes an inline marker for a passage it never cited leaves a
    dangling reference, which is rewritten to [unverified] with nothing in
    `rejected` to show for it. Checking the list alone let two of the first
    300 examples through carrying the marker.
    """
    with seeded(tmp_path, {"2608.00001v1": "open"}) as corpus:
        citation, match = cited(corpus, "2608.00001v1")
        answer = answer_with([citation])
        answer = Answer(
            question=answer.question,
            text="Dielectric loss dominates [1], and so does something else [unverified].",
            citations=answer.citations, rejected=[], declined=False,
        )
        tally = Tally()
        example = keep(answer, [match], corpus, generator_model="m", tally=tally)

    assert example is None
    assert tally.rejected == {"an unverified marker in the text": 1}


def test_an_answer_that_disowns_its_evidence_while_citing_is_rejected(
    tmp_path: Path,
) -> None:
    """Abstain-and-cite is not a behaviour worth teaching."""
    with seeded(tmp_path, {"2608.00001v1": "open"}) as corpus:
        citation, match = cited(corpus, "2608.00001v1")
        answer = Answer(
            question="q",
            text=(
                "The provided passages do not mention Katherine Freese; "
                "however, dielectric loss dominates [1]."
            ),
            citations=[citation], rejected=[], declined=False,
        )
        tally = Tally()
        example = keep(answer, [match], corpus, generator_model="m", tally=tally)

    assert example is None
    assert tally.rejected == {"abstains while citing": 1}


def test_a_claim_about_the_paper_is_not_mistaken_for_an_abstention(
    tmp_path: Path,
) -> None:
    """"The authors do not provide a proof" is a finding, not a refusal."""
    with seeded(tmp_path, {"2608.00001v1": "open"}) as corpus:
        citation, match = cited(corpus, "2608.00001v1")
        answer = Answer(
            question="q",
            text="The authors do not provide a proof of convergence [1].",
            citations=[citation], rejected=[], declined=False,
        )
        tally = Tally()
        example = keep(answer, [match], corpus, generator_model="m", tally=tally)

    assert example is not None
    assert tally.kept == 1
