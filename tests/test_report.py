from pathlib import Path


from verbatim.answers import Answer
from verbatim.citations import Citation
from verbatim.corpus import EmbeddingModel
from verbatim.embedding import embed_pending
from verbatim.report import (
    Report,
    ReportSection,
    build_report,
    plan_sections,
    render_markdown,
)

from helpers import corpus_with


def test_plan_sections_parses_lines_and_survives_a_dead_model() -> None:
    planned = plan_sections(
        lambda system, user: "1. What is flux?\n- How does charge move?\n",
        "electromagnetism",
    )
    assert planned == ["What is flux?", "How does charge move?"]

    def broken(system: str, user: str) -> str:
        raise RuntimeError("model unreachable")

    assert plan_sections(broken, "electromagnetism") == ["electromagnetism"]


def test_a_section_that_starts_with_a_number_keeps_it() -> None:
    """A list marker is a number followed by '.' or ')'; '3D' and '4-bit' are not.

    Stripping every leading digit turned "4-bit quantization" into "bit
    quantization" - and a section's heading is also the query retrieval runs
    for it, so the number was lost from the search as well as the page.
    """
    planned = plan_sections(
        lambda system, user: (
            "1. 3D Gaussian splatting for scene reconstruction\n"
            "2) 4-bit quantization of language models\n"
            "- 2024 results on retrieval augmentation\n"
            "* 1.58-bit ternary weights\n"
        ),
        "efficient models",
    )
    assert planned == [
        "3D Gaussian splatting for scene reconstruction",
        "4-bit quantization of language models",
        "2024 results on retrieval augmentation",
        "1.58-bit ternary weights",
    ]


class StubEmbedder:
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


class ScriptedAnswers:
    """Answers each section with the payload scripted for its question."""

    def __init__(self, by_question: dict[str, dict]) -> None:
        self.by_question = by_question
        self.judged = 0

    def complete(self, system, user, schema):
        for question, payload in self.by_question.items():
            if question in user:
                return payload
        raise AssertionError(f"no scripted answer matches prompt: {user[:80]}")

    def judge(self, system, user):
        self.judged += 1
        return "the scripted subject"


def test_build_report_writes_each_section_from_its_own_retrieval(tmp_path: Path) -> None:
    with corpus_with(
        tmp_path,
        {
            "2608.11111v1": ("Flux Paper", ["Magnetic flux threads the ring."]),
            "2608.22222v1": ("Charge Paper", ["Charge moves along the wire."]),
        },
    ) as corpus:
        embed_pending(corpus, StubEmbedder())
        chunk_of = {c.arxiv_id: c for c in corpus.all_chunks()}
        completion = ScriptedAnswers(
            {
                "What is flux?": {
                    "relevant": True,
                    "answer": "Flux threads the ring [1].",
                    "citations": [{
                        "marker": 1,
                        "chunk_id": chunk_of["2608.11111v1"].chunk_id,
                        "quoted_text": "Magnetic flux threads the ring.",
                    }],
                },
                "How does charge move?": {
                    "relevant": True,
                    "answer": "Charge moves along wires [1].",
                    "citations": [{
                        "marker": 1,
                        "chunk_id": chunk_of["2608.22222v1"].chunk_id,
                        "quoted_text": "Charge moves along the wire.",
                    }],
                },
            }
        )
        built = build_report(
            corpus,
            StubEmbedder(),
            completion,
            lambda system, user: "What is flux?\nHow does charge move?",
            "electromagnetism",
        )

    assert [s.heading for s in built.sections] == ["What is flux?", "How does charge move?"]
    assert [s.answer.citations[0].arxiv_id for s in built.sections] == [
        "2608.11111v1", "2608.22222v1",
    ]
    # One subject check per section, each carried on the section itself.
    assert completion.judged == 2
    assert all(s.subject == "the scripted subject" for s in built.sections)


def test_a_section_the_corpus_cannot_support_says_so(tmp_path: Path) -> None:
    # No embeddings and no shared keywords: retrieval genuinely finds nothing.
    with corpus_with(
        tmp_path, {"2608.11111v1": ("Flux Paper", ["Magnetic flux threads the ring."])}
    ) as corpus:
        completion = ScriptedAnswers({})
        built = build_report(
            corpus,
            StubEmbedder(),
            completion,
            lambda system, user: "Photosynthesis in orchids?",
            "orchid biology",
        )

    section = built.sections[0]
    assert section.answer.declined
    assert section.subject is None      # nothing retrieved, nothing to describe
    assert completion.judged == 0


def citation(chunk_id: str, arxiv_id: str) -> Citation:
    return Citation(
        arxiv_id=arxiv_id, base_arxiv_id=arxiv_id.split("v")[0], version=1,
        title=f"Paper {arxiv_id}", authors=["Ada Lovelace"], page_number=1,
        char_start=0, char_end=10, quoted_text="quoted", chunk_id=chunk_id,
        source_sha256="s", source_url=f"https://arxiv.org/abs/{arxiv_id}",
        license_url=None, retrieved_at="2026-09-01T00:00:00Z",
    )


def report_with_shared_citation() -> Report:
    first = ReportSection(
        heading="What is flux?",
        answer=Answer(
            question="What is flux?", text="Flux threads rings [1].",
            citations=[citation("chunk-a", "2608.11111v1")], rejected=[],
        ),
    )
    second = ReportSection(
        heading="How does charge move?",
        answer=Answer(
            question="How does charge move?",
            text="Charge moves [1] while flux threads [2].",
            citations=[
                citation("chunk-b", "2608.22222v1"),
                citation("chunk-a", "2608.11111v1"),
            ],
            rejected=[],
        ),
    )
    return Report(topic="electromagnetism", sections=[first, second], generated_at="2026-09-01")


def test_render_markdown_numbers_one_bibliography_across_sections() -> None:
    markdown = render_markdown(report_with_shared_citation())

    # The second section's [1] is globally [2], and its [2] is the first
    # section's citation, printed once.
    assert "Flux threads rings [1]." in markdown
    assert "Charge moves [2] while flux threads [1]." in markdown
    assert markdown.count("## References") == 1
    assert markdown.count("arXiv:2608.11111v1") == 1
    assert "[1] Lovelace" in markdown and "[2] Lovelace" in markdown


def test_render_markdown_carries_the_subject_warning_per_section() -> None:
    drifted = ReportSection(
        heading="What is flux?",
        answer=Answer(question="What is flux?", text="…", citations=[], rejected=[]),
        subject="tensor slicing libraries",
        subject_closeness=0.31,
    )
    fine = ReportSection(
        heading="How does charge move?",
        answer=Answer(question="How does charge move?", text="…", citations=[], rejected=[]),
        subject="charge transport",
        subject_closeness=0.92,
    )
    markdown = render_markdown(
        Report(topic="t", sections=[drifted, fine], generated_at="2026-09-01")
    )

    assert "tensor slicing libraries" in markdown
    assert "0.31 similarity" in markdown
    assert "charge transport" not in markdown  # no warning above the threshold
