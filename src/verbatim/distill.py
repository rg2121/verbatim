"""Training examples generated from the corpus, under the licence policy.

The long-term goal of this project is a small model that answers the way this
pipeline answers: the flat ANSWER/CITE protocol, the abstention wording, the
citation discipline. This module writes the data to teach it - not by taking
the papers' prose, which TRAINING-POLICY.md puts out of scope, but by
recording what the full pipeline did: a question, the passages retrieval
returned for it, and the answer that survived verification.

Three rules come straight from the policy and are enforced here rather than
trusted.

**Only eligible papers, and never after the fact.** The licence gate is a
`PaperFilter` pushed into retrieval, so an ineligible paper is never among the
passages an answer is written from. Discarding examples afterwards would leave
the ineligible text in the prompt that produced the kept ones. A final check
still runs over the citations, and if it ever fires it is a bug in the gate
and is reported as a fault, not as an ordinary rejection.

**Only verified answers.** An example is kept only if every citation the model
wrote survived the quote check. The pipeline's own behaviour is to mark a
failed quote `[unverified]` and carry on, which is right for a reader and
wrong for training data: an example containing `[unverified]` teaches a model
to produce it. One bad quote rejects the whole example.

**Provenance per example, sufficient to delete it.** Every record carries its
chunk ids, its papers with their licence URLs and classes, and the source
checksums, so that if a licence turns out to have been misrecorded its
examples are identifiable and removable without regenerating the rest.

Questions are seeded from real passages, because a question the corpus cannot
answer produces no example and costs a generation call to find out. Seeding
guarantees at least one paper can answer it; retrieval is then run for real,
and may or may not find that paper - which is the task the student has to
learn, not a lookup of where the question came from.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import random
import re

from verbatim.answers import Answer, Completion, answer_from_matches
from verbatim.chunking import BODY, Chunk
from verbatim.corpus import ChunkMatch, CorpusStore, PaperFilter
from verbatim.embedding import Embedder
from verbatim.retrieval import hybrid_search

# From the policy's eligibility table. `arxiv-nonexclusive` grants arXiv
# distribution rights and grants this project nothing; `no-derivatives` and
# `noncommercial` exclude themselves; `unknown` means unknown, not permitted.
ELIGIBLE_CLASSES: tuple[str, ...] = ("open",)

# A seed passage short enough to be about one thing and long enough to hold an
# answer. Measured against nothing - these are the chunker's own bounds seen
# from the other side, and are here to skip headers and reference fragments.
MIN_SEED_WORDS = 60
MAX_SEED_WORDS = 400

# Seed passages drawn per example asked for. Generous on purpose; see
# `generate`, where the reason is that seeds are free and generations
# are not.
SEED_OVERDRAW = 4

QUESTION_SYSTEM = """You are given a passage from a research paper. Write questions that this passage answers.

Rules:
- Each question must be answerable from the passage.
- Each question must stand on its own. Someone who has never seen this passage
  must be able to search for the answer using only the words of the question.
- Name the subject. Write "what does FlashAttention do about memory bandwidth",
  never "what does the proposed method do".
- Do not refer to the passage, the paper, the authors, the study, the figure or
  the table.
- Ask about the research, never about the paper's cover. No questions about who
  wrote it, where they work, their email addresses, funding or acknowledgements,
  even when the passage lists them.

Write one question per line, in this exact form, and write nothing else:

Q: <question>"""

QUESTION_LINE = re.compile(r"^\s*Q:\s*(.+?)\s*$", re.MULTILINE)

# A question that points at its source instead of naming its subject is
# useless as a retrieval query: "what does this paper propose" retrieves
# everything. The model is told not to write them; this is the check, because
# a model asked to follow a rule is not evidence that it did.
DEICTIC = re.compile(
    r"\b(?:this|the|these|those)\s+"
    r"(?:passage|paper|study|work|article|authors?|section|figure|table|"
    r"experiments?|approach|method|model|system|framework|dataset)\b"
    r"|\bthe\s+proposed\b|\bthe\s+above\b|\bhere\b",
    re.IGNORECASE,
)
MIN_QUESTION_WORDS = 5

# A passage from a paper's front matter carries its title, authors and their
# affiliations, and a question generator handed one dutifully asks who works
# where. Measured on the first 300 examples: 83 of them - 28% - were
# bibliographic trivia ("what is the email address of ...").
#
# The check is on the question rather than on the passage because the passage
# has no clean signal: front matter shares `chunk_index` 0 with 48% of the
# corpus and with most of the *good* seeds, since an HTML paper's first chunk
# usually holds the abstract too. What is wrong here is what was asked, not
# what it was asked about.
BIBLIOGRAPHIC = re.compile(
    r"\b(?:affiliation|affiliated|e-?mail|who\s+wrote|corresponding\s+author|"
    r"acknowledge(?:s|d|ment)?|funding|grant\s+number|which\s+institution|"
    r"authors?\s+of|listed\s+as)\b",
    re.IGNORECASE,
)

# An answer that says the evidence is missing while carrying citations teaches
# a student to abstain and cite at the same time. Deliberately anchored on
# references to the *supplied* evidence: "the authors do not provide a proof"
# is a claim about a paper and is fine.
DISOWNS_ITS_EVIDENCE = re.compile(
    r"\b(?:provided|given|these|the|supplied)\s+"
    r"(?:passages?|excerpts?|context|text|documents?)\s+"
    r"(?:do(?:es)?\s+not|contain\s+no|lack|make\s+no)"
    r"|\bno\s+information\s+(?:about|on|regarding)\b"
    r"|\bnot\s+(?:mentioned|discussed|present|found)\s+in\s+the\s+"
    r"(?:provided\s+)?(?:passages?|excerpts?|context)\b",
    re.IGNORECASE,
)

# The marker `renumber_markers` leaves where a citation did not survive - or
# where the model pointed at a passage it never cited at all, which produces
# the marker with an empty `rejected` list. Checking the text catches both.
UNVERIFIED_MARKER = "[unverified]"


@dataclass(frozen=True)
class SourcePaper:
    """One paper behind an example, with what permits its use."""

    arxiv_id: str
    title: str
    authors: tuple[str, ...]
    license_url: str | None
    license_class: str

    def to_dict(self) -> dict[str, object]:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": list(self.authors),
            "license_url": self.license_url,
            "license_class": self.license_class,
        }


@dataclass(frozen=True)
class TrainingExample:
    """One (question, passages, verified answer) triple and its provenance."""

    example_id: str
    question: str
    passages: tuple[tuple[str, str], ...]  # (chunk_id, text), as the model saw
    answer: str
    papers: tuple[SourcePaper, ...]
    source_sha256s: tuple[str, ...]
    generated_at: str
    generator_model: str

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(chunk_id for chunk_id, _ in self.passages)

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "question": self.question,
            "passages": [
                {"chunk_id": chunk_id, "text": text} for chunk_id, text in self.passages
            ],
            "answer": self.answer,
            "chunk_ids": list(self.chunk_ids),
            "papers": [paper.to_dict() for paper in self.papers],
            "source_sha256s": list(self.source_sha256s),
            "generated_at": self.generated_at,
            "generator_model": self.generator_model,
        }


@dataclass
class Tally:
    """What was made, and what was thrown away and why.

    Kept as a first-class result rather than a log line: the rejection mix is
    the only signal that the generator is working. A run that keeps everything
    has stopped checking, and one that keeps nothing is misconfigured.
    """

    kept: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    faults: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def fault(self, reason: str) -> None:
        """A rejection that should have been impossible. Never silent."""
        self.faults[reason] = self.faults.get(reason, 0) + 1

    @property
    def considered(self) -> int:
        return self.kept + sum(self.rejected.values()) + sum(self.faults.values())

    def render(self) -> str:
        lines = [f"  kept {self.kept} of {self.considered} candidate example(s)"]
        for reason, count in sorted(self.rejected.items(), key=lambda kv: -kv[1]):
            lines.append(f"    rejected, {reason:<22} {count}")
        for reason, count in sorted(self.faults.items(), key=lambda kv: -kv[1]):
            lines.append(f"    FAULT, {reason:<25} {count}  (the licence gate leaked)")
        return "\n".join(lines)


def seed_passages(
    corpus: CorpusStore,
    *,
    limit: int,
    seed: int = 0,
    classes: Sequence[str] = ELIGIBLE_CLASSES,
) -> list[Chunk]:
    """Body passages from eligible papers, at most one per paper.

    One per paper because a hundred passages from one paper would teach the
    student that paper rather than the behaviour, and because the corpus has
    3,806 eligible papers to draw on before it needs a second bite of any.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    placeholders = ",".join("?" * len(classes))
    rows = corpus.connection.execute(
        f"""
        SELECT c.chunk_id, MIN(c.chunk_index) AS keep
        FROM chunks AS c
        JOIN papers AS p ON p.arxiv_id = c.arxiv_id
        WHERE c.kind = ?
          AND p.is_latest = 1
          AND p.license_class IN ({placeholders})
          AND LENGTH(c.text) - LENGTH(REPLACE(c.text, ' ', '')) + 1
              BETWEEN ? AND ?
        GROUP BY c.arxiv_id
        """,
        [BODY, *classes, MIN_SEED_WORDS, MAX_SEED_WORDS],
    ).fetchall()
    chunk_ids = [row["chunk_id"] for row in rows]
    # Shuffled with a seed rather than ORDER BY RANDOM(): a run has to be
    # reproducible, and a resumed run has to pick the same passages.
    random.Random(seed).shuffle(chunk_ids)
    return corpus.chunks_by_id(chunk_ids[:limit])


def questions_for(
    complete: Callable[[str, str], str], chunk: Chunk, *, count: int = 2
) -> list[str]:
    """Questions a passage answers, kept only if they stand on their own."""
    if count < 1:
        raise ValueError("count must be at least 1")
    reply = complete(
        QUESTION_SYSTEM, f"Write {count} question(s).\n\nPASSAGE:\n{chunk.text}"
    )
    found = [match.group(1) for match in QUESTION_LINE.finditer(reply or "")]
    return [q for q in dict.fromkeys(found) if usable(q)][:count]


def usable(question: str) -> bool:
    """Whether a question is worth spending a retrieval and an answer on."""
    return standalone(question) and not bibliographic(question)


def standalone(question: str) -> bool:
    """Whether a question names its subject instead of pointing at a source."""
    if len(question.split()) < MIN_QUESTION_WORDS:
        return False
    return DEICTIC.search(question) is None


def bibliographic(question: str) -> bool:
    """Whether a question asks about the paper's cover rather than its content.

    Who wrote it, where they work and who funded it are all answerable from
    the corpus and none of them are what this pipeline is for.
    """
    return BIBLIOGRAPHIC.search(question) is not None


def build_example(
    corpus: CorpusStore,
    embedder: Embedder,
    completion: Completion,
    question: str,
    *,
    generator_model: str,
    passages: int = 6,
    max_per_paper: int = 2,
    classes: Sequence[str] = ELIGIBLE_CLASSES,
    tally: Tally,
) -> TrainingExample | None:
    """Retrieve, answer, verify, and keep only what all three agree on."""
    matches = hybrid_search(
        corpus,
        embedder,
        question,
        limit=passages,
        max_per_paper=max_per_paper,
        papers=PaperFilter(license_classes=tuple(classes)),
    )
    if not matches:
        tally.reject("no eligible passages")
        return None
    answer = answer_from_matches(corpus, completion, question, matches)
    return keep(answer, matches, corpus,
                generator_model=generator_model, classes=classes, tally=tally)


def keep(
    answer: Answer,
    matches: Sequence[ChunkMatch],
    corpus: CorpusStore,
    *,
    generator_model: str,
    classes: Sequence[str] = ELIGIBLE_CLASSES,
    tally: Tally,
) -> TrainingExample | None:
    """The acceptance rules, separated so they can be tested without a model."""
    if answer.declined:
        tally.reject("writer declined")
        return None
    if answer.rejected:
        # The reader-facing pipeline marks these [unverified] and continues.
        # Training data may not: the example would teach the marker.
        tally.reject("a quote failed verification")
        return None
    if UNVERIFIED_MARKER in answer.text:
        # An empty `rejected` list is not enough. A model that writes an
        # inline marker for a passage it never cited leaves a dangling
        # reference, which is rewritten to [unverified] with nothing rejected.
        tally.reject("an unverified marker in the text")
        return None
    if not answer.citations:
        tally.reject("no citation")
        return None
    if DISOWNS_ITS_EVIDENCE.search(answer.text):
        # "The provided passages do not mention X" carrying citations teaches
        # a student to abstain and cite at once. A real abstention has no
        # citations and is rejected above; this is the half-abstention.
        tally.reject("abstains while citing")
        return None

    cited = sorted({citation.arxiv_id for citation in answer.citations})
    papers = corpus.papers_by_id(cited)
    # `Paper` does not carry the licence class - it is derived from the
    # licence URL and stored beside it - so the classes come in one extra
    # query rather than one per citation.
    placeholders = ",".join("?" * len(cited))
    license_classes = {
        row["arxiv_id"]: row["license_class"] or "unknown"
        for row in corpus.connection.execute(
            f"SELECT arxiv_id, license_class FROM papers"
            f" WHERE arxiv_id IN ({placeholders})",
            cited,
        )
    }
    sources: list[SourcePaper] = []
    for arxiv_id in cited:
        license_class = license_classes.get(arxiv_id, "unknown")
        if license_class not in classes:
            # Impossible if the filter held. Loud, because the alternative is
            # an ineligible paper quoted in a training set.
            tally.fault(f"cited {license_class}")
            return None
        paper = papers.get(arxiv_id)
        sources.append(
            SourcePaper(
                arxiv_id=arxiv_id,
                title=paper.title if paper else "",
                authors=tuple(paper.authors) if paper else (),
                license_url=paper.license_url if paper else None,
                license_class=license_class,
            )
        )

    used = tuple((match.chunk.chunk_id, match.chunk.text) for match in matches)
    tally.kept += 1
    return TrainingExample(
        example_id=example_id(answer.question, [chunk_id for chunk_id, _ in used]),
        question=answer.question,
        passages=used,
        answer=answer.text,
        papers=tuple(sources),
        source_sha256s=tuple(
            dict.fromkeys(match.chunk.source_sha256 for match in matches)
        ),
        generated_at=datetime.now(timezone.utc).isoformat(),
        generator_model=generator_model,
    )


def example_id(question: str, chunk_ids: Sequence[str]) -> str:
    """Stable across runs, so a resumed run does not duplicate what it made."""
    digest = hashlib.sha256()
    digest.update(question.strip().lower().encode("utf-8"))
    for chunk_id in sorted(chunk_ids):
        digest.update(b"\x00" + chunk_id.encode("utf-8"))
    return digest.hexdigest()[:20]


def generate(
    corpus: CorpusStore,
    embedder: Embedder,
    completion: Completion,
    *,
    complete_text: Callable[[str, str], str],
    generator_model: str,
    examples: int,
    questions_per_passage: int = 2,
    seed: int = 0,
    passages: int = 6,
    classes: Sequence[str] = ELIGIBLE_CLASSES,
    tally: Tally | None = None,
    skip: Sequence[str] = (),
) -> Iterator[TrainingExample]:
    """Yield examples until `examples` are kept or the seed passages run out."""
    if examples < 1:
        raise ValueError("examples must be at least 1")
    tally = tally if tally is not None else Tally()
    already = set(skip)
    # Draw far more seed passages than the arithmetic needs. A passage yields
    # at most `questions_per_passage` candidates and most yield fewer - the
    # writer declines, or a quote fails verification, and the whole example
    # goes - so sizing the draw to an assumed keep rate makes the run quietly
    # return fewer examples than asked for. Seeds are one SQL query against
    # 3,806 eligible papers and cost nothing; generations are what cost. The
    # loop stops as soon as the target is met, so an over-draw is never paid
    # for.
    wanted = examples * SEED_OVERDRAW
    for chunk in seed_passages(corpus, limit=wanted, seed=seed, classes=classes):
        for question in questions_for(
            complete_text, chunk, count=questions_per_passage
        ):
            example = build_example(
                corpus, embedder, completion, question,
                generator_model=generator_model,
                passages=passages, classes=classes, tally=tally,
            )
            if example is None:
                continue
            if example.example_id in already:
                # `keep` counted it before this could be known - the id needs
                # the passages, which only retrieval can supply. Move it from
                # kept to rejected rather than letting the tally overcount.
                tally.kept -= 1
                tally.reject("already generated")
                continue
            already.add(example.example_id)
            yield example
            if tally.kept >= examples:
                return


def source_papers(records: Iterable[dict]) -> list[SourcePaper]:
    """Rebuild the papers behind a dataset from its written records.

    The dataset on disk is the whole dataset; a resumed run holds only its own
    examples in memory. ATTRIBUTIONS has to cover every example, so it is built
    from the file rather than from the run.
    """
    seen: dict[str, SourcePaper] = {}
    for record in records:
        seen.setdefault(
            str(record["arxiv_id"]),
            SourcePaper(
                arxiv_id=str(record["arxiv_id"]),
                title=str(record.get("title", "")),
                authors=tuple(record.get("authors") or ()),
                license_url=record.get("license_url"),
                license_class=str(record.get("license_class", "unknown")),
            ),
        )
    return list(seen.values())


def attributions(papers: Iterable[SourcePaper]) -> str:
    """The ATTRIBUTIONS file the policy requires of any released artifact.

    Written whether or not anything is released. It costs nothing now and
    cannot be reconstructed later if the dataset is regenerated.
    """
    seen = {paper.arxiv_id: paper for paper in papers}
    lines = [
        "Source papers for this dataset.",
        "",
        f"{len(seen)} paper(s), licence class(es): "
        + (", ".join(sorted({p.license_class for p in seen.values()})) or "none"),
        "",
    ]
    for arxiv_id, paper in sorted(seen.items()):
        authors = ", ".join(paper.authors[:8]) + (
            ", et al." if len(paper.authors) > 8 else ""
        )
        lines.append(f"arXiv:{arxiv_id}")
        lines.append(f"  {paper.title}")
        if authors:
            lines.append(f"  {authors}")
        lines.append(f"  licence: {paper.license_url or 'not recorded'}"
                     f" ({paper.license_class})")
        lines.append("")
    return "\n".join(lines)
