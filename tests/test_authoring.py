"""Writing held-out cases that the system has not been allowed to see."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from helpers import chunked_corpus
from verbatim.authoring import (
    MAX_SHARED_RUN,
    Case,
    author,
    candidate_papers,
    excluded_papers,
    longest_shared_run,
    usable_case,
)

ABSTRACT = (
    "We show that transmon qubits lose coherence through dielectric loss in "
    "the amorphous oxide at the substrate interface, and that annealing "
    "reduces the loss tangent by a factor of three."
)


def test_the_authoring_module_cannot_search() -> None:
    """The contamination rule, made structural instead of remembered.

    A held-out question must be authored without knowing how it retrieves. A
    module that cannot search cannot be tempted to check first, so the rule is
    enforced by what this file is allowed to import rather than by discipline.
    """
    source = Path("src/verbatim/authoring.py").read_text()
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = {"verbatim.retrieval", "verbatim.vectors",
                 "verbatim.embedding", "verbatim.graph",
                 "verbatim.expansion"}

    assert not (imported & forbidden), (
        f"authoring.py may not import {imported & forbidden}: a module that "
        "can retrieve can check a question before keeping it"
    )


@pytest.mark.parametrize(
    "question, expected",
    [
        ("what limits coherence in transmon qubits", 2),
        ("do transmon qubits lose coherence through dielectric loss", 7),
        ("what is the airspeed velocity of an unladen swallow", 1),
    ],
)
def test_longest_shared_run_counts_borrowed_phrasing(
    question: str, expected: int
) -> None:
    """Runs, not vocabulary overlap.

    A question about a subject necessarily shares words with an abstract about
    that subject; penalising that would reject every real question. What a
    keyword index rewards is borrowed *phrasing*, so that is what is counted.
    """
    assert longest_shared_run(question, ABSTRACT) == expected


def test_a_question_that_copies_the_abstract_is_refused() -> None:
    assert usable_case("what limits coherence in transmon qubits", ABSTRACT)
    assert not usable_case(
        "do transmon qubits lose coherence through dielectric loss in the "
        "amorphous oxide", ABSTRACT
    )


def test_the_threshold_is_the_one_the_hand_authored_set_sets() -> None:
    """Documented so it cannot drift into an arbitrary number.

    Measured over the 88 hand-authored cases: median shared run 2, max 6, and
    4 of 88 at 5 or more. Rejecting 5-or-more makes a generated set no leakier
    than the hand-authored one.
    """
    assert MAX_SHARED_RUN == 5


def test_papers_already_scored_elsewhere_are_excluded(tmp_path: Path) -> None:
    """A paper the system was tuned against cannot also be held out."""
    key = tmp_path / "dev.jsonl"
    key.write_text(
        "# a comment\n"
        '{"question": "q", "expect_papers": ["2608.00001v1"]}\n'
        '{"question": "q2", "expect_none": true}\n'
        # A two-paper key names its papers under require_papers instead.
        '{"question": "q3", "require_papers": ["2608.00002v1", "2608.00003v2"]}\n'
        "\n"
    )

    excluded = excluded_papers([str(key), str(tmp_path / "missing.jsonl")])

    # Both the versioned id and its base, so v1 in a key excludes v2 here.
    assert excluded == {
        "2608.00001v1", "2608.00001",
        "2608.00002v1", "2608.00002",
        "2608.00003v2", "2608.00003",
    }


def test_candidates_skip_excluded_papers_and_are_reproducible(
    tmp_path: Path,
) -> None:
    """A frozen key that changes when regenerated is not frozen."""
    pages = {f"2608.0000{n}v1": ["Transmon qubits and their losses."] * 30
             for n in range(1, 6)}
    with chunked_corpus(tmp_path, pages) as corpus:
        corpus.connection.execute("UPDATE papers SET summary = ?", (ABSTRACT,))
        corpus.connection.commit()
        first = candidate_papers(corpus, exclude=set(), limit=3, stride=1)
        again = candidate_papers(corpus, exclude=set(), limit=3, stride=1)
        without = candidate_papers(
            corpus, exclude={"2608.00001v1"}, limit=5, stride=1
        )

    assert [p[0] for p in first] == [p[0] for p in again]
    assert "2608.00001v1" not in [p[0] for p in without]


def test_author_rejects_and_counts_rather_than_silently_shrinking(
    tmp_path: Path,
) -> None:
    """The rejection mix is the only signal the authoring rules are working."""
    pages = {f"2608.0000{n}v1": ["Transmon qubits and their losses."] * 30
             for n in range(1, 7)}
    replies = iter([
        "Q: do transmon qubits lose coherence through dielectric loss in the oxide",
        "Q: what does this paper show",
        "Q: what is the affiliation of the first author",
        "Q: what reduces the loss tangent of a superconducting resonator",
        "Q: how much does annealing improve qubit lifetime",
    ])
    with chunked_corpus(tmp_path, pages) as corpus:
        corpus.connection.execute("UPDATE papers SET summary = ?", (ABSTRACT,))
        corpus.connection.commit()
        rejected: dict[str, int] = {}
        written = list(author(
            corpus, lambda system, user: next(replies, "Q: "),
            cases=2, exclude=set(), rejected=rejected, stride=1,
        ))

    assert len(written) == 2
    assert all(isinstance(case, Case) for case in written)
    assert sum(rejected.values()) == 3
    assert any("borrowed a" in reason for reason in rejected)
    assert "does not stand alone" in rejected
    assert "about the paper's cover" in rejected


def test_a_written_case_matches_the_key_format(tmp_path: Path) -> None:
    import json

    row = json.loads(Case("why do qubits decohere", "2608.00001v1", 2).to_row())

    assert row["question"] == "why do qubits decohere"
    assert row["expect_papers"] == ["2608.00001v1"]
    assert "heldout-generated" in row["note"]


def linked_corpus(tmp_path: Path, links: list[tuple[str, str]], papers: int):
    """Papers in the passage band, and bibliography rows linking them."""
    pages = {f"2608.{n:05d}v1": ["Transmon qubits and their losses."] * 30
             for n in range(1, papers + 1)}
    corpus = chunked_corpus(tmp_path, pages)
    corpus.connection.execute("UPDATE papers SET summary = ?", (ABSTRACT,))
    for ordinal, (citing, cited_base) in enumerate(links):
        corpus.connection.execute(
            "INSERT INTO paper_references (citing_arxiv_id, ordinal, raw_text,"
            " cited_arxiv_id, cited_doi, source_sha256, extracted_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (citing, ordinal, "a reference", cited_base, None, "sha", "2026-09-14T00:00:00Z"),
        )
    corpus.connection.commit()
    return corpus


def test_pairs_are_linked_by_citation_and_never_reuse_a_paper(tmp_path: Path) -> None:
    """Related by the corpus's own record, and no subject weighted twice.

    Paper 1 cites both 2 and 3. Only one of those pairs may be kept, because a
    paper in two questions weights its subject twice.
    """
    from verbatim.authoring import candidate_pairs

    links = [("2608.00001v1", "2608.00002"), ("2608.00001v1", "2608.00003"),
             ("2608.00003v1", "2608.00004"), ("2608.00005v1", "2608.00006")]
    with linked_corpus(tmp_path, links, papers=6) as corpus:
        pairs = candidate_pairs(corpus, exclude=set(), limit=10, stride=1)
        without = candidate_pairs(corpus, exclude={"2608.00005"}, limit=10, stride=1)

    ids = [paper[0] for pair in pairs for paper in pair]
    assert len(ids) == len(set(ids)), "a paper appeared in two pairs"
    assert {frozenset((a[0], b[0])) for a, b in pairs} == {
        frozenset(("2608.00002v1", "2608.00001v1")),
        frozenset(("2608.00004v1", "2608.00003v1")),
        frozenset(("2608.00006v1", "2608.00005v1")),
    }
    # Excluding either paper by its base identifier drops the pair.
    assert not any("2608.00005v1" in (a[0], b[0]) for a, b in without)


def test_pair_questions_are_filtered_and_every_kind_fills(tmp_path: Path) -> None:
    """The same blind filters as single-paper cases, checked against BOTH abstracts."""
    import json

    from verbatim.authoring import PAIR_KINDS, author_pairs
    from verbatim.evaluation import load_cases

    links = [(f"2608.{n:05d}v1", f"2608.{n + 1:05d}") for n in range(1, 12, 2)]
    replies = iter([
        "Q: do transmon qubits lose coherence through dielectric loss in the amorphous oxide",
        "Q: how do annealing and substrate treatment differ in reducing resonator loss",
        "Q: what does this paper show",
        "Q: what limits coherence in transmon qubits and which treatment helps most",
        "Q: what approaches exist to lowering dielectric loss in superconducting circuits",
    ])
    with linked_corpus(tmp_path, links, papers=12) as corpus:
        rejected: dict[str, int] = {}
        cases = list(author_pairs(
            corpus, lambda system, user: next(replies, "Q: "),
            per_kind=1, exclude=set(), rejected=rejected, stride=1,
        ))

    assert sorted(case.kind for case in cases) == sorted(PAIR_KINDS)
    assert any(reason.startswith("borrowed a") for reason in rejected)
    assert rejected.get("does not stand alone") == 1
    for case in cases:
        assert case.arxiv_ids[0] != case.arxiv_ids[1]
        row = json.loads(case.to_row())
        assert row["require_papers"] == list(case.arxiv_ids)
        assert "expect_papers" not in row  # "any of" would contradict "all of"

    key = tmp_path / "breadth.jsonl"
    key.write_text("".join(case.to_row() + "\n" for case in cases))
    loaded = load_cases(key)
    assert all(len(c.require_papers) == 2 for c in loaded)


def test_a_follow_up_case_must_depend_on_its_opening(tmp_path: Path) -> None:
    """A follow-up that names its own subject measures nothing.

    Retrieval would find the paper from that turn alone, and the conversation
    would never be consulted - which is exactly what these cases exist to
    test.
    """
    import json

    from verbatim.authoring import author_conversations, usable_follow_up

    assert usable_follow_up("why does it need bias correction")
    assert not usable_follow_up("why does the Adam optimizer need bias correction")

    pages = {f"2608.0000{n}v1": ["Transmon qubits and their losses."] * 30
             for n in range(1, 5)}
    replies = iter([
        # Rejected: the follow-up stands on its own.
        "Q1: what limits coherence in transmon qubits\nQ2: what reduces the loss tangent of transmon qubits\n",
        # Rejected: the opening points at the paper instead of naming a subject.
        "Q1: what does this paper measure\nQ2: why does it matter\n",
        # Kept.
        "Q1: what limits coherence in transmon qubits\nQ2: how much does annealing improve it\n",
    ])
    with chunked_corpus(tmp_path, pages) as corpus:
        corpus.connection.execute("UPDATE papers SET summary = ?", (ABSTRACT,))
        corpus.connection.commit()
        rejected: dict[str, int] = {}
        written = list(author_conversations(
            corpus, lambda system, user: next(replies, "Q1: \nQ2: \n"),
            cases=1, exclude=set(), rejected=rejected, stride=1,
        ))

    assert len(written) == 1
    row = json.loads(written[0].to_row())
    assert row["question"] == "what limits coherence in transmon qubits"
    assert row["follow_up"] == "how much does annealing improve it"
    assert row["expect_papers"] == [written[0].arxiv_id]
    assert rejected == {
        "follow-up does not depend on the opening": 1,
        "opening does not stand alone": 1,
    }
