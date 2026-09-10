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
        "\n"
    )

    excluded = excluded_papers([str(key), str(tmp_path / "missing.jsonl")])

    # Both the versioned id and its base, so v1 in a key excludes v2 here.
    assert excluded == {"2608.00001v1", "2608.00001"}


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
