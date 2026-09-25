"""Saying what an answer rests on.

Seven detectors have failed at the covered-but-unanswered case. These tests
cover the alternative: not a verdict, but facts a reader can weigh - and, as
much, the fact deliberately left unsaid.
"""

from pathlib import Path

from verbatim.evidence import THIN_EVIDENCE_PAPERS, Evidence


def test_an_answer_says_how_much_it_rests_on() -> None:
    lines = Evidence(passages=8, papers=5).lines()

    assert lines[0] == "Answered from 8 passages in 5 papers."


def test_a_single_paper_is_called_out() -> None:
    """One paper is a voice, not a literature, whatever the scores say."""
    lines = Evidence(passages=4, papers=1).lines()

    assert "Answered from 4 passages in 1 paper." == lines[0]
    assert "All of it from a single paper." in lines
    assert THIN_EVIDENCE_PAPERS == 2


def test_the_subject_is_always_stated_not_only_when_a_threshold_trips() -> None:
    lines = Evidence(passages=6, papers=4, subject="multidimensional arrays",
                     closeness=0.91).lines()

    assert "Those passages are about: multidimensional arrays" in lines


def test_a_paper_the_subject_is_named_after_is_named() -> None:
    lines = Evidence(
        passages=6, papers=4, definitional_subject="Adam optimizer",
        named_after=("Adam: A Method for Stochastic Optimization (arXiv:1412.6980v9)",),
    ).lines()

    assert any("Named after it here: Adam" in line for line in lines)


def test_the_absence_of_a_named_paper_is_left_unsaid() -> None:
    """The negative was built and measured: it fired on all 4 no_answer cases
    and on 13 of 19 definition-shaped answerable ones. True every time, and at
    a 68% fire rate on good answers it is a line readers learn to skip - the
    same reason the subject threshold was not raised to 0.79."""
    evidence = Evidence(passages=6, papers=4, definitional_subject="nuclear fusion")

    # The fact is still available to a caller, just not printed.
    assert evidence.definitional_subject and not evidence.named_after
    assert not any("named after" in line.lower() for line in evidence.lines())


def test_nothing_is_claimed_when_nothing_was_retrieved() -> None:
    assert Evidence(passages=0, papers=0).lines() == []


def test_the_article_is_not_part_of_the_subject(tmp_path: Path) -> None:
    """"What is a tensor" asks after tensors; no paper is titled "a tensor"."""
    from verbatim.graph import definitional_subject

    assert definitional_subject("what is a tensor?") == "tensor"
    assert definitional_subject("what is the Adam optimizer?") == "Adam optimizer"
    assert definitional_subject("why does PPO clip its objective?") is None
