"""Following up on an earlier question.

The invariant under test is not "the rewrite is good" - it is that the
conversation reaches retrieval and never reaches the answering model, because
text from an earlier turn cannot be verified against any paper.
"""


import pytest

from verbatim.conversation import (
    MAX_TURNS,
    Turn,
    refers_back,
    standalone_question,
)


def reply(text: str):
    return lambda system, user: text


def test_a_pronoun_is_resolved_against_the_conversation() -> None:
    history = [Turn("what is proximal policy optimization?", "PPO is a family of methods.")]

    asked = standalone_question(reply("Why does PPO clip its objective?"), history, "why does it clip?")

    assert asked == "Why does PPO clip its objective?"


def test_the_first_question_is_never_sent_for_rewriting() -> None:
    """With no history there is nothing to resolve, and a call would only cost
    twenty seconds and risk changing a question that was already complete."""
    def explode(system: str, user: str) -> str:
        raise AssertionError("should not have been called")

    assert standalone_question(explode, [], "what is PPO?") == "what is PPO?"


def test_an_unreachable_model_still_lets_the_question_be_asked() -> None:
    def broken(system: str, user: str) -> str:
        raise RuntimeError("Cannot reach Ollama")

    history = [Turn("what is PPO?", "A policy method.")]

    assert standalone_question(broken, history, "why does it clip?") == "why does it clip?"


def test_an_empty_or_chatty_reply_falls_back_or_is_trimmed() -> None:
    history = [Turn("what is PPO?", "A policy method.")]

    assert standalone_question(reply("   "), history, "why clip?") == "why clip?"
    # A model that starts explaining has stopped rewriting.
    chatty = standalone_question(
        reply('"Why does PPO clip?"\nI resolved "it" to PPO.'), history, "why clip?"
    )
    assert chatty == "Why does PPO clip?"


def test_only_the_recent_turns_are_sent() -> None:
    """A local rewrite call costs about a second; an essay costs far more."""
    seen: list[str] = []

    def capture(system: str, user: str) -> str:
        seen.append(user)
        return "rewritten"

    history = [Turn(f"question {n}", f"answer {n}") for n in range(20)]
    standalone_question(capture, history, "and that?")

    assert "question 19" in seen[0]
    assert "question 0" not in seen[0]
    assert seen[0].count("Q:") == MAX_TURNS + 1  # the history plus the new question


def test_a_long_answer_is_truncated_not_dropped() -> None:
    seen: list[str] = []

    def capture(system: str, user: str) -> str:
        seen.append(user)
        return "rewritten"

    history = [Turn("what is PPO?", "word " * 500)]
    standalone_question(capture, history, "why?")

    assert "what is PPO?" in seen[0]
    assert len(seen[0]) < 1500


def test_a_turn_with_no_answer_yet_still_gives_its_question() -> None:
    """The subject usually lives in the question, not the answer."""
    assert Turn("what is PPO?").render() == "Q: what is PPO?"


def test_a_question_that_stands_alone_is_not_sent_for_rewriting() -> None:
    """A rewrite can only damage a question that needed no resolving.

    It also costs a model call per turn, and the conversation is most useful
    when a follow-up is cheap.
    """
    def explode(system: str, user: str) -> str:
        raise AssertionError("a standalone question must not be rewritten")

    history = [Turn("what is PPO?", "A policy method.")]

    assert standalone_question(
        explode, history, "what limits transmon coherence times"
    ) == "what limits transmon coherence times"


@pytest.mark.parametrize(
    "question, needs_history",
    [
        ("why does it clip?", True),
        ("what about the same dataset", True),
        ("how does that compare with dense attention", True),
        ("what did the authors measure", True),
        ("why?", True),                                   # too short to carry a subject
        ("what limits transmon coherence times", False),
        ("how does masked pretraining affect retrieval quality", False),
    ],
)
def test_only_a_question_pointing_backwards_needs_the_conversation(
    question: str, needs_history: bool
) -> None:
    """Generous on purpose: an unneeded rewrite is shown, a missed one is not."""
    assert refers_back(question) is needs_history
