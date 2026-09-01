"""Following up on an earlier question.

The invariant under test is not "the rewrite is good" - it is that the
conversation reaches retrieval and never reaches the answering model, because
text from an earlier turn cannot be verified against any paper.
"""

import pytest

from verbatim.conversation import MAX_TURNS, Turn, standalone_question


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
