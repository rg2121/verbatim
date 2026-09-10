"""The command line, where the answering paths actually diverge.

cli.py is the largest module in the project and had no tests: every flag that
argparse accepts and the branch below quietly ignores is invisible to the rest
of the suite. These check the wiring, not the retrieval - anything needing a
model or an ONNX embedder is out of scope here by design, so the checks run
offline in milliseconds like the rest.
"""

import pytest

from verbatim.answers import Usage
from verbatim.cli import _usage_footer, run


def test_use_compare_with_agents_is_refused_rather_than_ignored(monkeypatch) -> None:
    """The audit's complaint: a flag argparse accepts and the code drops."""
    monkeypatch.setattr(
        "sys.argv", ["verbatim", "report", "a topic", "--agents", "--use-compare"]
    )

    with pytest.raises(SystemExit) as raised:
        run()

    message = str(raised.value)
    assert "--use-compare has no effect with --agents" in message
    # Refused before anything expensive is built: no model, no corpus, no ONNX.
    assert "shape router" in message


def test_the_cost_footer_reports_what_the_backend_counted() -> None:
    class Counted:
        usage = Usage()

    Counted.usage.record(
        {"prompt_eval_count": 1200, "eval_count": 340, "total_duration": 21_000_000_000}
    )

    assert _usage_footer(Counted()) == "1 model call, 1,200 tokens in, 340 out, 21.0s in the model"
    # A scripted stand-in has no usage, and asking must not raise.
    assert _usage_footer(object()) == "cost not reported"


def test_usage_totals_across_calls() -> None:
    usage = Usage()
    usage.record({"prompt_eval_count": 10, "eval_count": 5, "total_duration": 1_000_000_000})
    usage.record({"prompt_eval_count": 7, "eval_count": 3})  # a reply with no timings

    assert (usage.calls, usage.prompt_tokens, usage.output_tokens) == (2, 17, 8)
    assert usage.seconds == pytest.approx(1.0)


@pytest.mark.parametrize(
    "flag, expected",
    [
        ("--tools", "the agent is the last rung of the ladder"),
        ("--use-compare", "the shape router chooses the strategy"),
        ("--plan", "the ladder answers one question"),
    ],
)
def test_deep_refuses_the_flags_it_would_otherwise_ignore(
    monkeypatch, flag: str, expected: str
) -> None:
    """--deep subsumes each of these, so accepting them would be a lie.

    The audit's complaint, applied to the new flag before it can become one:
    a flag argparse accepts and the branch quietly drops is invisible.
    """
    monkeypatch.setattr(
        "sys.argv", ["verbatim", "ask", "what is Adam?", "--deep", flag]
    )

    with pytest.raises(SystemExit) as raised:
        run()

    assert expected in str(raised.value)
    # Refused before anything expensive is built: no model, no corpus, no ONNX.
    assert "Drop one of the two." in str(raised.value)


def test_eval_accepts_several_key_files(monkeypatch, tmp_path) -> None:
    """The frozen key is two files now, and must be scorable as one.

    Held-out cases live in two files because they were authored two ways, and
    the split has to survive into every result. Scoring only one by default
    would quietly halve the sample this exercise existed to enlarge.

    Checked with a missing second file, so the assertion lands on the loader
    and the command never reaches an ONNX embedder - this file stays offline.
    """
    import sys

    from verbatim.cli import run as cli_run

    present = tmp_path / "a.jsonl"
    present.write_text('{"question": "q one", "expect_papers": ["a1v1"]}\n')
    missing = tmp_path / "b.jsonl"

    monkeypatch.setattr(
        sys, "argv",
        ["verbatim", "eval", "--cases", str(present), str(missing)],
    )

    with pytest.raises(FileNotFoundError) as raised:
        cli_run()

    # Both paths were parsed: it got past the first file and failed on the
    # second, rather than rejecting the extra argument.
    assert "b.jsonl" in str(raised.value)
