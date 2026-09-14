"""A real model against the real sample corpus. Opt-in, never offline.

Every other test here injects its model, which is what keeps the suite at two
seconds with no network. That also means nothing in the suite has ever proved
the whole path works against a model that can say anything. This does, when
asked:

    VERBATIM_LIVE_MODEL=gemma4:e4b-mlx uv run pytest tests/test_live.py -v

It runs against a copy of the sample, never the committed file: opening a
SQLite database writes beside it, and a smoke test that dirties the repository
is a worse problem than the one it checks for.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from verbatim.cli import run

LIVE_MODEL = os.environ.get("VERBATIM_LIVE_MODEL", "").strip()
SAMPLE = Path(__file__).resolve().parent.parent / "sample" / "corpus.db"

pytestmark = pytest.mark.skipif(
    not LIVE_MODEL,
    reason="set VERBATIM_LIVE_MODEL to an Ollama model name to run against a real model",
)


def ask(monkeypatch, capsys, tmp_path: Path, question: str) -> str:
    corpus = tmp_path / "corpus.db"
    shutil.copy(SAMPLE, corpus)
    monkeypatch.setattr(
        "sys.argv",
        ["verbatim", "ask", question, "--state", str(corpus), "--llm", LIVE_MODEL],
    )
    run()
    return capsys.readouterr().out


def test_the_flagship_question_gets_a_verified_answer(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """The README's first demo, run for real.

    The sample holds the QLoRA paper, so a working pipeline answers this with
    at least one citation that survived verification.
    """
    out = ask(monkeypatch, capsys, tmp_path, "what problem does QLoRA solve?")

    assert "References" in out, out[-1500:]
    assert "arxiv.org/abs/" in out


def test_every_reference_is_a_paper_the_sample_actually_holds(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """The README's second demo, with an assertion that can fail.

    Forty papers include nothing on neural Turing machines. A real model may
    decline or may cite nearby passages; either is allowed. What is not allowed
    is a reference to a paper outside the sample - that would mean a citation
    reached the reader without coming from the corpus, which is the one thing
    this project promises cannot happen.
    """
    import re
    import sqlite3

    out = ask(monkeypatch, capsys, tmp_path, "what is a neural turing machine?")

    # Read from the copy the run used. Opening the committed sample, even
    # read-only, would write SQLite's side files beside it.
    db = sqlite3.connect(tmp_path / "corpus.db")
    try:
        held = {row[0] for row in db.execute("SELECT arxiv_id FROM papers")}
    finally:
        db.close()

    assert out.strip(), "the command printed nothing"
    cited = set(re.findall(r"arXiv:(\d{4}\.\d{4,5}v\d+)", out))
    assert cited <= held, f"references outside the sample: {sorted(cited - held)}"
