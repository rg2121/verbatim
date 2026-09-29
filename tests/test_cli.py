"""The command line, where the answering paths actually diverge."""

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
    """--deep chooses its own retrieval, so it refuses flags it would ignore."""
    monkeypatch.setattr(
        "sys.argv", ["verbatim", "ask", "what is Adam?", "--deep", flag]
    )

    with pytest.raises(SystemExit) as raised:
        run()

    assert expected in str(raised.value)
    # Refused before anything expensive is built: no model, no corpus, no ONNX.
    assert "Drop one of the two." in str(raised.value)


def test_eval_accepts_several_key_files(monkeypatch, tmp_path) -> None:
    """The frozen key is two files now, and must be scorable as one."""
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


def test_breadth_authoring_refuses_to_overwrite_a_frozen_key(monkeypatch, tmp_path) -> None:
    """Re-authoring after an arm has run is what pre-registration rules out.

    Refused before any model or corpus is built, so this stays offline.
    """
    key = tmp_path / "questions.jsonl"
    key.write_text('{"question": "q", "expect_papers": ["a1v1", "b2v1"]}\n')
    monkeypatch.setattr(
        "sys.argv", ["verbatim", "author-cases", "--breadth", "--out", str(key)]
    )

    with pytest.raises(SystemExit) as raised:
        run()

    assert "already exists" in str(raised.value)


@pytest.mark.parametrize(
    "mode, flags",
    [
        ("--deep", ["--arxiv-id", "2608.11111v1"]),
        ("--deep", ["--include-references"]),
        ("--deep", ["--no-graph"]),
        ("--tools", ["--arxiv-id", "2608.11111v1"]),
        ("--tools", ["--no-graph"]),
        ("--tools", ["--use-compare"]),
        ("--tools", ["--plan"]),
        ("--tools", ["--passages", "3"]),
        ("--tools", ["--no-subject-check"]),
    ],
)
def test_a_routed_ask_refuses_retrieval_flags_it_would_drop(
    monkeypatch, mode: str, flags: list[str]
) -> None:
    """The ladder and the tool-choosing model pick their own retrieval."""
    monkeypatch.setattr("sys.argv", ["verbatim", "ask", "what is Adam?", mode, *flags])

    with pytest.raises(SystemExit) as raised:
        run()

    assert f"has no effect with {mode}" in str(raised.value)
    assert flags[0] in str(raised.value)


def test_chunk_reports_the_passages_it_actually_wrote(monkeypatch, capsys, tmp_path) -> None:
    """The summary counts passages written, not papers."""
    from helpers import store_paper
    from verbatim.corpus import CorpusStore
    from verbatim.downloads import file_sha256
    from verbatim.html_source import HtmlSource

    database = tmp_path / "corpus.db"
    page = tmp_path / "paper.html"
    page.write_text(
        "<h1>T</h1><p>One passage.</p><h2>2 Method</h2><p>Another passage.</p>",
        encoding="utf-8",
    )
    with CorpusStore(database) as corpus:
        store_paper(corpus, tmp_path, "2608.11111v1")
        corpus.record_html_source(HtmlSource(
            arxiv_id="2608.11111v1", url="https://arxiv.org/html/2608.11111v1",
            local_path=str(page), sha256=file_sha256(page),
            byte_count=page.stat().st_size,
        ))
    monkeypatch.setattr("sys.argv", [
        "verbatim", "chunk", "--state", str(database),
        "--output", str(tmp_path / "chunks.jsonl"),
    ])

    run()

    with CorpusStore(database) as corpus:
        stored = len(list(corpus.iter_chunks()))
    assert stored == 2
    assert f"into {stored} new passage" in capsys.readouterr().out


def test_stats_counts_the_vectors_embed_actually_wrote(monkeypatch, tmp_path, capsys) -> None:
    """`embed` names its vectors by the embedder's model id - `+context` when
    passages carry their title - and `stats` counted under the bare model name,
    so it reported every passage still to embed."""
    from helpers import chunked_corpus
    from verbatim.corpus import EmbeddingModel
    from verbatim.embedding import FastEmbedEmbedder

    with chunked_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux threads the ring."]}) as corpus:
        chunk_id = list(corpus.iter_chunks())[0].chunk_id
        corpus.register_embedding_model(EmbeddingModel("stub+context", 3, True, "v0"))
        corpus.record_embeddings("stub+context", [(chunk_id, [1.0, 0.0, 0.0])])

    monkeypatch.setattr(
        "verbatim.cli.FastEmbedEmbedder",
        lambda model_name, **_: FastEmbedEmbedder(model_name=model_name, with_context=True),
    )
    monkeypatch.setattr(
        "sys.argv", ["verbatim", "stats", "--state", str(tmp_path / "corpus.db"), "--model", "stub"]
    )
    run()

    assert "0 to embed" in capsys.readouterr().out


def test_pipeline_hands_its_model_to_the_stages_it_runs(monkeypatch, tmp_path) -> None:
    """It counted pending vectors for --model and launched `embed` without it,
    so the stage embedded with the default and the count never moved."""
    captured = {}

    def capture(pending, **options):
        captured.update(options)
        return []

    monkeypatch.setattr("verbatim.cli.run_pipeline", capture)
    monkeypatch.setattr(
        "sys.argv",
        ["verbatim", "pipeline", "--state", str(tmp_path / "c.db"), "--model", "other/model"],
    )
    run()

    for stage in ("embed", "index"):
        assert captured["stage_arguments"][stage] == ("--model", "other/model")


def test_the_licence_stage_is_in_the_run_ledger(monkeypatch, tmp_path, capsys) -> None:
    """Every other stage records its runs; a failing licence lookup never
    reached `runs` or the health check, because this one recorded nothing."""
    from helpers import paper
    from verbatim.corpus import CorpusStore
    from verbatim.runs import recent_runs

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        corpus.upsert_papers([paper("2608.11111v1"), paper("2608.22222v1")])

    def lookup(base_id):
        if base_id == "2608.22222":
            raise OSError("arXiv did not answer")
        return "http://creativecommons.org/licenses/by/4.0/"

    monkeypatch.setattr("verbatim.cli.fetch_license", lookup)
    monkeypatch.setattr(
        "sys.argv",
        ["verbatim", "licenses", "--state", str(tmp_path / "corpus.db"), "--delay", "0"],
    )
    run()

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        (recorded,) = recent_runs(corpus, command="licenses")
    assert recorded["ok"] is True
    assert recorded["outputs"] == {"looked_up": 2, "found": 1, "failed": 1}


def test_every_committed_key_is_kept_out_of_new_authoring() -> None:
    """A paper a key already scores cannot be written about again."""
    from pathlib import Path

    from verbatim.cli import SCORED_KEYS

    root = Path(__file__).parent.parent
    on_disk = {
        path.relative_to(root)
        for path in [*root.glob("evals/*.jsonl"), *root.glob("evals/*/questions.jsonl")]
    }

    assert on_disk == set(SCORED_KEYS)
