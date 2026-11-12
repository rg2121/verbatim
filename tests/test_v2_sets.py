"""The v2 evaluation's sets are drawn by frozen seeds, written once and hashed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
import arm_records  # noqa: E402
import planted_gaps  # noqa: E402
import v2_sets  # noqa: E402

from verbatim import retrieval  # noqa: E402
from verbatim.authoring import MAX_SHARED_RUN, longest_shared_run  # noqa: E402
from verbatim.gaps import checkable  # noqa: E402

from helpers import chunked_corpus  # noqa: E402

ABSTRACT = (
    "We show that transmon qubits lose coherence through dielectric loss in "
    "the amorphous oxide at the substrate interface, and that annealing "
    "reduces the loss tangent by a factor of three."
)
KEPT = "no paper measures transmon qubit coherence on sapphire above one second"
ALSO_KEPT = "no study compares annealed and unannealed fluxonium circuits at millikelvin temperatures"
BORROWED = "no paper shows that transmon qubits lose coherence through dielectric loss"


def write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# a header\n" + "".join(json.dumps(row) + "\n" for row in rows))
    return path


def questions(prefix: str, count: int) -> list[dict]:
    return [{"question": f"{prefix} question {i}", "expect_papers": [f"2608.{i:05d}v1"]}
            for i in range(count)]


def test_seeds_come_from_fixed_labels_not_from_results() -> None:
    assert v2_sets.seed("dev") == int(hashlib.sha256(b"verbatim-v2:dev").hexdigest()[:8], 16)
    assert v2_sets.seed("dev") != v2_sets.seed("human")


def test_the_dev_split_is_seeded_written_once_and_disjoint_from_test_a(tmp_path) -> None:
    first = write(tmp_path / "generated.jsonl", questions("first", 30))
    second = write(tmp_path / "generated-2.jsonl", questions("second", 5))
    dev_out, test_out = tmp_path / "v2" / "dev.jsonl", tmp_path / "v2" / "test-a.jsonl"

    dev, test = v2_sets.split(first, second, dev_out, test_out, size=6)
    assert len(dev) == 6 and len(test) == 24 + 5
    assert {row["question"] for row in dev}.isdisjoint(row["question"] for row in test)
    assert v2_sets.rows_of(dev_out) == dev and v2_sets.rows_of(test_out) == test
    with pytest.raises(SystemExit, match="written once"):
        v2_sets.split(first, second, dev_out, test_out, size=6)

    dev_out.unlink()
    test_out.unlink()
    again, _ = v2_sets.split(first, second, dev_out, test_out, size=6)
    assert again == dev


def test_claims_use_planted_gaps_prompt_and_leak_filter_and_never_search(
    tmp_path, monkeypatch
) -> None:
    assert checkable(KEPT) and checkable(ALSO_KEPT)
    assert checkable(BORROWED) and longest_shared_run(BORROWED, ABSTRACT) >= MAX_SHARED_RUN
    replies = iter([f"CLAIM: {KEPT}", f"CLAIM: {BORROWED}", "no claim line at all",
                    "CLAIM: more work is needed", f"CLAIM: {ALSO_KEPT}"])
    prompts = []

    def complete(system: str, user: str) -> str:
        prompts.append((system, user))
        return next(replies)

    def no_search(*args, **kwargs):
        raise AssertionError("claims are written from titles and abstracts only")

    monkeypatch.setattr(retrieval, "hybrid_search", no_search)
    pages = {f"2608.0000{n}v1": ["Transmon qubits and their losses."] * 30 for n in range(1, 8)}
    with chunked_corpus(tmp_path, pages) as corpus:
        corpus.connection.execute("UPDATE papers SET summary = ?", (ABSTRACT,))
        corpus.connection.commit()
        rejected: dict[str, int] = {}
        rows = v2_sets.write_claims(corpus, complete, count=2, label="t",
                                    exclude={"2608.00001v1"}, rejected=rejected)

    assert [row["claim"] for row in rows] == [KEPT, ALSO_KEPT]
    assert all(system == planted_gaps.CLAIM_SYSTEM for system, _ in prompts)
    assert all(user.startswith("TITLE: Paper ") and user.endswith(ABSTRACT) for _, user in prompts)
    assert "2608.00001v1" not in {row["arxiv_id"] for row in rows}
    # The Falsifier is given the paper's title as its topic, as planted_gaps.py gives it.
    assert all(row["topic"] == f"Paper {row['arxiv_id']}" for row in rows)
    assert rejected["no claim written"] == 1 and rejected["cannot be disproved"] == 1
    assert sum(rejected.values()) == 3
    assert arm_records.identity(rows[0]) == KEPT


def test_a_claims_file_excludes_every_paper_another_claims_file_planted(tmp_path) -> None:
    claims = write(tmp_path / "gaps-a.jsonl", [{"claim": KEPT, "topic": "t", "arxiv_id": "2608.00002v3"}])
    assert v2_sets.claimed([claims]) == {"2608.00002v3", "2608.00002"}


def test_abstention_hides_the_paper_that_answers_each_question() -> None:
    test = questions("test", 20)
    unanswerable = [{"question": "how is lager brewed?", "expect_none": True}]
    first = v2_sets.abstention(test, size=5, label="a", unanswerable=unanswerable)
    assert first == v2_sets.abstention(test, size=5, label="a", unanswerable=unanswerable)
    assert len(first) == 6 and first[-1] == unanswerable[0]
    assert all(row["hide"] == row["expect_papers"] for row in first[:5])
    assert first[:5] != v2_sets.abstention(test, size=5, label="b")


def test_your_topics_are_split_by_the_seed_not_by_hand() -> None:
    topics = [f"topic number {i}" for i in range(20)]
    a, b = v2_sets.split_topics(topics)
    assert len(a) == len(b) == 10 and set(a) | set(b) == set(topics)
    assert (a, b) == v2_sets.split_topics(topics)
    with pytest.raises(SystemExit, match="20 different topics"):
        v2_sets.split_topics(topics[:19] + [topics[0]])


def test_the_manifest_hashes_every_set_and_is_written_once(tmp_path) -> None:
    rows = questions("dev", 3)
    dev = write(tmp_path / "v2" / "dev.jsonl", rows)
    team = write(tmp_path / "breadth" / "questions.jsonl",
                 [{"question": "q", "require_papers": ["a", "b"]}])
    path = tmp_path / "v2" / "manifest.json"
    sets = {"dev": (dev, "questions"), "team-a": (team, "breadth")}
    v2_sets.write_manifest(path, sets, {"human": 7}, {"model": "m", "digest": "d"})

    entries = arm_records.load_manifest(path)
    assert arm_records.read_set(entries["dev"]) == rows
    assert entries["team-a"].arm == "team"
    payload = json.loads(path.read_text())
    assert payload["sets"]["team-a"]["path"] == "../breadth/questions.jsonl"
    assert payload["seeds"] == {"human": 7} and payload["authoring"]["model"] == "m"
    with pytest.raises(SystemExit, match="written once"):
        v2_sets.write_manifest(path, sets, {"human": 7}, {"model": "m", "digest": "d"})
