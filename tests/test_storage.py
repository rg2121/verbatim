import json
from pathlib import Path

from verbatim.arxiv import parse_feed
from verbatim.storage import load_papers, save_papers


def test_save_papers_writes_inspectable_snapshot(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "arxiv_feed.xml"
    destination = tmp_path / "nested" / "papers.json"

    save_papers(parse_feed(fixture.read_bytes()), destination)

    payload = json.loads(destination.read_text())
    assert payload["paper_count"] == 1
    assert payload["papers"][0]["arxiv_id"] == "2401.12345v2"


def test_paper_snapshot_round_trips_through_storage(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "arxiv_feed.xml"
    papers = parse_feed(fixture.read_bytes())
    destination = tmp_path / "papers.json"

    save_papers(papers, destination)

    assert load_papers(destination) == papers
