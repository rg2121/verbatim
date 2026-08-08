from hashlib import sha256
from pathlib import Path

from verbatim.arxiv import Paper
from verbatim.corpus import CorpusStore, split_arxiv_version
from verbatim.downloads import DownloadedPaper


def paper(arxiv_id: str, *, updated: str = "2026-08-08T10:00:00Z") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        summary="A test abstract.",
        authors=["Ada Lovelace"],
        categories=["physics.gen-ph"],
        published="2026-08-01T10:00:00Z",
        updated=updated,
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        license_url=None,
    )


def download(arxiv_id: str) -> DownloadedPaper:
    content = f"PDF {arxiv_id}".encode()
    return DownloadedPaper(
        arxiv_id=arxiv_id,
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        license_url=None,
        local_path=f"data/pdfs/{arxiv_id}.pdf",
        sha256=sha256(content).hexdigest(),
        byte_count=len(content),
    )


def test_split_arxiv_version_supports_modern_and_legacy_ids() -> None:
    assert split_arxiv_version("2608.12345v2") == ("2608.12345", 2)
    assert split_arxiv_version("math/0309136v1") == ("math/0309136", 1)


def test_corpus_accumulates_versions_and_selects_only_current_undownloaded_papers(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        first = corpus.upsert_papers([paper("2608.12345v1"), paper("2608.33333v1")])
        corpus.record_downloads([download("2608.12345v1")])
        second = corpus.upsert_papers([paper("2608.12345v2", updated="2026-08-09T10:00:00Z")])

        pending = corpus.pending_downloads(10)
        all_downloads = corpus.all_downloads()

    assert (first.inserted, first.updated) == (2, 0)
    assert (second.inserted, second.updated) == (1, 0)
    assert [item.arxiv_id for item in pending] == ["2608.12345v2", "2608.33333v1"]
    assert [item.arxiv_id for item in all_downloads] == ["2608.12345v1"]


def test_corpus_refreshes_existing_metadata_without_creating_a_duplicate(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "corpus.db") as corpus:
        corpus.upsert_papers([paper("2608.12345v1")])
        summary = corpus.upsert_papers([paper("2608.12345v1", updated="2026-08-10T10:00:00Z")])
        pending = corpus.pending_downloads(1)

    assert (summary.inserted, summary.updated) == (0, 1)
    assert pending[0].updated == "2026-08-10T10:00:00Z"
