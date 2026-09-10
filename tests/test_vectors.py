from pathlib import Path

import pytest

from verbatim.corpus import CorpusStore, EmbeddingModel
from verbatim.chunking import extract_pdf_chunks
from verbatim.retrieval import _similarity, vector_search
from verbatim.vectors import SqliteVecIndex, ensure_current

from helpers import chunked_corpus, downloaded, paper


class WordCountEmbedder:
    """Vectors that are trivially predictable, so ranking is checkable by hand."""

    def __init__(self) -> None:
        self.model = EmbeddingModel(
            model_id="stub", dimensions=3, normalized=True, revision="v0"
        )

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        raw = [
            float("flux" in lowered),
            float("charge" in lowered),
            float("plant" in lowered),
        ]
        norm = sum(value * value for value in raw) ** 0.5 or 1.0
        return [value / norm for value in raw]

    def embed_passages(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def embedded_corpus(tmp_path: Path, pages_by_paper):
    from verbatim.embedding import embed_pending

    corpus = chunked_corpus(tmp_path, pages_by_paper)
    embed_pending(corpus, WordCountEmbedder())
    return corpus


def test_vector_search_ranks_by_meaning_not_shared_words(tmp_path: Path) -> None:
    with embedded_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Magnetic flux threads the ring."],
            "2608.22222v1": ["A plant converts sunlight."],
        },
    ) as corpus:
        matches = vector_search(corpus, WordCountEmbedder(), "flux", limit=2)

    assert matches[0].chunk.arxiv_id == "2608.11111v1"
    assert matches[0].score > matches[1].score


def test_vector_search_can_be_scoped_to_one_paper(tmp_path: Path) -> None:
    with embedded_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Magnetic flux threads the ring."],
            "2608.22222v1": ["Charge moves through the wire."],
        },
    ) as corpus:
        scoped = vector_search(
            corpus, WordCountEmbedder(), "flux", limit=5, arxiv_id="2608.22222v1"
        )

    assert [match.chunk.arxiv_id for match in scoped] == ["2608.22222v1"]


def test_vector_search_returns_nothing_before_anything_is_embedded(tmp_path: Path) -> None:
    with chunked_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux."]}) as corpus:
        assert vector_search(corpus, WordCountEmbedder(), "flux") == []


def test_index_rebuilds_only_when_it_has_drifted(tmp_path: Path) -> None:
    with embedded_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux.", "Charge moves."]}) as corpus:
        index = SqliteVecIndex(corpus.connection, 3)
        assert ensure_current(index, corpus.embedding_count("stub"), "stub",
                              corpus.chunk_generation()) is True
        assert index.count() == 2
        # Unchanged corpus, unchanged model: the cheap path, no work done.
        assert ensure_current(index, corpus.embedding_count("stub"), "stub",
                              corpus.chunk_generation()) is False


def test_index_rejects_a_query_of_the_wrong_width(tmp_path: Path) -> None:
    with embedded_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux."]}) as corpus:
        index = SqliteVecIndex(corpus.connection, 3)
        index.rebuild("stub")

        with pytest.raises(ValueError, match="expects 3"):
            index.search([1.0, 0.0], limit=1)


def test_similarity_inverts_l2_distance_for_unit_vectors() -> None:
    assert _similarity(0.0) == pytest.approx(1.0)  # identical
    assert _similarity(2.0**0.5) == pytest.approx(0.0)  # orthogonal
    assert _similarity(2.0) == pytest.approx(-1.0)  # opposite


def test_hybrid_beats_either_leg_when_the_answer_needs_both(tmp_path: Path) -> None:
    """Exact identifier lives in one passage, the concept in another.

    Keyword alone misses the concept; vector alone misses the identifier.
    """
    from verbatim.retrieval import hybrid_search

    with embedded_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Magnetic flux threads the ring."],
            "2608.22222v1": ["Charge conservation holds in tokamak-7 exactly."],
        },
    ) as corpus:
        keyword = corpus.search_chunks("flux tokamak-7", match="any", limit=10)
        vector = vector_search(corpus, WordCountEmbedder(), "flux tokamak-7", limit=10)
        fused = hybrid_search(corpus, WordCountEmbedder(), "flux tokamak-7", limit=10)

    assert {m.chunk.arxiv_id for m in fused} == {"2608.11111v1", "2608.22222v1"}
    assert len(fused) >= max(len(keyword), len(vector))


def test_hybrid_ranks_a_passage_found_by_both_legs_first(tmp_path: Path) -> None:
    from verbatim.retrieval import hybrid_search

    with embedded_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Magnetic flux threads the ring."],
            "2608.22222v1": ["An unrelated plant grows."],
            "2608.33333v1": ["Charge moves along the wire."],
        },
    ) as corpus:
        fused = hybrid_search(corpus, WordCountEmbedder(), "flux", limit=3)

    assert fused[0].chunk.arxiv_id == "2608.11111v1"
    assert fused[0].score > fused[1].score


def test_hybrid_scopes_to_one_paper(tmp_path: Path) -> None:
    from verbatim.retrieval import hybrid_search

    with embedded_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Magnetic flux threads the ring."],
            "2608.22222v1": ["Charge moves along the wire."],
        },
    ) as corpus:
        fused = hybrid_search(
            corpus, WordCountEmbedder(), "flux charge", limit=5, arxiv_id="2608.22222v1"
        )

    assert {m.chunk.arxiv_id for m in fused} == {"2608.22222v1"}


def test_cap_per_paper_keeps_rank_order_and_promotes_other_papers(tmp_path: Path) -> None:
    from verbatim.corpus import ChunkMatch
    from verbatim.retrieval import cap_per_paper

    with chunked_corpus(
        tmp_path,
        {"2608.11111v1": ["flux one", "flux two", "flux three"], "2608.22222v1": ["charge one"]},
    ) as corpus:
        chunks = {c.arxiv_id: [] for c in corpus.all_chunks()}
        for c in corpus.all_chunks():
            chunks[c.arxiv_id].append(c)
        ranked = [
            ChunkMatch(chunk=chunks["2608.11111v1"][0], score=0.9, snippet=""),
            ChunkMatch(chunk=chunks["2608.11111v1"][1], score=0.8, snippet=""),
            ChunkMatch(chunk=chunks["2608.11111v1"][2], score=0.7, snippet=""),
            ChunkMatch(chunk=chunks["2608.22222v1"][0], score=0.6, snippet=""),
        ]

    capped = cap_per_paper(ranked, 2)

    assert [m.chunk.arxiv_id for m in capped] == [
        "2608.11111v1",
        "2608.11111v1",
        "2608.22222v1",
    ]
    assert [m.score for m in capped] == [0.9, 0.8, 0.6]


def test_hybrid_cap_promotes_a_second_paper_into_the_results(tmp_path: Path) -> None:
    from verbatim.retrieval import hybrid_search

    with embedded_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Magnetic flux one.", "Magnetic flux two.", "Magnetic flux three."],
            "2608.22222v1": ["Charge moves here."],
        },
    ) as corpus:
        uncapped = hybrid_search(corpus, WordCountEmbedder(), "flux charge", limit=3)
        capped = hybrid_search(
            corpus, WordCountEmbedder(), "flux charge", limit=3, max_per_paper=1
        )

    assert len({m.chunk.arxiv_id for m in capped}) >= len({m.chunk.arxiv_id for m in uncapped})
    assert sum(1 for m in capped if m.chunk.arxiv_id == "2608.11111v1") == 1


def test_index_filters_by_kind_and_paper_without_passing_ids(tmp_path: Path) -> None:
    """Filtering happens inside the KNN query, so it survives a large corpus."""
    from verbatim.chunking import REFERENCES
    from verbatim.corpus import CorpusStore

    with embedded_corpus(
        tmp_path,
        {
            "2608.11111v1": ["Magnetic flux threads the ring."],
            "2608.22222v1": ["Charge moves along the wire."],
        },
    ) as corpus:
        # Mark one paper's chunk as a bibliography and reindex.
        corpus.connection.execute(
            "UPDATE chunks SET kind = ? WHERE arxiv_id = ?", (REFERENCES, "2608.22222v1")
        )
        corpus.connection.commit()
        index = SqliteVecIndex(corpus.connection, 3)
        index.rebuild("stub")
        query = WordCountEmbedder().embed_query("flux charge")

        everything = index.search(query, limit=10)
        body_only = index.search(query, limit=10, kind="body")
        one_paper = index.search(query, limit=10, arxiv_id="2608.22222v1")

    assert len(everything) == 2
    assert len(body_only) == 1
    assert len(one_paper) == 1


def test_an_index_built_before_metadata_columns_is_replaced(tmp_path: Path) -> None:
    with embedded_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux."]}) as corpus:
        SqliteVecIndex(corpus.connection, 3)  # loads the sqlite-vec extension
        # Simulate the older schema: no arxiv_id or kind columns.
        corpus.connection.execute("DROP TABLE vec_chunks")
        corpus.connection.execute(
            "CREATE VIRTUAL TABLE vec_chunks USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[3])"
        )
        corpus.connection.commit()

        index = SqliteVecIndex(corpus.connection, 3)  # should detect and replace
        index.rebuild("stub")
        hits = index.search(WordCountEmbedder().embed_query("flux"), limit=5, kind="body")

    assert len(hits) == 1


def test_index_sync_adds_only_what_is_missing(tmp_path: Path) -> None:
    """Adding a paper must not reload the whole index."""
    from verbatim.embedding import embed_pending

    with embedded_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux.", "Charge moves."]}) as corpus:
        index = SqliteVecIndex(corpus.connection, 3)
        first = index.sync("stub")
        assert (first.added, first.removed) == (2, 0)

        # A second sync with nothing new must do no work at all.
        assert index.sync("stub") == type(first)(added=0, removed=0)

        # Add a paper: only its chunks are indexed.
        record = downloaded(tmp_path, "2608.22222v1")
        corpus.upsert_papers([paper("2608.22222v1")])
        corpus.record_downloads([record])
        corpus.record_chunks(
            extract_pdf_chunks([record], extract_pages=lambda _: ["## S\n\nCharge flows onward."])
        )
        embed_pending(corpus, WordCountEmbedder())
        second = index.sync("stub")

        assert second.removed == 0
        assert second.added >= 1
        assert index.count() == corpus.embedding_count("stub")


def test_index_sync_drops_vectors_whose_chunks_are_gone(tmp_path: Path) -> None:
    with embedded_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux.", "Charge moves."]}) as corpus:
        index = SqliteVecIndex(corpus.connection, 3)
        index.sync("stub")
        before = index.count()

        corpus.connection.execute("DELETE FROM embeddings WHERE model_id = 'stub'")
        corpus.connection.commit()
        change = index.sync("stub")

    assert before == 2
    assert change.removed == 2


def test_the_paper_index_finds_papers_by_subject(tmp_path: Path) -> None:
    """Nearest-neighbour over abstracts, keyed by paper rather than passage."""
    from helpers import paper
    from verbatim.corpus import CorpusStore
    from verbatim.vectors import SqlitePaperIndex

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        corpus.upsert_papers([paper("2608.11111v1"), paper("2608.22222v1")])
        corpus.register_embedding_model(
            EmbeddingModel(model_id="stub", dimensions=4, normalized=True, revision="v0")
        )
        corpus.record_paper_embeddings(
            "stub", [("2608.11111v1", [1.0, 0.0, 0.0, 0.0]),
                     ("2608.22222v1", [0.0, 1.0, 0.0, 0.0])]
        )
        index = SqlitePaperIndex(corpus.connection, dimensions=4)
        added = index.sync("stub")
        nearest = index.search([1.0, 0.0, 0.0, 0.0], limit=1)

    assert added.added == 2
    assert nearest[0][0] == "2608.11111v1"


def test_the_paper_index_drops_vectors_the_corpus_no_longer_has(tmp_path: Path) -> None:
    from helpers import paper
    from verbatim.corpus import CorpusStore
    from verbatim.vectors import SqlitePaperIndex

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        corpus.upsert_papers([paper("2608.11111v1")])
        corpus.register_embedding_model(
            EmbeddingModel(model_id="stub", dimensions=4, normalized=True, revision="v0")
        )
        corpus.record_paper_embeddings("stub", [("2608.11111v1", [1.0, 0.0, 0.0, 0.0])])
        index = SqlitePaperIndex(corpus.connection, dimensions=4)
        index.sync("stub")

        corpus.connection.execute("DELETE FROM paper_embeddings")
        corpus.connection.commit()
        change = index.sync("stub")
        remaining = index.count()

    assert (change.removed, remaining) == (1, 0)


class ConstantEmbedder:
    """Two of these differ in model id and in the vector they write."""

    def __init__(self, model_id: str, value: float) -> None:
        self.model = EmbeddingModel(
            model_id=model_id, dimensions=3, normalized=True, revision="v0"
        )
        self.value = value

    def embed_passages(self, texts):
        return [[self.value, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [self.value, 0.0, 0.0]


def test_changing_the_embedding_model_does_not_leave_the_old_vectors_searchable(
    tmp_path: Path,
) -> None:
    """The count check cannot tell one model's vectors from another's.

    `embeddings` is keyed (chunk_id, model_id) so both sets coexist, and the
    variants this project uses - `bge-small` and `bge-small+context` - have
    identical widths. Before the index recorded whose vectors it held, the
    counts matched, no sync ran, and every search answered from the previous
    model's vectors without raising anything.
    """
    from verbatim.embedding import embed_pending

    with chunked_corpus(tmp_path, {"2608.11111v1": ["One."], "2608.22222v1": ["Two."]}) as corpus:
        first, second = ConstantEmbedder("model-a", 1.0), ConstantEmbedder("model-b", -1.0)
        embed_pending(corpus, first)
        index = SqliteVecIndex(corpus.connection, 3)
        index.rebuild("model-a")
        assert index.indexed_model() == "model-a"

        embed_pending(corpus, second)
        stored = corpus.embedding_count("model-b")
        assert index.count() == stored  # the counts agree, and mean nothing

        assert ensure_current(index, stored, "model-b", corpus.chunk_generation()) is True
        assert index.indexed_model() == "model-b"

        held = corpus.connection.execute(
            "SELECT vec_to_json(embedding) FROM vec_chunks LIMIT 1"
        ).fetchone()[0]

    assert held.startswith("[-1")  # the model asked for, not the one before


def test_an_index_that_never_said_which_model_it_holds_is_rebuilt_once(
    tmp_path: Path,
) -> None:
    """A database from before this record cannot be trusted by count alone."""
    from verbatim.embedding import embed_pending

    with chunked_corpus(tmp_path, {"2608.11111v1": ["One."]}) as corpus:
        embed_pending(corpus, ConstantEmbedder("model-a", 1.0))
        index = SqliteVecIndex(corpus.connection, 3)
        index.sync("model-a")
        corpus.connection.execute("DROP TABLE vector_index_state")  # an older corpus
        stored = corpus.embedding_count("model-a")

        generation = corpus.chunk_generation()
        assert index.indexed_model() is None
        # Rebuilt to be sure: an index that never said what it holds cannot be
        # trusted to hold what is being asked for.
        assert ensure_current(index, stored, "model-a", generation) is True
        assert index.indexed_model() == "model-a"
        assert ensure_current(index, stored, "model-a", generation) is False


def test_a_replaced_source_does_not_silently_drop_out_of_vector_search(
    tmp_path: Path,
) -> None:
    """The count check cannot see ids change underneath it.

    A chunk id is a hash of the paper, its source checksum, the page and the
    chunker version. Replacing a paper's source therefore gives every one of
    its passages a new id while leaving the count alone - and the index went
    on holding the old ids, which resolve to nothing. Measured before the fix:
    a search returned 0 passages out of 1 embedded, and raised nothing.
    """
    from verbatim.downloads import DownloadedPaper, file_sha256
    from verbatim.embedding import embed_pending
    from verbatim.retrieval import vector_search

    with embedded_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux threads the ring."]}) as corpus:
        index = SqliteVecIndex(corpus.connection, 3)
        index.rebuild("stub", corpus.chunk_generation())
        first = [row[0] for row in corpus.connection.execute("SELECT chunk_id FROM vec_chunks")]

        # arXiv replaces the paper's PDF: same paper, new bytes, one passage
        # before and one after, so nothing about the count changes.
        pdf = tmp_path / "replaced.pdf"
        pdf.write_bytes(b"%PDF-1.7 DIFFERENT BYTES ENTIRELY\n%%EOF\n")
        replaced = DownloadedPaper(
            arxiv_id="2608.11111v1", source_url="u", pdf_url="p", license_url=None,
            local_path=str(pdf), sha256=file_sha256(pdf), byte_count=pdf.stat().st_size,
        )
        corpus.record_downloads([replaced])
        corpus.record_chunks(
            extract_pdf_chunks([replaced], extract_pages=lambda _: ["Magnetic flux threads the ring."]),
            for_papers=["2608.11111v1"],
        )
        embed_pending(corpus, WordCountEmbedder())

        live = [c.chunk_id for c in corpus.chunks_for_paper("2608.11111v1")]
        assert live != first                      # the ids really did change
        assert corpus.embedding_count("stub") == len(first)   # and the count did not

        found = vector_search(corpus, WordCountEmbedder(), "flux", limit=5)

    assert [m.chunk.chunk_id for m in found] == live


def test_reopening_an_index_at_a_different_width_rebuilds_it(tmp_path: Path) -> None:
    """A vec0 table declares its dimension, and CREATE IF NOT EXISTS keeps it.

    Opening a 3-wide index at 5 used to succeed and then fail deep inside an
    insert with a dimension mismatch the caller could do nothing about.
    """
    with embedded_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux."]}) as corpus:
        SqliteVecIndex(corpus.connection, 3).rebuild("stub", corpus.chunk_generation())

        wider = SqliteVecIndex(corpus.connection, 5)
        declared = corpus.connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'vec_chunks'"
        ).fetchone()[0]
        emptied = wider.count()

    assert "float[5]" in declared and "float[3]" not in declared
    assert emptied == 0          # rebuilt empty, ready for the new vectors


def test_an_index_run_leaves_nothing_for_the_next_search_to_reconcile(
    tmp_path: Path,
) -> None:
    """Indexing records what it built from, so a search after it does no work.

    Without the generation, state says 0, and the next search sees a mismatch
    and reconciles work that had just been done - correct, but wasted.
    """
    with embedded_corpus(tmp_path, {"2608.11111v1": ["Magnetic flux."]}) as corpus:
        generation = corpus.chunk_generation()
        index = SqliteVecIndex(corpus.connection, 3)
        index.rebuild("stub", generation)

        assert index.indexed_generation() == generation
        # The search after an index run finds nothing to do.
        assert ensure_current(
            index, corpus.embedding_count("stub"), "stub", generation
        ) is False

        # And an index built without one is repaired on first use rather than
        # trusted, which is the safe direction for the omission.
        index.rebuild("stub")
        assert index.indexed_generation() == 0
        assert ensure_current(
            index, corpus.embedding_count("stub"), "stub", generation
        ) is True
