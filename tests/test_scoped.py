"""M2b: what an agent can read of the corpus, and the tools it reads it with.

ScopedCorpus is an allow-list, so a hidden paper's id and text can never reach an
agent (I11): not through any tool in TOOL_SCHEMAS, the other tools.py reads,
INSPECT, or the opening search. With nothing hidden it changes nothing.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import operator
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict

import pytest

from helpers import chunked_corpus, paper
from verbatim import tools
from verbatim.agents import toolbox
from verbatim.agents.scoped import ALLOWED, ScopedCorpus
from verbatim.chunking import REFERENCES
from verbatim.references import Reference
from verbatim.retrieval import default_search_options, hybrid_search

HIDDEN, NEWER = "2608.99999v1", "2608.99999v2"  # one paper, hidden in both versions
CITER, CITED, OTHER = "2608.11111v1", "2608.22222v1", "2608.33333v1"
SECRET = "phonons"  # only the hidden paper's text holds this word
QUERY = "quasicrystal phonons zone boundary"


class NoVectors:
    """An embedder for a corpus that holds no embeddings: the vector leg finds nothing."""

    model = SimpleNamespace(model_id="none", dimensions=3)


EMBEDDER = NoVectors()


@pytest.fixture
def store(tmp_path: Path):
    corpus = chunked_corpus(tmp_path, {
        CITER: ["## Intro\n\nMagnetic flux threads the ring in every sample.",
                # Its bibliography names the hidden paper, by title and id.
                "## References\n\n[1] A. Writer. Quasicrystal phonons. arXiv:2608.99999."],
        CITED: ["## Method\n\nThe ring is cooled below one kelvin before measuring."],
        OTHER: ["## Notes\n\nQuasicrystals grow slowly from the melt."],
        HIDDEN: ["## Result\n\nQuasicrystal phonons soften near the zone boundary.",
                 "## More\n\nThe phonons harden again at low temperature."],
        NEWER: ["## Result\n\nQuasicrystal phonons soften sharply near the zone boundary."],
    })
    with corpus:
        corpus.upsert_papers([paper(NEWER, updated="2026-09-01T10:00:00Z")])  # the newest paper
        corpus.record_references([
            Reference(CITER, 1, "[1] arXiv:2608.99999", "2608.99999"),
            Reference(CITER, 2, "[2] arXiv:2608.22222", "2608.22222"),
        ], source_sha256="citer")
        corpus.record_references([
            Reference(NEWER, 1, "[1] arXiv:2608.22222", "2608.22222"),
            Reference(NEWER, 2, "[2] arXiv:2601.00001", "2601.00001"),
        ], source_sha256="hidden")
        yield corpus


def hidden_chunks(store) -> list[str]:
    return [chunk.chunk_id for version in (HIDDEN, NEWER) for chunk in store.chunks_for_paper(version)]


# Every tool an agent is offered, each asked for the hidden paper some way.
PROBES = [
    ("search_chunks", {"query": QUERY}),
    ("search_chunks", {"query": SECRET, "arxiv_id": "2608.99999"}),
    ("get_paper", {"identifier": HIDDEN}),
    ("get_paper", {"identifier": "2608.99999"}),
    ("get_paper", {"identifier": "Paper 2608.99999"}),
    ("list_papers", {"limit": 50}),
    ("get_citations", {"identifier": CITER, "direction": "references"}),
    ("get_citations", {"identifier": CITED, "direction": "cited_by"}),
    ("get_citations", {"identifier": "2608.99999", "direction": "cited_by"}),
    ("read_pages", {"arxiv_id": NEWER, "pages": [1, 2]}),
    ("read_pages", {"arxiv_id": CITER, "pages": [1, 2]}),  # a bibliography that names it
]
# The other ways an agent reads: tools.py's MCP reads, INSPECT and the opening search.
READS: dict[str, Callable[[ScopedCorpus, list[str]], Any]] = {
    "corpus_overview": lambda corpus, ids: tools.corpus_overview(corpus),
    "paper_record": lambda corpus, ids: tools.paper_record(corpus, NEWER),
    # Quoting the hidden words other than SECRET: the reply echoes the quote back.
    "verify_citations": lambda corpus, ids: tools.verify_citations(
        corpus, [{"chunk_id": chunk_id, "quote": "soften near the zone boundary"}
                 for chunk_id in ids]),
    "export_citations": lambda corpus, ids: tools.export_citations(corpus, ids),
    "inspect": lambda corpus, ids: toolbox.inspect(corpus, ids[:4]),
    "opening search": lambda corpus, ids: corpus.search(EMBEDDER, QUERY, limit=8,
                                                        **default_search_options()),
}


def seen(read: Callable[[], Any], asked: tuple[str, ...] = ()) -> str:
    """What a read shows, with the caller's own words taken out of any error."""
    try:
        return json.dumps(read(), default=str)
    except ValueError as error:  # tools refuse bad arguments with ValueError
        message = str(error)
        for value in asked:
            message = message.replace(value, "<asked>")
        return message


def test_every_tool_offered_is_probed() -> None:
    assert {name for name, _ in PROBES} == {schema["name"] for schema in tools.TOOL_SCHEMAS}


@pytest.mark.parametrize(("name", "arguments"), PROBES)
def test_no_tool_shows_a_hidden_paper(store, name: str, arguments: dict) -> None:
    asked = tuple(str(value) for value in arguments.values())
    # The probe does reach the paper when nothing is hidden ...
    shown = seen(lambda: toolbox.call(name, arguments, ScopedCorpus(store), EMBEDDER))
    assert "2608.99999" in shown or SECRET in shown
    # ... and finds nothing of it, neither its id nor its words, once it is hidden.
    hidden = seen(lambda: toolbox.call(name, arguments, ScopedCorpus(store, [HIDDEN]), EMBEDDER), asked)
    assert "2608.99999" not in hidden and SECRET not in hidden


@pytest.mark.parametrize("read", READS)
def test_no_other_read_shows_a_hidden_paper(store, read: str) -> None:
    ids = hidden_chunks(store)
    shown = seen(lambda: READS[read](ScopedCorpus(store), ids))
    assert "2608.99999" in shown or SECRET in shown
    hidden = seen(lambda: READS[read](ScopedCorpus(store, ["2608.99999"]), ids), tuple(ids))
    assert "2608.99999" not in hidden and SECRET not in hidden


def test_a_paper_is_hidden_in_every_version(store) -> None:
    scoped = ScopedCorpus(store, [HIDDEN])  # hiding v1 hides v2 too
    assert scoped.get_paper(NEWER) is None and scoped.latest_version_of("2608.99999") is None
    assert scoped.hidden == ("2608.99999",)


def test_a_method_nobody_listed_is_not_there(store) -> None:
    scoped = ScopedCorpus(store, [HIDDEN])
    for name in ("connection", "search_chunks", "iter_chunks", "upsert_papers", "record_references"):
        with pytest.raises(AttributeError):
            getattr(scoped, name)
    public = {name for name in dir(ScopedCorpus) if not name.startswith("_")}
    assert public == {*ALLOWED, "search", "hidden", "for_thread"}


def method_calls(store) -> dict[str, tuple[tuple, dict]]:
    """Each listed method, called with arguments that name the hidden paper where it can."""
    every_chunk = [chunk.chunk_id for chunk in store.iter_chunks()]
    return {
        "get_paper": ((CITER,), {}), "papers_by_id": (([CITER, HIDDEN],), {}),
        "latest_version_of": (("2608.99999",), {}),
        "papers_cited_by": ((CITER,), {"limit": 5}),
        "citation_counts": ((["2608.99999", "2608.22222"],), {}),
        "chunks_by_id": ((every_chunk,), {}), "reference_count": ((), {}),
        "passage_span": ((HIDDEN,), {}), "papers_updated_since": ((), {"limit": 3}),
        "papers_citing": (("2608.22222",), {"limit": 5}),
        "most_cited_held_papers": ((), {"limit": 5}),
        "most_cited_absent_papers": ((), {"limit": 5}), "holdings": ((), {}),
        "find_papers_by_title": (("Paper 2608",), {"limit": 3}),
        "citation_graph_stats": ((), {}), "chunks_on_pages": ((HIDDEN, [1, 2]), {}),
        "chunks_for_paper": ((HIDDEN,), {}),
    }


def test_no_method_shows_a_hidden_paper(store) -> None:
    scoped = ScopedCorpus(store, [HIDDEN])
    for name, (args, kwargs) in method_calls(store).items():
        shown = repr(getattr(scoped, name)(*args, **kwargs))
        assert "2608.99999" not in shown and SECRET not in shown, name
    # Asked about by id, a hidden paper answers exactly as one the corpus does not hold.
    absent = {HIDDEN: "2601.00002v1", NEWER: "2601.00002v2", "2608.99999": "2601.00002"}
    for name, args in (("get_paper", (HIDDEN,)), ("latest_version_of", ("2608.99999",)),
                       ("passage_span", (HIDDEN,)), ("chunks_for_paper", (NEWER,)),
                       ("chunks_on_pages", (NEWER, [1])), ("papers_cited_by", (NEWER,)),
                       ("papers_citing", ("2608.99999",)), ("citation_counts", (["2608.99999"],))):
        swapped = tuple(absent.get(arg, arg) if isinstance(arg, str) else
                        [absent.get(item, item) for item in arg] if isinstance(arg, list) else arg
                        for arg in args)
        assert getattr(scoped, name)(*args) == getattr(store, name)(*swapped), name


def test_with_nothing_hidden_every_method_returns_what_the_store_does(store) -> None:
    calls = method_calls(store)
    assert set(calls) == set(ALLOWED)
    scoped = ScopedCorpus(store)
    for name, (args, kwargs) in calls.items():
        assert getattr(scoped, name)(*args, **kwargs) == getattr(store, name)(*args, **kwargs), name


def test_with_nothing_hidden_search_is_v1s_search(store) -> None:
    # An empty scope passes no filter, so the title and graph legs run as in v1.
    options = {"limit": 5, **default_search_options()}
    assert ScopedCorpus(store).search(EMBEDDER, QUERY, **options) == hybrid_search(
        store, EMBEDDER, QUERY, **options)
    for arguments in ({"query": "ring"}, {"query": "ring", "arxiv_id": CITER, "limit": 2}):
        assert toolbox.call("search_chunks", arguments, ScopedCorpus(store), EMBEDDER) == (
            tools.search_chunks(store, EMBEDDER, arguments["query"], limit=arguments.get("limit", 8),
                                arxiv_id=arguments.get("arxiv_id")))


def test_a_hidden_paper_does_not_take_a_place_in_a_limited_list(store) -> None:
    # The hidden paper is the newest, so a list of two would hold only one other.
    shown = [p.arxiv_id for p in ScopedCorpus(store, [HIDDEN]).papers_updated_since(limit=2)]
    assert shown == [CITER, CITED]


def test_inspect_reads_at_most_four_passages_sharing_its_characters(store, monkeypatch) -> None:
    ids = [chunk.chunk_id for paper_id in (CITER, CITED, OTHER) for chunk in store.chunks_for_paper(paper_id)]
    texts = {chunk.chunk_id: chunk.text for chunk in store.chunks_by_id(ids)}
    scoped = ScopedCorpus(store)
    with pytest.raises(ValueError, match="at most 4"):
        toolbox.inspect(scoped, [*ids, *hidden_chunks(store)][:5])
    # Passages run near 1,200 characters, so four fit in 6,000; a smaller budget shows the cut.
    monkeypatch.setattr(toolbox, "INSPECT_CHARACTERS", 60)
    rows = toolbox.inspect(scoped, [ids[2], ids[0], ids[2]])
    assert [row["chunk_id"] for row in rows] == [ids[2], ids[0]]  # in the order asked, once
    assert [row["text"] for row in rows] == [texts[chunk_id][:30 - len(toolbox.CUT)] + toolbox.CUT
                                            for chunk_id in (ids[2], ids[0])]


# -- what the review of M2b found ---------------------------------------------------------


def test_a_parallel_branch_reads_through_a_copy_of_the_scope_on_its_own_thread(store) -> None:
    import sqlite3
    from concurrent.futures import ThreadPoolExecutor

    scoped = ScopedCorpus(store, [HIDDEN])

    def branch() -> tuple:
        with scoped.for_thread() as mine:
            return mine.hidden, mine.get_paper(CITER), mine.get_paper(NEWER)

    with ThreadPoolExecutor(1) as pool:
        # The store's connection belongs to the thread that opened it ...
        with pytest.raises(sqlite3.ProgrammingError):
            pool.submit(scoped.get_paper, CITER).result()
        # ... so a branch opens its own, hiding the same papers.
        hidden, shown, gone = pool.submit(branch).result()
    assert hidden == ("2608.99999",) and shown.arxiv_id == CITER and gone is None


def test_langgraph_branches_each_read_through_their_own_copy(store) -> None:
    from dataclasses import dataclass

    from langgraph.graph import END, START, StateGraph
    from langgraph.runtime import Runtime
    from langgraph.types import Send

    @dataclass
    class Deps:
        scope: ScopedCorpus

    class State(TypedDict, total=False):
        found: Annotated[list[str], operator.add]

    def branch(brief: dict, runtime: Runtime[Deps]) -> dict:
        with runtime.context.scope.for_thread() as mine:
            return {"found": [p.arxiv_id for p in mine.papers_updated_since(limit=5)]}

    graph = StateGraph(State, context_schema=Deps)
    graph.add_node("branch", branch)
    graph.add_conditional_edges(START, lambda state: [Send("branch", {"i": i}) for i in range(3)],
                                ["branch"])
    graph.add_edge("branch", END)
    out = graph.compile().invoke({}, context=Deps(ScopedCorpus(store, [HIDDEN])))
    assert out["found"] == [CITER, CITED, OTHER] * 3


def test_a_hidden_paper_a_question_is_named_after_stays_out_of_the_title_and_graph_legs(
        tmp_path: Path) -> None:
    from helpers import titled_paper

    anchor, first, second = "2608.50001v1", "2608.50002v1", "2608.50003v1"
    corpus = chunked_corpus(tmp_path, {
        anchor: ["## Method\n\nFlux pinning holds vortices in place at defects in the lattice."],
        first: ["## Intro\n\nWe measure transport in thin films of niobium."],
        second: ["## Intro\n\nWe image vortices with a scanning probe."],
    })
    question = "What is flux pinning?"
    with corpus:
        corpus.upsert_papers([titled_paper(anchor, "Flux Pinning")])
        for citer in (first, second):
            corpus.record_references([Reference(citer, 1, "[1] arXiv:2608.50001", "2608.50001")],
                                     source_sha256=citer)

        def found(scope: ScopedCorpus) -> set[str]:
            opening = scope.search(EMBEDDER, question, limit=8, **default_search_options())
            titled = scope.search(EMBEDDER, question, limit=8, use_titles=True)
            tool = toolbox.search_chunks(scope, EMBEDDER, question)
            return ({m.chunk.arxiv_id for m in (*opening, *titled)}
                    | {row["arxiv_id"] for row in tool})

        assert anchor in found(ScopedCorpus(corpus))
        assert anchor not in found(ScopedCorpus(corpus, [anchor]))


def test_nothing_the_scope_counts_comes_from_a_hidden_paper(store) -> None:
    scoped = ScopedCorpus(store, [HIDDEN])
    # 2601.00001 is in the hidden paper's bibliography and nobody else's.
    assert scoped.most_cited_absent_papers(limit=10) == []
    # CITED is cited by CITER and by the hidden paper, so one citer is shown.
    assert scoped.citation_counts(["2608.22222"]) == {"2608.22222": 1}
    cited_by = toolbox.call("get_citations", {"identifier": CITED, "direction": "cited_by"},
                            scoped, EMBEDDER)
    assert cited_by["total"] == len(cited_by["papers"]) == 1
    assert scoped.most_cited_held_papers(limit=10) == [(store.get_paper(CITED), 1)]
    holdings = scoped.holdings()
    assert (holdings["papers"], holdings["passages"], holdings["updated_up_to"]) == (
        3, 4, "2026-08-08")
    assert scoped.citation_graph_stats() == {"edges": 1, "connected_papers": 2, "papers": 3}
    assert scoped.reference_count() == 2  # CITER's two entries, not the hidden paper's


@pytest.mark.parametrize("spelling", ["arXiv:2608.99999v1", " 2608.99999v1 ", "2608.99999",
                                      "https://arxiv.org/abs/2608.99999v1",
                                      "https://arxiv.org/pdf/2608.99999v2.pdf"])
def test_a_hidden_paper_is_hidden_however_its_id_is_written(store, spelling: str) -> None:
    scoped = ScopedCorpus(store, [spelling])
    assert scoped.hidden == ("2608.99999",) and scoped.get_paper(NEWER) is None


def test_a_hidden_list_names_held_papers_and_is_a_list(store) -> None:
    with pytest.raises(ValueError, match="2608.9999 "):
        ScopedCorpus(store, ["2608.9999v1"])  # mistyped: no such paper is held
    with pytest.raises(TypeError, match="list"):
        ScopedCorpus(store, HIDDEN)  # one string, which would be read as characters


def test_inspect_takes_its_limit_from_the_contract(store, monkeypatch) -> None:
    from verbatim.agents import contract
    from verbatim.agents.state import Inspect

    ids = [chunk.chunk_id for chunk in store.iter_chunks() if not chunk.arxiv_id.startswith("2608.9")]
    monkeypatch.setattr(contract, "MAX_INSPECT", 2)
    assert contract.lead_action_refused(Inspect(tuple(ids[:3])), board=set(ids), attempts=[])
    with pytest.raises(ValueError, match="at most 2"):
        toolbox.inspect(ScopedCorpus(store), ids[:3])


def test_the_scope_offers_exactly_the_store_methods_tools_py_calls() -> None:
    import ast

    tree = ast.parse(Path(tools.__file__).read_text(encoding="utf-8"))
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name) and node.func.value.id == "corpus"}
    assert called == set(ALLOWED)


def test_the_copied_search_refuses_bad_arguments_as_v1s_does(store) -> None:
    for limit, arxiv_id in ((0, "2601.00009"), (8, "2601.00009"), (0, None)):
        with pytest.raises(ValueError) as v1:
            tools.search_chunks(store, EMBEDDER, "flux", limit=limit, arxiv_id=arxiv_id)
        with pytest.raises(ValueError) as copied:
            toolbox.search_chunks(ScopedCorpus(store), EMBEDDER, "flux", limit=limit,
                                  arxiv_id=arxiv_id)
        assert str(copied.value) == str(v1.value)


def test_a_list_asks_again_while_hidden_versions_fill_it(store) -> None:
    store.upsert_papers([paper(HIDDEN, updated="2026-08-31T10:00:00Z")])  # both versions newest
    shown = ScopedCorpus(store, [HIDDEN]).papers_updated_since(limit=1, latest_only=False)
    assert [p.arxiv_id for p in shown] == [CITER]


def test_while_a_paper_is_hidden_no_reference_list_is_read(store) -> None:
    # Other papers' bibliographies name it, so none is read while anything is hidden;
    # mentions in body text stay. With nothing hidden they read as in v1.
    def references(scope: ScopedCorpus) -> list[str]:
        searched = [m.chunk for m in scope.search(EMBEDDER, "Quasicrystal phonons", limit=8,
                                                  include_references=True)]
        read = [*scope.chunks_for_paper(CITER), *scope.chunks_on_pages(CITER, [2]),
                *scope.chunks_by_id([chunk.chunk_id for chunk in store.chunks_for_paper(CITER)])]
        return [chunk.chunk_id for chunk in (*searched, *read) if chunk.kind == REFERENCES]

    assert len(references(ScopedCorpus(store))) == 4
    assert references(ScopedCorpus(store, [HIDDEN])) == []
