from verbatim.html_source import parse_html_sections

# Shaped like a real arXiv HTML page: chrome, a feedback dialog, sectioning
# commands from the author's LaTeX, and maths carrying its own source.
PAGE = """
<html><head><style>body{}</style></head><body>
<nav class="ltx_page_navbar">Skip to content</nav>
<div class="modal-form"><h5 class="modal-title">Report GitHub Issue</h5>
  <label for="t">Title:</label><p>Content selection saved. Describe the issue below:</p>
</div>
<h1>Fission Product Yields</h1>
<p>Authors and affiliations here.</p>
<h2>Abstract</h2>
<p>We predict yields for <math alttext="^{235}U"><msup><mi>U</mi></msup></math> nuclei.</p>
<h2>1 Introduction</h2>
<p>Fission releases energy when <math alttext="\\beta"><mi>&#946;</mi></math> decay follows.</p>
<h3>1.1 Method</h3>
<p>A Bayesian network is trained.</p>
<script>var tracking = 1;</script>
</body></html>
"""


def test_headings_become_a_section_hierarchy() -> None:
    sections = parse_html_sections(PAGE)

    paths = [" > ".join(s.path) for s in sections]
    assert "Fission Product Yields > Abstract" in paths
    assert "Fission Product Yields > 1 Introduction > 1.1 Method" in paths


def test_maths_is_kept_as_its_original_latex() -> None:
    """The whole point of reading HTML rather than the PDF."""
    sections = parse_html_sections(PAGE)
    text = " ".join(s.text for s in sections)

    assert "$^{235}U$" in text
    assert "$\\beta$" in text
    # The rendered glyph must not survive alongside the source.
    assert "β" not in text


def test_arxiv_chrome_is_not_mistaken_for_content() -> None:
    """The feedback dialog reads like prose and would otherwise be indexed."""
    text = " ".join(s.text for s in parse_html_sections(PAGE))

    assert "Report GitHub Issue" not in text
    assert "Content selection saved" not in text
    assert "Skip to content" not in text
    assert "tracking" not in text


def test_offsets_advance_so_a_citation_can_address_a_span() -> None:
    sections = parse_html_sections(PAGE)

    starts = [s.char_start for s in sections]
    assert starts == sorted(starts)
    assert starts[0] == 0
    assert len(set(starts)) == len(starts)


def test_a_page_with_no_headings_still_yields_its_text() -> None:
    sections = parse_html_sections("<html><body><p>Body only, no headings.</p></body></html>")

    assert len(sections) == 1
    assert sections[0].path == ()
    assert "Body only" in sections[0].text


def test_an_empty_page_yields_nothing() -> None:
    assert parse_html_sections("<html><body></body></html>") == []


def test_a_deeper_heading_nests_and_a_shallower_one_pops() -> None:
    sections = parse_html_sections(
        "<h2>A</h2><p>one</p><h3>A.1</h3><p>two</p><h2>B</h2><p>three</p>"
    )

    assert [" > ".join(s.path) for s in sections] == ["A", "A > A.1", "B"]


def _source(tmp_path, html: str):
    from hashlib import sha256
    from verbatim.html_source import HtmlSource

    path = tmp_path / "paper.html"
    path.write_text(html, encoding="utf-8")
    data = path.read_bytes()
    return HtmlSource(
        arxiv_id="2608.11111v1", url="https://arxiv.org/html/2608.11111v1",
        local_path=str(path), sha256=sha256(data).hexdigest(), byte_count=len(data),
    )


def test_html_chunks_are_located_by_section_not_by_page(tmp_path) -> None:
    """arXiv HTML has no pages, so a page number would be invented."""
    from verbatim.html_source import extract_html_chunks

    chunks = extract_html_chunks(
        [_source(tmp_path, "<h1>T</h1><p>One.</p><h2>2 Method</h2><p>Two.</p>")],
        chunk_size=200, overlap=20,
    )

    assert [c.page_number for c in chunks] == [1, 2]      # section ordinals
    assert all(c.source_kind == "html" for c in chunks)


def test_html_chunks_verify_the_bytes_before_emitting_text(tmp_path) -> None:
    import pytest
    from verbatim.html_source import extract_html_chunks

    source = _source(tmp_path, "<p>Original.</p>")
    (tmp_path / "paper.html").write_text("<p>Tampered.</p>", encoding="utf-8")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        extract_html_chunks([source])


def test_an_html_chunk_id_cannot_collide_with_a_pdf_one(tmp_path) -> None:
    """Both forms of the same paper coexist; their ids must stay distinct."""
    from verbatim.html_source import extract_html_chunks

    chunks = extract_html_chunks([_source(tmp_path, "<h1>T</h1><p>Body.</p>")])
    assert chunks
    assert all(len(c.chunk_id) == 20 for c in chunks)
    # Same paper, section and position, but derived from a different namespace.
    from verbatim.html_source import _html_chunk_id
    source = _source(tmp_path, "<h1>T</h1><p>Body.</p>")
    from verbatim.chunking import chunker_fingerprint
    assert _html_chunk_id(source, 1, 0, chunker_fingerprint(1200, 200)) == chunks[0].chunk_id


def test_a_citation_from_html_says_section_not_page() -> None:
    from verbatim.chunking import Chunk
    from verbatim.citations import citation_from_chunk, render_citation
    from helpers import paper

    def rendered(source_kind: str) -> str:
        chunk = Chunk(
            chunk_id="c", arxiv_id="2608.11111v1", source_url="u", license_url=None,
            source_sha256="s", page_number=3, chunk_index=0, char_start=0, char_end=4,
            text="text", chunker_version="6:1200:200", source_kind=source_kind,
        )
        return render_citation(1, citation_from_chunk(chunk, paper("2608.11111v1"), "text"))

    assert "p. 3" in rendered("pdf")
    assert "section 3" in rendered("html")
    assert "p. 3" not in rendered("html")


def test_maths_in_a_skipped_region_does_not_silence_the_rest_of_the_paper() -> None:
    """A table of contents sits inside <nav> and routinely contains maths.

    Its <math> tags used to be counted as open while their end tags were
    discarded with the rest of the skipped region, so the parser believed it
    was inside an equation for the remainder of the file and dropped every
    heading and every character after it. One paper lost 89% of its text.
    """
    from verbatim.html_source import parse_html_sections

    sections = parse_html_sections(
        "<nav><ul><li><a>2 The $g$ term</a>"
        '<math alttext="g_{\\mu\\nu}">g</math></li></ul></nav>'
        "<h2>2 The metric</h2><p>The body survives.</p>"
    )

    assert [s.path for s in sections] == [("2 The metric",)]
    assert sections[0].text == "The body survives."


def test_a_nested_element_does_not_close_a_skipped_region_early() -> None:
    from verbatim.html_source import parse_html_sections

    sections = parse_html_sections(
        "<nav><div><div>menu text</div>still menu</div></nav><p>Real body.</p>"
    )

    assert [s.text for s in sections] == ["Real body."]


def test_latex_subscripts_and_operators_survive_intact() -> None:
    """Markdown cleaning read _ as emphasis and ate every subscript.

    $g_{\\mu\\nu}$ was stored as $g{\\mu\\nu}$ - a different tensor, and not one
    the author wrote. HTML carries no Markdown, so it must not be stripped.
    """
    from verbatim.html_source import parse_html_sections

    (section,) = parse_html_sections(
        '<p>The metric <math alttext="g_{\\mu\\nu}">g</math> and '
        '<math alttext="a * b">ab</math> and <math alttext="a &lt; b">a</math>.</p>'
    )

    assert "$g_{\\mu\\nu}$" in section.text
    assert "$a * b$" in section.text
    assert "$a < b$" in section.text


def test_joining_inline_elements_does_not_leave_gaps_before_punctuation() -> None:
    from verbatim.html_source import parse_html_sections, tidy_spacing

    (section,) = parse_html_sections(
        "<p>a Yang-Mills connection <cite>[<a>6</a>]</cite> .</p>"
    )

    assert section.text == "a Yang-Mills connection [6]."
    # Maths is never reformatted: the LaTeX is why HTML is read at all.
    assert tidy_spacing(r"see ( a ) with $( a , b )$ .") == r"see (a) with $( a , b )$."


def test_a_truncated_rendering_is_measured_as_incomplete() -> None:
    """arXiv publishes partial output when LaTeXML fails partway through."""
    from verbatim.html_source import MIN_COMPLETENESS, completeness

    assert completeness(11_586, 140_826) < MIN_COMPLETENESS  # a real case
    assert completeness(21_368, 15_403) >= MIN_COMPLETENESS  # LaTeX adds characters
    # Nothing to compare against is not evidence against the rendering.
    assert completeness(5_000, 0) == 1.0


def test_an_unusable_rendering_keeps_its_bytes_and_its_reason(tmp_path) -> None:
    from helpers import store_paper
    from verbatim.chunking import chunker_fingerprint
    from verbatim.corpus import CorpusStore
    from verbatim.html_source import HtmlSource

    with CorpusStore(tmp_path / "corpus.db") as corpus:
        store_paper(corpus, tmp_path, "2608.11111v1")
        page = tmp_path / "2608.11111v1.html"
        page.write_text("<p>Only the abstract.</p>", encoding="utf-8")
        corpus.record_html_source(
            HtmlSource(arxiv_id="2608.11111v1", url="https://arxiv.org/html/2608.11111v1",
                       local_path=str(page), sha256="c" * 64, byte_count=page.stat().st_size),
            usable=False, note="rendering covers 8% of the PDF text",
        )

        row = corpus.connection.execute(
            "SELECT * FROM html_sources WHERE arxiv_id = ?", ("2608.11111v1",)
        ).fetchone()
        # Not offered as a source, but the paper still gets chunked from its PDF.
        pending_html = corpus.html_sources_pending_chunking(chunker_fingerprint(1200, 200))
        pending_pdf = corpus.pending_chunking(chunker_fingerprint(1200, 200))

    assert row["usable"] == 0
    assert row["sha256"] == "c" * 64  # the evidence for the decision is kept
    assert "8%" in row["note"]
    assert pending_html == []
    assert [d.arxiv_id for d in pending_pdf] == ["2608.11111v1"]


def test_a_bibliography_is_read_entry_by_entry() -> None:
    from verbatim.references import parse_bibliography

    refs = parse_bibliography(
        '<ul class="ltx_biblist">'
        '<li id="bib.bib1" class="ltx_bibitem">'
        '<span class="ltx_tag ltx_tag_bibitem">[1]</span>'
        '<span class="ltx_bibblock">D. Kingma and J. Ba, "Adam", arXiv:1412.6980.</span></li>'
        '<li id="bib.bib2" class="ltx_bibitem">'
        '<span class="ltx_tag ltx_tag_bibitem">[2]</span>'
        '<span class="ltx_bibblock">A. Einstein, Annalen Phys. 49 (1916).</span></li>'
        "</ul>",
        "2608.11111v1",
    )

    assert [r.ordinal for r in refs] == [1, 2]
    assert refs[0].cited_arxiv_id == "1412.6980"
    # The entry's printed marker duplicates the ordinal, so it is not stored.
    assert refs[0].raw_text.startswith("D. Kingma")
    assert refs[1].cited_arxiv_id is None


def test_an_identifier_is_only_taken_when_the_reference_states_one() -> None:
    """A reference resolved by title similarity would be a guess stored as fact."""
    from verbatim.references import extract_arxiv_id, extract_doi

    assert extract_arxiv_id("see arXiv:2401.12345v3 for details") == "2401.12345"
    assert extract_arxiv_id("https://arxiv.org/abs/hep-th/9711200") == "hep-th/9711200"
    assert extract_arxiv_id("Attention Is All You Need, NeurIPS 2017") is None
    assert extract_doi("Phys. Rev. D 12, 345 (1975), doi:10.1103/PhysRevD.12.345.") == (
        "10.1103/PhysRevD.12.345"
    )
