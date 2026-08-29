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
