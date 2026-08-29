"""Reading a paper from arXiv's HTML rather than its PDF.

arXiv renders every TeX submission to HTML, and that rendering is made from the
author's source rather than from a printed page. Measured on this corpus, 96%
of papers have one. It is better input than a PDF in three ways:

- **Maths survives.** Every <math> element carries the original LaTeX in its
  `alttext`, so an equation stays an equation instead of becoming the character
  soup a PDF extractor produces.
- **Structure is real.** Headings come from the author's sectioning commands,
  not from guessing at font sizes.
- **No extraction damage.** None of the ligatures, lone surrogates or
  reading-order problems that PDF text carries.

What it loses is page numbers: HTML has none, so an HTML-sourced passage is
located by section instead. `Chunk.page_number` therefore means "page" for a
PDF and "section ordinal" for HTML, which is why chunks record which they came
from.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import re

from verbatim.chunking import Section, clean_markdown_text

HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Chrome and navigation arXiv wraps around the paper.
SKIP_CONTENT = {"script", "style", "nav", "header", "footer", "button", "select",
                "option", "label", "form", "dialog", "template", "noscript"}
# arXiv injects a "Report GitHub Issue" dialog whose prose reads like content.
# Matched on class or id, since it carries no distinguishing tag.
SKIP_MARKED = ("modal", "ltx_page_navbar", "ltx_align_center ltx_flex")
# Long enough to be prose, short enough to exclude a whole swallowed document.
MAX_HEADING_LENGTH = 120
WHITESPACE = re.compile(r"\s+")


@dataclass
class _Block:
    level: int
    title: str


class ArxivHTMLParser(HTMLParser):
    """Collect headings and text, keeping maths as the LaTeX it came from."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[Section] = []
        self._stack: list[_Block] = []
        self._body: list[str] = []
        self._heading: list[str] | None = None
        self._heading_level = 0
        self._skip_depth = 0
        self._skip_tags: list[str] = []
        self._math_depth = 0
        self._offset = 0

    # -- structure -------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        marker = f"{attributes.get('class') or ''} {attributes.get('id') or ''}"
        if tag in SKIP_CONTENT or any(m in marker for m in SKIP_MARKED):
            # Track the tag so the matching end tag closes the right skip.
            self._skip_tags.append(tag)
            self._skip_depth += 1
            return
        if tag == "math":
            # Take the LaTeX and ignore the rendered glyphs inside, which would
            # otherwise arrive as unreadable fragments of the same equation.
            latex = (attributes.get("alttext") or "").strip()
            if latex:
                self._emit(f"${latex}$")
            self._math_depth += 1
            return
        if self._math_depth:
            return
        if tag in HEADINGS:
            self._flush()
            self._heading = []
            self._heading_level = int(tag[1])

    def handle_endtag(self, tag: str) -> None:
        if self._skip_tags and self._skip_tags[-1] == tag:
            self._skip_tags.pop()
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "math":
            self._math_depth = max(0, self._math_depth - 1)
            return
        if tag in HEADINGS and self._heading is not None:
            title = WHITESPACE.sub(" ", "".join(self._heading)).strip()[:MAX_HEADING_LENGTH]
            self._heading = None
            if title:
                while self._stack and self._stack[-1].level >= self._heading_level:
                    self._stack.pop()
                self._stack.append(_Block(self._heading_level, title))

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._math_depth:
            return
        if self._heading is not None:
            self._heading.append(data)
        else:
            self._emit(data)

    # -- text ------------------------------------------------------------
    def _emit(self, text: str) -> None:
        if self._heading is not None:
            self._heading.append(text)
        elif not self._skip_depth:
            self._body.append(text)

    def _flush(self) -> None:
        text = clean_markdown_text(" ".join(self._body))
        self._body = []
        if not text:
            return
        self.sections.append(
            Section(
                path=tuple(block.title for block in self._stack),
                text=text,
                char_start=self._offset,
            )
        )
        self._offset += len(text) + 1

    def close(self) -> None:  # noqa: D102 - inherited
        super().close()
        self._flush()


def parse_html_sections(html: str) -> list[Section]:
    """Turn one arXiv HTML page into the sections the chunker already expects."""
    parser = ArxivHTMLParser()
    parser.feed(html)
    parser.close()
    return parser.sections
