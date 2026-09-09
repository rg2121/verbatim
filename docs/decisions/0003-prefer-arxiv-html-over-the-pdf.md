# 3. Prefer arXiv's HTML rendering over the PDF

**Status:** accepted. 94% of this corpus is read from HTML.

## Context

A PDF is a printed page: mathematics becomes glyph soup, section structure is
inferred, and tables flatten into prose where no reader can tell which values
formed a row. arXiv renders most recent papers from the author's TeX.

## Decision

Where a rendering exists and covers enough of the paper, it is the source. A
paper is chunked from exactly one source, never both. Tables are carried as
their own passages with spans placed in their real columns. Papers arXiv
renders nothing for are recorded as such so they are asked for once.

## Consequences

Maths survives as LaTeX and headings are the author's own, so a citation can
name a real section rather than an invented page number. Measured over 1,547
renderings: 9,343 tables and 84,829 data rows kept as tables. The cost is a
second fetch stage and a completeness check, because a truncated rendering is
worse than the PDF.
