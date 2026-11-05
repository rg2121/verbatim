"""Quotation marks in an answer's own text, checked against the passages it cites.

The quote check proves the words on each citation line, but a model also puts
words in quotation marks inside its answer, and a reader takes those as quoted
too, though nothing checked them. So each quoted phrase is held to the citation
line's rule: one that no cited passage holds loses its quotation marks, since it
is not a quotation, and one found in a different cited passage than its marker
names keeps them and is reported. Measured on saved answers in
evals/quotation_check.py.

A bracket inside a quotation is the quoted text's own, such as a paper's
reference number, so whatever reads markers passes over it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re

# Fewer words in quotation marks are usually a term, not a quotation.
MIN_WORDS = 3
STRAIGHT = '"'
CURLY = re.compile("\u201c[^\u201c\u201d]*\u201d")
MARKER_GROUP = re.compile(r"(?:\[(?:\d+|unverified)\]\s*)+")
NUMBER = re.compile(r"\[(\d+)\]")
SENTENCE_END = re.compile("[.!?](?=\\s+[A-Z\"\u201c(]|\\s*$)")


@dataclass(frozen=True)
class QuotationProblem:
    """Words the answer quotes that are not word for word where it says they are."""

    quote: str
    # The citation the quotation is attributed to, when a marker follows it.
    marker: int | None
    # The citation whose passage does hold the words, when it is another one.
    found_in: int | None

    def describe(self) -> str:
        if self.found_in is None:
            return "not word for word in a cited passage, so shown without quotation marks"
        return f"quoted from [{self.found_in}], not [{self.marker}]"

    def to_dict(self) -> dict[str, object]:
        return {
            "quote": self.quote, "marker": self.marker,
            "found_in": self.found_in, "note": self.describe(),
        }


def check_quotations(text: str, passages: list[str]) -> tuple[str, list[QuotationProblem]]:
    """Check each quoted phrase in `text`, where `passages[n - 1]` is the passage behind [n].

    Returns the text with the quotation marks taken off words that no cited
    passage holds, and what was found. Nothing else in the text changes.
    """
    problems: list[QuotationProblem] = []
    unquoted: list[int] = []
    for opening, closing in quotations(text):
        quote = text[opening + 1:closing]
        named = _markers_after(text, closing + 1)
        holding = [n for n, passage in enumerate(passages, start=1) if word_for_word(quote, passage)]
        if any(n in holding for n in named) or (holding and not named):
            continue
        found_in = holding[0] if holding else None
        problems.append(QuotationProblem(
            quote=" ".join(quote.split()), marker=named[0] if named else None, found_in=found_in,
        ))
        if found_in is None:
            unquoted += [opening, closing]
    for position in sorted(unquoted, reverse=True):
        text = text[:position] + text[position + 1:]
    return text, problems


def word_for_word(quote: str, passage: str) -> bool:
    """Whether the passage holds the quote, as the citation check reads it, or with its
    first letter's case changed to fit the answer's sentence."""
    from verbatim.answers import verify_quote  # answers imports this module

    if verify_quote(quote, passage) is None:
        return True
    first = next((i for i, character in enumerate(quote) if character.isalpha()), None)
    if first is None:
        return False
    swapped = quote[:first] + quote[first].swapcase() + quote[first + 1:]
    return verify_quote(swapped, passage) is None


def quotations(text: str) -> list[tuple[int, int]]:
    """Each quotation of MIN_WORDS words or more, as its opening and closing mark."""
    return [
        (opening, closing) for opening, closing in quotation_spans(text)
        if len(text[opening + 1:closing].split()) >= MIN_WORDS
    ]


def quoted_from(text: str, passages: Sequence[str]) -> list[tuple[int, int]]:
    """The quotations one of the passages holds word for word, brackets and all."""
    return [
        (opening, closing) for opening, closing in quotations(text)
        if any(word_for_word(text[opening + 1:closing], passage) for passage in passages)
    ]


def mask_quoted_brackets(text: str) -> str:
    """The text with the brackets inside its quotations written as braces.

    For reading markers only: the length is unchanged, so offsets and sentences
    are the text's own. Checked answers keep only quotations a cited passage holds.
    """
    characters = list(text)
    for opening, closing in quotations(text):
        for position in range(opening + 1, closing):
            if characters[position] in "[]":
                characters[position] = "{" if characters[position] == "[" else "}"
    return "".join(characters)


def sub_outside(
    pattern: re.Pattern[str],
    replace: Callable[[re.Match[str]], str] | str,
    text: str,
    spans: Sequence[tuple[int, int]],
) -> str:
    """`pattern.sub(replace, text)`, leaving a match that starts inside a span as it is."""
    def rewrite(match: re.Match[str]) -> str:
        if any(opening < match.start() < closing for opening, closing in spans):
            return match.group(0)
        return replace(match) if callable(replace) else match.expand(replace)

    return pattern.sub(rewrite, text)


def quotation_spans(text: str) -> list[tuple[int, int]]:
    """Each quotation's opening and closing mark, the outermost only.

    Straight marks pair in order, and only when there is an even number of them:
    an odd count means one is not a quotation mark, and a guess could take the
    marks off the wrong words. Curly marks pair by direction. A quotation inside
    another, as when an answer quotes a passage that quotes, is part of it.
    """
    straight = [i for i, character in enumerate(text) if character == STRAIGHT]
    pairs = list(zip(straight[0::2], straight[1::2])) if len(straight) % 2 == 0 else []
    pairs += [(match.start(), match.end() - 1) for match in CURLY.finditer(text)]
    outermost: list[tuple[int, int]] = []
    for opening, closing in sorted(pairs):
        if outermost and opening < outermost[-1][1]:
            continue
        outermost.append((opening, closing))
    return outermost


def _markers_after(text: str, start: int) -> list[int]:
    """The citations named by the first markers after a quotation, within its sentence."""
    rest = text[start:]
    group = MARKER_GROUP.search(rest)
    if group is None:
        return []
    end = SENTENCE_END.search(rest)
    if end is not None and group.start() > end.end() and rest[end.end():group.start()].strip():
        return []  # the markers belong to a later sentence
    return [int(number) for number in NUMBER.findall(group.group(0))]
