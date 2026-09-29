"""Claims of absence ("no paper does X"), tested by an agent trying to refute them.

The Prospector reads what the corpus holds on a topic and proposes specific
gaps. The Falsifier writes its own queries to find one paper that disproves
each. Code decides the verdict:

- REFUTED: a passage was cited as filling the gap and its quote verified.
  Whether it truly fills the gap remains the model's judgement.
- STANDS: the gap survived a search that met the effort floor. That is not
  proof the literature lacks it.
- UNPROVEN: the search was too thin for its failure to mean anything.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re

from verbatim.answers import Answer, Completion, answer_from_matches
from verbatim.citations import Citation
from verbatim.corpus import ChunkMatch, CorpusStore, PaperFilter
from verbatim.embedding import Embedder
from verbatim.retrieval import hybrid_search

# The effort a failed refutation needs before it counts: a floor, not a tuned value.
MIN_QUERIES = 3
MIN_PAPERS = 8

REFUTED = "refuted"
STANDS = "stands"
UNPROVEN = "unproven"

# Unfalsifiable phrasing ("more work is needed"): no paper could ever refute it.
VAGUE = re.compile(
    r"\b(?:more|further|additional|future)\s+(?:work|research|study|studies|"
    r"investigation|exploration)\b"
    r"|\bunder-?explored\b|\bunder-?studied\b|\blittle\s+(?:is\s+)?known\b"
    r"|\bremains?\s+(?:an?\s+)?open\b|\bin\s+general\b",
    re.IGNORECASE,
)
MIN_GAP_WORDS = 6

# Claims pointing at the prompt instead of a subject. Unlike the `distill` check,
# "here" is allowed: "nothing here measures X" is a fine claim.
POINTS_AT_THE_PROMPT = re.compile(
    r"\b(?:these|those|the)\s+(?:passages?|excerpts?|papers|list|above)\b"
    r"|^\s*(?:this|it|that)\b",
    re.IGNORECASE,
)
# Words that carry no subject. A claim made only of these names nothing that
# could be searched for.
EMPTY = {
    "no", "not", "nothing", "none", "paper", "papers", "work", "study",
    "studies", "research", "here", "corpus", "there", "the", "a", "an", "is",
    "are", "was", "were", "be", "been", "on", "in", "of", "for", "to", "and",
    "or", "with", "that", "this", "it", "exist", "exists", "missing", "gap",
    "gaps", "any", "does", "do", "has", "have",
}
MIN_SUBJECT_WORDS = 3

PROSPECTOR_SYSTEM = """You are given passages that represent everything a corpus holds about a topic.

Name things a researcher working on this topic would expect to find here, and which these passages show are missing.

Rules:
- Each gap must be specific enough that a single paper could disprove it.
- Name the subject and the thing not done with it. Write "no paper evaluates
  sparse attention on multilingual retrieval", never "evaluation is
  under-explored".
- Do not write "more work is needed", "under-studied", or "remains an open
  question". Those cannot be disproved and are not gaps.
- Do not refer to the passages, the corpus, or this list.

Write one per line, in this exact form, and nothing else:

GAP: <claim>"""

GAP_LINE = re.compile(r"^\s*GAP:\s*(.+?)\s*$", re.MULTILINE)

FALSIFIER_SYSTEM = """You are given a claim that a body of literature contains nothing on some subject.

Your job is to destroy the claim. Write search queries that would find a paper proving it wrong.

Rules:
- Write queries, not questions. Use the words a paper on the subject would use.
- Vary them. Different vocabulary, different angles, synonyms a different
  research community would choose.
- Do not repeat the claim back as a query.

Write one per line, in this exact form, and nothing else:

QUERY: <query>"""

QUERY_LINE = re.compile(r"^\s*QUERY:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Gap:
    """A claim that the corpus holds nothing on some subject."""

    topic: str
    claim: str

    def as_question(self) -> str:
        """The claim turned into what the Falsifier must find an answer to."""
        return f"what work has been done on {self.claim.rstrip('.')}?"


@dataclass(frozen=True)
class Effort:
    """What the Falsifier actually did, so its failure can be weighed."""

    queries: tuple[str, ...]
    papers: tuple[str, ...]
    passages: int

    @property
    def genuine(self) -> bool:
        """Whether a failure to refute means anything. Arithmetic, not judgement."""
        return len(self.queries) >= MIN_QUERIES and len(self.papers) >= MIN_PAPERS

    def render(self) -> str:
        return (
            f"{len(self.queries)} quer(ies), {len(self.papers)} paper(s), "
            f"{self.passages} passage(s)"
        )


@dataclass(frozen=True)
class Finding:
    """One gap, and what became of it."""

    gap: Gap
    verdict: str
    effort: Effort
    citation: Citation | None = None
    reason: str = ""

    def render(self) -> str:
        head = {
            REFUTED: "REFUTED",
            STANDS: "STANDS",
            UNPROVEN: "unproven",
        }[self.verdict]
        lines = [f"  [{head}] {self.gap.claim}"]
        if self.citation is not None:
            lines.append(
                f"    refuted by {self.citation.arxiv_id}: "
                f"\"{self.citation.quoted_text[:110]}\""
            )
        lines.append(f"    the falsifier tried {self.effort.render()}")
        if self.reason:
            lines.append(f"    {self.reason}")
        return "\n".join(lines)


def checkable(claim: str) -> bool:
    """Whether a single paper could disprove this claim."""
    if len(claim.split()) < MIN_GAP_WORDS:
        return False
    if VAGUE.search(claim):
        return False
    if POINTS_AT_THE_PROMPT.search(claim):
        return False
    # Something has to be left to search for once the scaffolding is removed.
    subject = [w for w in re.findall(r"[a-z0-9-]+", claim.lower()) if w not in EMPTY]
    return len(subject) >= MIN_SUBJECT_WORDS


def propose_gaps(
    corpus: CorpusStore,
    embedder: Embedder,
    complete: Callable[[str, str], str],
    topic: str,
    *,
    passages: int = 12,
    limit: int = 5,
) -> tuple[list[Gap], list[str]]:
    """The Prospector: propose gaps, returning them and the claims rejected as unfalsifiable."""
    matches = hybrid_search(corpus, embedder, topic, limit=passages, max_per_paper=2)
    if not matches:
        return [], []
    held = "\n\n".join(
        f"<passage arxiv_id=\"{m.chunk.arxiv_id}\">{m.chunk.text[:600]}</passage>"
        for m in matches
    )
    reply = complete(PROSPECTOR_SYSTEM, f"TOPIC: {topic}\n\n{held}")
    claims = [m.group(1) for m in GAP_LINE.finditer(reply or "")]
    gaps, rejected = [], []
    for claim in dict.fromkeys(claims):
        if checkable(claim):
            gaps.append(Gap(topic=topic, claim=claim))
        else:
            rejected.append(claim)
    return gaps[:limit], rejected


def falsify(
    corpus: CorpusStore,
    embedder: Embedder,
    completion: Completion,
    complete: Callable[[str, str], str],
    gap: Gap,
    *,
    queries: int = 4,
    per_query: int = 6,
    hidden: Sequence[str] = (),
) -> Finding:
    """The Falsifier: try to refute one gap and record the effort.

    `hidden` papers are filtered out of retrieval, to plant a gap with a known answer.
    """
    reply = complete(FALSIFIER_SYSTEM, f"CLAIM: {gap.claim}")
    searches = [m.group(1) for m in QUERY_LINE.finditer(reply or "")]
    searches = list(dict.fromkeys(searches))[:queries]
    if not searches:
        # A Falsifier that writes no queries has not tried, whatever it says.
        searches = [gap.claim]

    papers = PaperFilter(exclude_ids=tuple(hidden)) if hidden else None
    seen: dict[str, ChunkMatch] = {}
    for query in searches:
        for match in hybrid_search(
            corpus, embedder, query, limit=per_query, max_per_paper=2, papers=papers
        ):
            seen.setdefault(match.chunk.chunk_id, match)
    effort = Effort(
        queries=tuple(searches),
        papers=tuple(dict.fromkeys(m.chunk.arxiv_id for m in seen.values())),
        passages=len(seen),
    )

    answer: Answer | None = None
    if seen:
        answer = answer_from_matches(
            corpus, completion, gap.as_question(), list(seen.values())
        )
    if answer is not None and answer.citations:
        # A refutation is a verified citation or nothing.
        return Finding(
            gap=gap, verdict=REFUTED, effort=effort, citation=answer.citations[0],
            reason=(
                "the falsifier cited this passage as filling the gap; the "
                "quote is verified, its relevance is the model's judgement"
            ),
        )
    if not effort.genuine:
        return Finding(
            gap=gap, verdict=UNPROVEN, effort=effort,
            reason=(
                f"the falsifier did not try hard enough to count: "
                f"{MIN_QUERIES} queries and {MIN_PAPERS} papers are the floor"
            ),
        )
    return Finding(
        gap=gap, verdict=STANDS, effort=effort,
        reason=(
            "survived a bounded adversarial search - which is not the same as "
            "proof of absence"
        ),
    )


def find_gaps(
    corpus: CorpusStore,
    embedder: Embedder,
    completion: Completion,
    complete: Callable[[str, str], str],
    topic: str,
    *,
    limit: int = 5,
    hidden: Sequence[str] = (),
    **options: object,
) -> tuple[list[Finding], list[str]]:
    """Propose gaps, then try to destroy each one."""
    gaps, rejected = propose_gaps(
        corpus, embedder, complete, topic, limit=limit
    )
    findings = [
        falsify(corpus, embedder, completion, complete, gap, hidden=hidden,
                **options)  # type: ignore[arg-type]
        for gap in gaps
    ]
    return findings, rejected


def render(findings: Sequence[Finding], rejected: Sequence[str]) -> str:
    """What survived, what did not, and what was never checkable."""
    stood = [f for f in findings if f.verdict == STANDS]
    refuted = [f for f in findings if f.verdict == REFUTED]
    unproven = [f for f in findings if f.verdict == UNPROVEN]
    lines = [
        f"  {len(findings)} gap(s) proposed, "
        f"{len(refuted)} refuted, {len(stood)} stood, {len(unproven)} unproven",
        "",
        "  A gap that STANDS survived a bounded adversarial search. That is not",
        "  proof the literature lacks it: the falsifier searches this corpus,",
        "  and about one standing gap in twelve was measured false.",
        "",
    ]
    for finding in [*stood, *refuted, *unproven]:
        lines.append(finding.render())
        lines.append("")
    if rejected:
        lines.append(f"  {len(rejected)} claim(s) rejected as unfalsifiable:")
        lines += [f"    {claim[:100]}" for claim in rejected]
    return "\n".join(lines)
