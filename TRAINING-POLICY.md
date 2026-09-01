# Training data policy

**Status: IN FORCE — signed off by Rishab on 2026-09-01.**
**Decisions taken: weights and dataset stay private (not distributed);
CC-BY-SA text is eligible while they stay private and drops out of the
training set if that ever changes. Changing either decision requires a
revision to this document before any further training.**

This project's standing rule is that no model is trained on the corpus until
a written licence and provenance policy exists. This is that document. It
exists because arXiv's default submission licence grants *arXiv* distribution
rights and grants this project nothing, so "the papers are on arXiv" licenses
nothing at all; only an explicitly permissive subset is usable, and even that
subset carries conditions. The open decisions below are recorded as they were
drafted; the status line above records how each was decided.

## What may be trained on

Eligibility is decided by the `license_class` column, which is populated from
arXiv's OAI-PMH records (the query API does not return licences at all).

| class | in the corpus | eligible | why |
|---|---|---|---|
| `open` (CC-BY, CC-BY-SA, CC0) | 3,824 papers | **yes** | these licences permit reuse and derivative works |
| `arxiv-nonexclusive` | most of the corpus | **never** | grants arXiv distribution rights, not us |
| `no-derivatives` | some | **never** | a trained model is arguably a derivative |
| `unknown` | large | **never** | unknown means unknown, not permitted |

Within `open`, CC0 and CC-BY are unconditionally eligible. **CC-BY-SA is open
decision 1**: share-alike obligations applied to model weights are legally
unsettled — including CC-BY-SA papers may oblige any released weights to
carry a share-alike licence. The conservative default written into this draft
is: **CC-BY-SA text is eligible only while decision 2 keeps the weights
private, and drops out of the training set automatically if the weights are
ever to be released.**

## What the training is

Distillation, not raw-text pretraining: the training examples are generated
(question, retrieved passages, verified-citation answer) triples, where every
passage comes from an eligible paper and every quoted span is verified
against its source before the example is written. The model being taught is
the answering behaviour — the flat ANSWER/CITE protocol, the abstention
wording, the citation discipline — not the papers' prose for its own sake.
Raw-text pretraining on the corpus is out of scope for this policy and would
need a revision to add.

## Provenance, per example

Every training example carries a record sufficient to reconstruct or delete
it:

```json
{
  "example_id": "…",
  "question": "…",
  "chunk_ids": ["…"],
  "papers": [
    {"arxiv_id": "2302.13971v1", "license_url": "…", "license_class": "open"}
  ],
  "source_sha256s": ["…"],
  "generated_at": "…",
  "generator_model": "…"
}
```

The records live beside the dataset as JSONL. If a paper's licence turns out
to have been misrecorded, its examples are identifiable and removable, and
the model is retrained without them.

## Attribution

Any released artifact — dataset or weights — ships an `ATTRIBUTIONS` file
listing every source paper: authors, title, versioned arXiv id, licence URL.
This satisfies CC-BY's attribution condition for the dataset with certainty
and for the weights by best effort, since what attribution means for weights
is unsettled; shipping the file costs nothing and omitting it risks
everything.

## Distribution

**Open decision 2: are the weights ever distributed?**

- **Private (recommended default):** the model is trained and used on this
  machine only. Legal exposure is minimal; CC-BY-SA text may be included; the
  portfolio value is the write-up and the numbers, not the artifact.
- **Released:** weights published (e.g. with the GitHub repo). Then CC-BY-SA
  papers are excluded from training, the `ATTRIBUTIONS` file is mandatory,
  and the release notes state the training data's licence composition.

The generated *dataset* is treated the same way as the weights: private by
default, and releasable only under the same conditions, since it quotes the
papers directly and is more clearly a derivative than the weights are.

## Revisions

This policy is versioned in git. Changes take effect only with Rishab's
explicit sign-off, recorded by replacing the DRAFT line at the top with the
sign-off date and the decisions taken.
