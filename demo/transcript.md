# What it looks like

Real output, captured by `demo/capture.sh` against `sample/corpus.db` - the
forty-paper corpus committed to this repository, so every command below runs
on a clean clone. `demo/demo.tape` records the same sequence as a GIF.

## `uv run verbatim stats --state sample/corpus.db`

Forty papers, their passages and their vectors, already in the repository.

```
  stage      done                         remaining              next
  papers     40                           40 current             
  licences   40 known                     0 to look up           verbatim licenses
  pdfs       0 (0.0 GB)                   40 to fetch            verbatim download
  html       0 (0 have none)              0 to check             verbatim html
  references 2,230                        0 papers to read       verbatim references
  chunks     3,139 (2,863 body / 0 refs)  0 papers to chunk      verbatim chunk
  vectors    3,139                        0 to embed             verbatim embed
```

## `uv run verbatim search 'how can a model be finetuned on one GPU' --state sample/corpus.db --limit 2`

Every passage carries the paper, the version, where in it, and the licence.

```
1. arXiv:2305.14314v1  section 1 Introduction  [chars 3792-4991]  score 0.03
   https://arxiv.org/abs/2305.14314v1  (no licence reported)
   …QLoRA reduces the average memory requirements of finetuning [a] 65B parameter [model] from $>$ 780GB of [GPU]…

2. arXiv:2305.14314v1  section 9 Broader Impacts  [chars 59456-60656]  score 0.03
   https://arxiv.org/abs/2305.14314v1  (no licence reported)
   …We have demonstrated that our best 33B [model] trained [on] the Open Assistant dataset [can] rival…

```

## `uv run verbatim ask 'what is QLoRA?' --state sample/corpus.db`

The point of the project. Every quoted span is checked against the passage it cites; one that fails is rejected and marked [unverified] rather than presented as sourced.

```
QLoRA is a method for "high-fidelity 4-bit finetuning" [1] that utilizes "Double Quantization" and "4-bit NormalFloat (NF4) quantization" [1]. It is designed to "significantly reduce the required memory for finetuning models" [2] and is the first method that would allow for the finetuning of 7B models on phones and other low resource settings [3]. QLoRA typically employs a "4-bit" low-precision storage data type and a "BFloat16" computation data type, meaning that weight tensors are dequantized to BFloat16 to perform matrix multiplication in 16-bit [1]. Additionally, it introduces "Paged Optimizers" to prevent out-of-memory errors caused by memory spikes during gradient checkpointing [1].

References
[1] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314v1, section 3 QLoRA Finetuning. https://arxiv.org/abs/2305.14314v1
[2] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314v1, section 4 QLoRA vs. Standard Finetuning. https://arxiv.org/abs/2305.14314v1
[3] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314v1, section 9 Broader Impacts. https://arxiv.org/abs/2305.14314v1
  Answered from 8 passages in 4 papers.
  Those passages are about: QLoRA and algorithmic search neutrality

  2 model calls, 2,528 tokens in, 585 out, 2.3s in the model
```

## `uv run verbatim graph --state sample/corpus.db --top 3`

The bibliographies form a citation graph the corpus can be asked about.

```
  12 edges between held papers; 11 of 40 papers (28%) touch at least one; average degree 0.60

  Most cited by this corpus, of the papers it holds:
     9 citing papers   arXiv:2302.13971v1  LLaMA: Open and Efficient Foundation Language Models
     1 citing papers   arXiv:2304.08485v2  Visual Instruction Tuning
     1 citing papers   arXiv:2304.10592v2  MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models
```

## `uv run verbatim runs --state sample/corpus.db --limit 5`

And every command records what it did, including the ones that failed.

```
  when                 command              took  what it did
  2026-09-09T18:01:38  ask          ok      3.0s  passages=8, papers=4, citations=3, rejected=0
  2026-09-09T18:01:02  ask          ok      4.1s  passages=8, papers=4, citations=3, rejected=0
  2026-09-09T17:56:55  index        ok      0.2s  added=0, removed=0, vectors=3139, model=BAAI/bge-small-en-v1.5
  2026-09-09T17:56:54  index        ok      0.2s  added=0, removed=0, vectors=3139, model=BAAI/bge-small-en-v1.5
  2026-09-09T17:28:53  ask          ok      3.2s  passages=8, papers=4, citations=1, rejected=1
```

## `VERBATIM_LOG_FORMAT=json uv run verbatim index --state sample/corpus.db`

Results go to stdout for a person; stage events go to stderr for anything that parses. Both are written from the same place as the ledger row, so they cannot disagree.

```
{"at": "2026-09-09T20:01:42", "level": "info", "event": "run.start", "command": "index", "run_id": "5a614992436f", "model": "BAAI/bge-small-en-v1.5", "rebuild": false}
{"at": "2026-09-09T20:01:42", "level": "info", "event": "run.finish", "command": "index", "run_id": "5a614992436f", "ok": true, "seconds": 0.157, "error": null, "added": 0, "removed": 0, "vectors": 3139, "model": "BAAI/bge-small-en-v1.5"}
```
