#!/bin/bash
# Run the demo's commands and write demo/transcript.md. This is what the tape
# records; capturing it separately means the transcript is real output rather
# than a transcription, and that a change breaking the demo is visible without
# installing a recorder.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=demo/transcript.md
S="--state sample/corpus.db"

# Results only. The stage events go to stderr precisely so they do not land
# in output like this; one step below shows them on purpose.
step() {
  printf '\n## `%s`\n\n%s\n\n```\n' "$1" "$2" >> $OUT
  eval "$1" >> $OUT 2>/dev/null
  printf '```\n' >> $OUT
}

# The other half: what a log shipper sees while the same command runs.
events() {
  printf '\n## `%s`\n\n%s\n\n```\n' "$1" "$2" >> $OUT
  eval "$1" 2>> $OUT >/dev/null
  printf '```\n' >> $OUT
}

cat > $OUT <<'HEAD'
# What it looks like

Real output, captured by `demo/capture.sh` against `sample/corpus.db` - the
forty-paper corpus committed to this repository, so every command below runs
on a clean clone. `demo/demo.tape` records the same sequence as a GIF.
HEAD
step "uv run verbatim stats $S" "Forty papers, their passages and their vectors, already in the repository."
step "uv run verbatim search 'how can a model be finetuned on one GPU' $S --limit 2" "Every passage carries the paper, the version, where in it, and the licence."
step "uv run verbatim ask 'what is QLoRA?' $S" "The point of the project. Every quoted span is checked against the passage it cites; one that fails is rejected and marked [unverified] rather than presented as sourced."
step "uv run verbatim graph $S --top 3" "The bibliographies form a citation graph the corpus can be asked about."
step "uv run verbatim runs $S --limit 5" "And every command records what it did, including the ones that failed."
events "VERBATIM_LOG_FORMAT=json uv run verbatim index $S" "Results go to stdout for a person; stage events go to stderr for anything that parses. Both are written from the same place as the ledger row, so they cannot disagree."
echo "wrote $OUT"
