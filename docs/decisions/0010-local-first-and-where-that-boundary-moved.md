# 10. Local-first, and the boundary is visible

**Status:** accepted; the default now crosses it deliberately.

## Context

The corpus is 26 GB of other people's papers under mixed licences. Sending it
to a third party should not be something a default decides quietly.

## Decision

Ollama is the only backend: no second client, no API key, no hosted path around
it. A hosted-API backend that once existed was removed. But *which model*
Ollama names is a decision with a boundary in it — a local model keeps the
corpus on this machine, while a model whose name ends `-cloud` is served by
Ollama's infrastructure, so the passages in each prompt leave it.

## Consequences

The choice is the user's and is visible in `OLLAMA_MODEL` rather than buried.
The README states where the boundary is instead of implying it cannot be
crossed. Testing must not depend on either: every model call sits behind a
protocol, so the suite runs offline in about two seconds with no Ollama at all.
