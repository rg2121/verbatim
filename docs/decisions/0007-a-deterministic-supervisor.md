# 7. The multi-agent supervisor is code, not a model

**Status:** accepted; `report --agents` is off by default.

## Context

The 2026 literature agrees on orchestrator-workers for breadth-first research
work, at 3–15× the tokens. It also documents how these systems fail: weak
specification, misaligned roles, missing verification, blind retries that
reproduce latent faults, and supervisors that corrupt worker output by
paraphrasing it.

## Decision

Roles differ in kind, and the supervisor is deterministic. A planner (model)
writes section questions; a router (code) picks a ladder of strategies by
question shape; retrievers run those strategies, with the tool-choosing agent
as the last rung; a writer (model) uses the existing verified-citation path; a
verifier (code) accepts a section only if a citation survived checking and the
passages are on-subject; the supervisor keeps a ledger, retries with the *next*
strategy rather than the same one, and forwards each answer untouched.

Work is divided by context, not by problem type. Splitting one question into
sub-questions lost on its own compound cases; a review's sections share a topic
and nothing else, which is the split parallel workers are for.

## Consequences

No critique agent that judges evidence, for the reason in
[0004](0004-report-evidence-rather-than-judging-it.md). Measured against the
deterministic baseline the orchestrator buys a ledger and little else on a
well-covered topic; on an unevenly covered one it climbs all three rungs and
declares a gap rather than answering the wrong question. Parallel workers buy
nothing against a single local model server, which is stated rather than
implied.
