"""Drive both arms of the v2 evaluation. The driver is built at M3; this header is
the evaluation, pre-registered at M0 (plan revision 6) and fixed from here on; any
correction made before a test run is dated under AMENDMENTS.

PINNED
v1, its worker (evals/v1_arms.py), the scorer (evals/score_arms.py) and every
set are fixed at commit 415a9c29b8ff1d325e0adb2a366b41abf5bfad34 on main,
checked out detached at ../verbatim-v1. evals/v2/manifest.json names each set
with its SHA-256; the manifest's own SHA-256 is
19fbb04706af797af9b22053714c8995500f04b20ce19a6a13c242e681b2fe2c.

ARMS
  ask, abstention, human  v1: `verbatim ask` with its CLI defaults (hybrid
                          search, citation-graph leg, gated ColBERT; reflection,
                          quotation check and subject check on). v2: `ask`.
  team                    v1: `verbatim team` (reflection on). v2: `team`.
  reports                 v1: `verbatim report --agents`. v2: `report`.
  gaps                    v1's falsifier and v2's `gaps` falsifier, on the same
                          frozen claims.

HOW THE ARMS RUN
- Two long-lived workers, one per checkout, each in its own environment; the v1
  worker runs in ../verbatim-v1. Both only read the store: the main checkout's
  data/corpus.db, given to both as VERBATIM_STATE_PATH.
- Both workers get the same explicit environment, with gemma4:e4b-mlx as the
  model for every role. Before the first case the driver loads each checkout's
  settings exactly as its worker will, and refuses to start if a setting both
  checkouts have differs or if either could call another model. (The main
  checkout's .env sets OLLAMA_MODEL=gemma4:31b-cloud; the pinned one has none.)
- Each case goes to both workers, back to back. In the set's frozen order the
  first case goes to v1 first, the next to v2 first, and so on.
- Every gate runs on gemma4:e4b-mlx. Each run records the model's digest, and a
  case whose two runs differ in digest is rerun for both arms. A later run of
  the same sets on gemma4:31b-cloud is reported, not gated.

RECORDS
One per arm per case: set, case id, arm, order, status; the answer (text;
citations with marker, chunk id and quote; rejected citations; declined); the
passage ids in rank order; subject and closeness; calls, tokens in and out,
seconds; model and digest; error kind. Team, report and gaps records carry the
same per section or per claim, plus the verdict and effort counts. The pinned
scorer scores both arms from these records alone and trusts neither: a citation
is verified only if its quote is, word for word, in the passage the store holds.

ERRORS
Connection errors, HTTP 401/402/429/5xx and a model that is not pulled stop the
driver, and the case is rerun for both arms when it resumes. Any other exception
counts against the arm that raised it: it scores 0 and counts as incomplete.

SETS (evals/v2; first evaluation | sealed second)                      gate
  dev         60 questions, 9 two-paper questions, 10 claims, and 2
              report topics written at M5; the only data used for
              iterating                                                none
  test        test-a 1,040 | test-b 1,040 (both more if sizing says)   P2, P3
  human       drawn from test-a | from test-b, by the seed "human"     P1
  abstention  400 from test-a with the answering paper hidden, plus
              heldout.jsonl's 5 unanswerable cases | 400 from test-b   abstention
  team        evals/breadth/questions.jsonl, 24 | team-b, 24           team
  reports     reports-a, 10 topics | reports-b, 10                     reports
  gaps        gaps-a, 50 claims | gaps-b, 50                           gaps
A hidden paper is hidden in every version. Any filter turns the title and graph
search legs off in both arms, so the abstention cases that hide a paper run
without them and the 5 unanswerable ones, which hide nothing, run with them.

PRIMARY METRICS (ask; all three must pass)                          margin
  P1  correct and supported: your blind score of 2 (of 0, 1, 2)    10 points
  P2  verified grounding: at least one verified citation, and
      every sentence that makes a claim carries one                  5 points
  P3  a verified citation to an expected paper (any version)         5 points
Decision: v2 is non-inferior if the lower bound of the 95% paired-bootstrap
interval of (v2 - v1) is above -margin, taking the lowest bound over 10 seeds of
10,000 resamples. Requiring all three is an intersection-union test, so a wrong
merge stays at or below 2.5% in one evaluation without adjustment, and at or
below about 4.9% over the two PROCEDURE allows. Superiority is claimed only from
Holm-adjusted tests.

SIZING (fixed after the M3 pilot, before any test run)
- Test sets: n = max(1,040, 13.0 * d_hi / margin^2) for test-a and test-b
  alike: 95% power for a v2 equal to v1. d_hi is the larger of P2's and P3's
  upper ends of d + 1.96 * sqrt(d * (1 - d) / 60), where d is the share of the
  60 dev questions on which the arms disagree on that metric.
- Human sample: n = 1,300 * d_P1, at least 120 and at most 300, where d_P1 is
  the share of your 40 blind dev pairs on which the arms disagree on P1.

OTHER GATES (count rules on small samples: they screen for regressions)
  abstention  the decline-or-warn rate is non-inferior with a margin of 8
              points, by the same bootstrap rule. The scorer recomputes it: no
              verified citation, or subject closeness below 0.65
              (SUBJECT_WARN_BELOW). A failed run does not decline.
  team        blind losses <= blind wins + 4
  reports     blind losses <= blind wins + 4; v2's share of verified claim
              sentences within 7 points of v1's; incomplete v2 reports <= failed
              v1 reports + 1; report wall time <= 3x v1's. Accepted knowingly: a
              7-3 blind loss passes on 10 topics.
  gaps        planted refutations v2 misses that v1 found, minus the reverse,
              <= 4. A refutation counts only with a quote the store holds.
  runtime     protocol failures <= 5% of v2 runs; budget_exhausted <= 5%
  cost        median plain-ask calls and median latency each <= 3x v1's, in the
              same session; checked first at the M3 and M5 pilots

BLIND SCORING
The scorer builds one sheet per set with the arms' labels hidden; the mapping is
sealed outside the repository and not opened before scoring. The sheet's
manifest is committed before any score and your labels after it. Unblinding
refuses a sheet with an answer unscored or a score outside 0-2. No sheet is
built and no gate is scored until both arms have records for every case of the
set.

REPORTED, NOT GATED
Holm-adjusted secondaries (failed checks per answer, unbacked sentences, uncited
claim sentences); false refutations with the planted paper hidden; the
Verifier's decisions on failed checks; INSPECT's use and the share followed by
anything other than accept (under 10% makes it a candidate for removal); size
beside the 1,585 lines it replaces.

PROCEDURE
Every gate runs once on its first set. If any fails, work continues on dev
only, then every gate runs once on its sealed second set. If that fails too
(M9), the cleanup still ships with the RulePolicy Lead (v1's rules as Lead
decisions under the same contract) as the default, the agent Lead behind
`--lead agent`, and the README saying which is the default and why.

ACCEPTED RISKS
A v2 equal to v1 passes the first evaluation about 73% of the time and one of
the two about 93%. The second chance works both ways: a v2 truly worse by a full
margin passes one evaluation at most 2.5% of the time, and one of the two at
most about 4.9% (1 - 0.975^2). A v2 truly 2 points worse on P2 passes that gate
about 58% of the time at 1,040 cases: a 5-point loss is the loss accepted.

AMENDMENTS
2026-11-12, before any test run, after a review: the wrong-merge figure now
counts the sealed second evaluation (at most 2.5% for one evaluation, about 4.9%
over both). No rule, margin, set or size changed.
"""

if __name__ == "__main__":
    raise SystemExit("error: the driver is built at M3; this file holds the pre-registration")
