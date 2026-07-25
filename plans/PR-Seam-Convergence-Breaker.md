# PR-Seam-Convergence-Breaker

## Why this slice exists

`AGENTS.md` 3k.2 ("Convergence circuit-breaker") already defines the failure: when
each push closes the reported review threads but the next push opens a comparable
count of same-class findings on the same file, the builder is instance-patching a
shared decision rather than fixing it. The rule is mandatory, names the required
response (a Decision-Seam Analysis), and explicitly bans "another token, regex,
vocabulary row, or oracle fixture" as the next push.

The rule is correct. It has never fired on its own, because it lives in a 73 KB
document and needs a human to notice and count rounds.

Measured on ATLAS #2181 (`feat: advisory warning layer on editorial audits`,
+2,569/-94, 22 commits): 94 bot findings across 20 pushes over 5.6 hours, dead flat
(push 1 produced 5 findings, push 20 produced 5), every round on
`atlas_brain/services/content_factory_copy_verification.py`. Round 1 was "exclude
product names from answer-claim detection"; round 20 was "limit product-term
negation to the governing predicate" -- the same seam at finer grain, because an
open category (does this sentence make an unqualified claim?) was implemented with
an enumerable mechanism. Applying 3k.2's own trip condition to that PR's real data
fires at push 3, with 17 pushes and 76 of the 94 findings still to come.

Size is not the predictor: #2174 (+1,098) and #2175 (+1,064) drew zero review
rounds, and #2133 (+5,641, the largest diff sampled) landed in 2 commits.

### Problem-derived contract

**Root cause:** an existing, correct, mandatory rule has no automatic trigger. The
signal it depends on (findings per push, flat over three pushes, concentrated on one
file) is mechanically computable from the GitHub API, but nothing computes it, so
enforcement depends on a human counting rounds during a loop that pushes every ~13
minutes. This slice fixes the missing trigger, not the rule.

**What a correct fix must touch and change:** one detector that reconstructs
per-push finding counts and the dominant file from live PR data; one CI trigger that
runs it on the same cadence the rule measures (pushes); a surfacing mechanism that
reaches the builder without a human relaying it; and both-direction tests proving it
fires on non-convergence and stays silent on convergence, on breadth, and on size.

**What must not change:** the rule text in `AGENTS.md` and
`docs/GUARD_CLASS_CLOSURE.md` (3k.2 already says the right thing); any existing
workflow or detector; the repository's CI permission posture, which grants no
workflow `pull-requests: write` and posts no comments; and the merge gate -- this
must never block, because capping or blocking counts the symptom and, on surfaces
where the findings are real, would ship defects.

## Scope (this PR)

Ownership lane: process-guardrails
Slice phase: workflow/process

1. Add an advisory detector that buckets bot review findings into the push that
   preceded them and trips on a non-converging, single-seam window.
2. Add the CI trigger that runs it per push, mirroring the 3k.1 sibling.
3. Add stdlib-only tests covering both sides of the trip boundary, including a
   replay of the observed ATLAS #2181 shape.
4. Add this plan doc.

Max files: 4

### Files touched

- `.github/workflows/seam_convergence.yml`
- `plans/PR-Seam-Convergence-Breaker.md`
- `scripts/check_seam_convergence.py`
- `tests/test_check_seam_convergence.py`

### Review Contract

Acceptance criteria, checked one by one:

1. The detector trips on ATLAS #2181 at push 3 and names
   `atlas_brain/services/content_factory_copy_verification.py`.
2. It does not trip on #2174, #2175, or #2133 -- the last proving it keys on
   non-convergence rather than diff size.
3. It trips on #2158 and #2161, the two other known spirals.
4. It exits 0 on a trip. `--strict` exists but is not wired into CI.
5. A Decision-Seam Analysis in the PR body suppresses the trip.
6. A push with zero findings breaks the streak; two pushes never trip; findings
   scattered across files never trip.
7. The workflow requests only `contents: read` and `pull-requests: read`, uses
   `pull_request` rather than `pull_request_target`, and therefore needs no entry in
   `ALLOWED_PULL_REQUEST_TARGET_JOBS`.
8. No existing file is modified.

**Reachability proof:** the real entrypoint is the `seam-convergence` job on
`pull_request`. Observable result: the job runs on this PR, reports success, and
emits no annotation (this PR is not a classifier). The detector's own behavior is
proven against live PR history by the replay commands in Verification, whose output
is the annotation text and per-push table.

Affected surfaces: CI only. No runtime, API, database, billing, delivery, or
user-visible surface. No dependency added -- the detector is stdlib plus the `gh`
CLI already used by `scripts/check_ai_reconciliation_live.py`.

Risk areas: false positives on a legitimately dense review of one file (mitigated by
requiring three consecutive pushes, a non-converging count, and single-file
dominance, and by never blocking); GraphQL pagination on very large PRs (capped at
50 pages, matching the sibling); timestamp comparison (ISO-8601 UTC from the API is
lexicographically ordered, so string comparison is exact and avoids a parsing
dependency).

- Reviewer rules triggered: R1, R2, R10, R13. R10 and R2 come from the evaluator /
  gate-predicate shape of the detector; R13 from its classifier shape (the boundary
  must be probed from both sides); R1 from the repo-wide CI surface.

## Mechanism

`bot_findings` filters review threads to bot-authored ones and returns
`(created_at, path)` pairs. Unlike the reconciliation check it deliberately counts
resolved and outdated threads too: a thread the builder already closed is exactly
the instance-patch 3k.2 looks for, so excluding it would hide the pattern.

`assign_findings_to_pushes` buckets each finding into the latest commit at or before
its creation time. Pushes are the unit because 3k.2 counts "consecutive pushes";
wall-clock gaps are not deterministic. A finding predating every commit belongs to no
push and is dropped.

`find_trip` slides a 3-push window. It trips when every push in the window has at
least one finding, the last push still carries at least half the first push's count
(not trending to zero), and one file accounts for more than half the window's
findings. A zero-finding push breaks the streak, because the bot finding nothing is
convergence.

`evaluate` returns the trip plus a per-push table, and suppresses the trip when the
PR body already carries a Decision-Seam Analysis -- the builder has then done what
3k.2 asks. The body is read from `ATLAS_CURRENT_PR_BODY_FILE` when CI provides it,
matching how `scripts/check_guard_class_closure.py` reads its waiver marker.

On a trip the detector prints a `::warning file=<seam>::` annotation naming the seam
and restating 3k.2's required next action, then returns 0.

## Intentional

- **Advisory, exits 0.** Not a round cap. The operator rejected capping (2026-07-25):
  it counts the symptom, and the preserved 2026-07-10 nuance is that on
  classifier/PII/gate surfaces the findings are real, so capping ships defects. This
  changes the permitted class of fix, not whether the PR may proceed.
- **`pull_request`, not `pull_request_target`.** The measured unit is the push, so the
  natural trigger is also the safe one. It fires one push after the bot comment, which
  on #2181 still means push 3 of 20.
- **Annotations, not a PR comment.** No workflow in this repository holds
  `pull-requests: write` and none posts comments. Matching
  `scripts/check_guard_class_closure.py` keeps that posture intact and still surfaces
  the warning inline on the Files tab.
- **Resolved threads are counted.** Counting only unresolved threads would erase the
  evidence, since the builder resolves each round before pushing again.
- **`--strict` exists but is unused.** Same shape the 3k.1 sibling shipped with, so a
  later promotion is a workflow edit rather than a detector rewrite.
- **String timestamp comparison.** ISO-8601 UTC from the GitHub API sorts
  lexicographically; parsing would add a dependency and no correctness.

## Deferred

- **`--strict` promotion to a failing or required check**, after an advisory-proving
  period. Unlocked by observing it on real PRs without a false trip. Trusted-base
  execution is a named precondition, exactly as `guard_class_closure.yml` records for
  its own promotion.
- **The 3k.3 plan-time gate** in `plan_admission.yml`: ask "can I enumerate the
  complete set of inputs that should trip this?" before code exists. Higher leverage
  than this breaker -- this catches the loop, that prevents it. Separate slice.
- **Port to the EOM repositories.** Codex reviews `eom-timetracker` and the website
  too, but neither carries 3k.2; the rule has to land in their `AUDIT_PROTOCOL.md`
  first.
- Parked hardening: none.

## Verification

Commands run locally, with results:

1. Detector unit tests -- `python -m pytest tests/test_check_seam_convergence.py -q
   --noconftest` -- **27 passed**.
2. Live replay, must trip -- the detector run against PR 2181 in this repository
   reported: tripped at push 3, findings per push 5, 9, 4, seam
   `atlas_brain/services/content_factory_copy_verification.py`, exit 0, with the full
   20-push table printed.
3. Live replay, must stay silent -- PRs 2174, 2175 and 2133 each reported "OK: no
   window of 3 consecutive pushes with same-seam findings that are not trending to
   zero", exit 0. 2133 is the largest diff sampled, proving the detector does not key
   on size.
4. Second-side probe on the other known spirals -- PR 2158 tripped at push 3 on
   the live EOM customer-import script (4, 2, 4); PR 2161 tripped at push 3 on
   the EOM portal customer-sync script (4, 11, 3).
5. ASCII gate -- the repository ASCII check for Python files passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/seam_convergence.yml` | 72 |
| `plans/PR-Seam-Convergence-Breaker.md` | 202 |
| `scripts/check_seam_convergence.py` | 373 |
| `tests/test_check_seam_convergence.py` | 230 |
| **Total** | **877** |

Over the 400 LOC soft cap and carrying a diff-budget override in the PR body: the
detector is ~350 lines and the remainder is the both-direction test suite the R13
boundary bar requires, plus this plan. Splitting the tests from the detector they
prove would produce a PR that cannot be reviewed on its own.
