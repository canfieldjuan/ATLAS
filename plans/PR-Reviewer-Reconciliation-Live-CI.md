# PR-Reviewer-Reconciliation-Live-CI

## Why this slice exists

Phase 2 of the review-workflow redesign (#1328) shipped
`scripts/audit_ai_reconciliation.py`: a local gate that the PR body contains a
well-formed "every AI finding fixed-or-waived" reconciliation block before LGTM.
By design it only validates the *recorded text*. It never compares that block
against the *actual open* Codex/Copilot findings on the PR. So a reviewer can
write a clean reconciliation while real bot findings sit open, and nothing
catches it. That is the loophole keeping operating-model gap (b) from being
fully closed (named in the Deferred section of
`plans/archive/PR-Reviewer-Reconciliation-Audit.md` as "CI-side reconciliation").

This slice adds the live, CI-side check: fetch the real bot review threads,
keep the ones still open, and fail when the recorded reconciliation omits any.
Phase 2 validates the paperwork; this validates reality.

Diff-budget overage (~575 LOC, over the 400 soft cap): the check script, its
failure-proving fixtures, the required CI workflow, and the doc/cross-ref edits
are one cohesive unit. Splitting the workflow from the script would leave
unwired dead code, and splitting the tests from the script would violate R2.
The script (~250) and its tests (~130) are the irreducible core.

## Scope (this PR)

Ownership lane: review-workflow/reconciliation-live-ci
Slice phase: Workflow/process

1. A script that fetches live Codex/Copilot review threads for a PR, filters to
   unresolved findings, parses the PR body reconciliation block, and fails when
   an open finding is unaccounted for.
2. A GitHub Actions workflow that runs it on PR review/comment events with a
   read-only token, as a required status check on the PR's final state.
3. Failure-proving fixtures (`AGENTS.md` 3h) using mocked API payloads, so the
   failure branch is tested without hitting live GitHub.
4. Doc updates closing the Phase-2 "recorded-body only" caveat in
   `AGENTS.md` 4a.1 and the "later slice" note in `docs/REVIEWER_RULES.md`.

### Review Contract
- Acceptance criteria: open bot thread + an all-clear or absent reconciliation
  record -> non-zero exit naming the thread(s); no open bot threads -> exit 0;
  body honestly acknowledges open findings -> exit 0 (the local audit owns
  blocking that); resolved/outdated threads ignored; the new test runs in CI.
- Affected surfaces: CI / config / observability. No app code, no DB, read-only
  GitHub token only.
- Risk areas: transient GitHub API failure, race with async bot comments, token
  scope/secret handling.
- Reviewer rules triggered: R2 (failure-branch fixtures), R3 (token handling),
  R6 (API-error handling, no swallowed errors), R10 (maintainability), R12 (CI
  actually runs the new test).

### Files touched
- `scripts/check_ai_reconciliation_live.py`
- `tests/test_check_ai_reconciliation_live.py`
- `.github/workflows/ai_reconciliation_live.yml`
- `.github/workflows/pre_push_audit.yml`
- `AGENTS.md`
- `scripts/audit_ai_reconciliation.py`
- `plans/PR-Reviewer-Reconciliation-Live-CI.md`

## Mechanism

Phase-2's `audit_ai_reconciliation.py` turned out to be section-level (resolved
/ unresolved markers), not per-finding keys, so the enforcement model is the
contradiction check, not fuzzy per-finding matching: fail when the body claims
all-clear (or has no record) while open bot threads exist. "All threads must be
GitHub-resolved" was rejected as self-resolvable by the PR author (a gameable
rigor gate).

`check_ai_reconciliation_live.py --pr <n>`:
1. Fetch the PR review threads via `gh api graphql` (`reviewThreads` with
   isResolved / isOutdated / path / line / author); filter to unresolved,
   non-outdated threads authored by a configured bot (default `copilot`/`codex`,
   substring match on login).
2. Classify the PR body's reconciliation record by IMPORTING the Phase-2 module
   (`extract_section` + `RESOLVED_RE` / `UNRESOLVED_RE`): claims_clear /
   acknowledges_open / absent / unmarked. Reusing the module means local and
   live cannot disagree on what a "resolved" record is.
3. Fail (exit 1) when open bot threads exist AND the body is claims_clear (the
   lie) or absent (no record), listing each open thread (path:line, author,
   snippet). Pass (exit 0) when there are no open bot threads, or the body
   honestly acknowledges open findings (the local audit owns blocking that).
   Exit 2 on usage error or a GitHub API failure (retryable, never a silent
   pass).

The workflow triggers on `pull_request` (opened / synchronize / reopened /
ready_for_review) plus `pull_request_review` and `pull_request_review_comment`,
uses the repo `GITHUB_TOKEN` (pull-requests: read). The `pull_request` trigger is
required so that, as a required status check, it always runs and reports (a quiet
PR with no bot threads reports green instead of wedging the merge), and re-runs
when a bot comment lands after open.

## Intentional
- Live counterpart to Phase 2's local audit, not a replacement. Local still
  validates body shape early in the pre-push bundle; this validates against
  reality once the PR and bot comments exist.
- Fail posture (operator-approved): no bot findings present is a pass, not a
  fail. A genuinely omitted open finding hard-fails (exit 1). A transient GitHub
  API error exits 2 and is surfaced as retryable, so GitHub flakiness never
  silently passes and never permanently walls a merge.
- Required-check on final state (operator-approved): runs on review-comment /
  review events and is evaluated on the PR's final state, not one-shot at open,
  so a bot comment that lands after the PR opens is still enforced. This is a
  required gate by design; it blocks merge when an open finding is unreconciled.
- Bot findings stay advisory (gap b): never auto-resolves or auto-applies, only
  enforces that a human accounted for each finding. No "auto-address all
  comments" loop.
- Reuse Phase-2's section parser (imported) so the local and live checks cannot
  disagree on what a resolved reconciliation record looks like.
- Bot author set is config, not hardcoded, so a new review bot can be added
  without code changes.

## Deferred
- Trend / secondary reviewer metrics (turnaround, rework cycles) from issue 2b.
  Separate slice.
- Scheduling `summarize_review_misses.py --fail-on-ungated` in CI once the ledger
  has real entries. Its own tiny slice.
- Auto-requesting or re-running the bots. Out of scope.

Parked hardening: none.

## Verification
- `tests/test_check_ai_reconciliation_live.py` with mocked API payloads:
  omitted-open-finding -> exit 1 (failure branch proven); all-reconciled -> 0;
  no-findings -> 0; waived-with-reason -> 0; resolved-thread-ignored -> 0;
  API-error -> 2.
- Read-only dry-run against a real PR showing correct pass/flag.
- `scripts/check_ascii_python.sh` and `scripts/local_pr_review.sh` green.

## Estimated diff size
| Area | Est LOC |
|---|---:|
| Plan | ~145 |
| Script | ~250 |
| Tests + fixtures | ~130 |
| Workflow + CI wiring | ~40 |
| Doc edits | ~15 |
| Total | ~580 |

Over the 400 soft cap; overage justified in "Why this slice exists" (cohesive
unit, splitting leaves dead code or violates R2).
