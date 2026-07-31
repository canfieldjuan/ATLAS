# PR-Live-Reconciliation-Open-Threads-Only

## Why this slice exists

The operator reported that CI/CD is not converging because
`live-reconciliation` can require a final Codex connector approval-style signal
that sometimes never appears. The intended gate is simpler: unresolved scoped
Codex findings make the check red; a quiet head should pass after a short window
that gives the connector a chance to react.

Diff budget overage is intentional: the reviewer-found root cause spans the
required workflow context, checker CLI, checker tests, and watcher contract
fixture. Splitting those would leave one of the red checks unfixed.

### Problem-derived contract

- Root cause: `scripts/check_ai_reconciliation_live.py` mixes open-feedback
  enforcement with a current-head clean-review attestation requirement.
- Correct fix must touch/change: remove the missing-attestation failure, keep
  current-head `CHANGES_REQUESTED` and unresolved scoped bot threads blocking,
  keep the watcher-owned docs-only success signal, keep fresh heads from going
  green before the review window closes, and keep watcher/status-producer
  invocations nonblocking by making the grace wait an explicit required-CI
  opt-in.
- Must not change: branch protection, PR opening behavior, bot identities,
  watcher merge authority, PR body reconciliation parsing, product surfaces, or
  active PR #2251.

## Scope (this PR)

Ownership lane: workflow/live-reconciliation-open-threads-only
Slice phase: Workflow/process

1. Change the quiet-path predicate from "current-head Codex attestation exists"
   to "no open scoped Codex threads, no current-head `CHANGES_REQUESTED`, and no
   active fresh-head review window."
2. In required live CI mode, opt into waiting for the fresh-head window once
   and refetch before returning a required-context result, so the check stays
   pending instead of permanently red.
3. Validate review-window inputs inside controlled error handling.
4. Update focused checker and watcher contract tests for the changed policy.

### Review Contract

- Acceptance criteria:
  1. Clear scoped bot threads pass without a current-head Codex
     review/clean-comment attestation after the review window.
  2. Current-head Codex `CHANGES_REQUESTED` still fails.
  3. Unresolved scoped Codex review threads still fail.
  4. A quiet fresh head waits/refetches in live mode instead of leaving the
     required context red indefinitely when the required workflow opts into
     `--wait-for-review-window`.
  5. Docs-only PRs with proven Markdown-only diffs still emit the watcher-owned
     docs-only success text.
  6. Malformed `ATLAS_CODEX_REVIEW_GRACE_SECONDS` or `--pr-updated-at` exits `2`
     with a diagnostic, not an argparse traceback.
  7. Default live checker invocations return immediately inside the fresh-head
     window, so watcher/status producers do not block on the grace sleep.
- Reachability proof: focused pytest covers the CLI/evaluator entrypoint with
  injected thread, review, comment, body, head-SHA, and timing data.
- Affected surfaces: `.github/workflows/ai_reconciliation_live.yml`,
  `scripts/check_ai_reconciliation_live.py`,
  `tests/test_check_ai_reconciliation_live.py`, `tests/test_pr_watcher.py`, and
  this plan.
- Risk areas: weakening open-thread failures, weakening current-head
  changes-requested failures, stale docs-only watcher readiness, red required
  contexts after the grace window, and malformed config handling.
- Reviewer rules triggered: R1, R2, R6, R8, R10, R11, R12, R13, R14.

### Boundary-change enumeration

- Boundary path/seam: `scripts/check_ai_reconciliation_live.py::evaluate` and
  live-mode `main()` fetch/wait/refetch orchestration.
- Replaced-path behaviors: missing-attestation failure becomes quiet success
  after the review window; required live mode waits/refetches once during the
  window only when `--wait-for-review-window` is present.
- Guard-relevant fields: review-thread resolution/author/path/line/body, review
  author/commit/state, PR body, head SHA, PR `updatedAt`, and review-window
  seconds.
- Caller x input shape: CI invokes `main()` with live GitHub data; tests invoke
  `evaluate()` and `main()` with injected fixtures.

### Deployed-config probing

- Deployed/default config values: default bots remain
  `chatgpt-codex-connector,chatgpt-codex-connector[bot]`;
  default review window remains 300 seconds.
- Explicit value probe: tests pass review states, head SHA, `--pr-updated-at`,
  and malformed review-window config.
- Absent value probe: tests cover no current-head attestation with closed threads
  inside and after the fresh-head window.
- Default-session/default-context probe: N/A - no session-state default change.
- Side-effect ordering: opt-in required live mode waits/refetches before
  docs-only file proof and final evaluation; default live mode evaluates
  immediately.

### Files touched

- `.github/workflows/ai_reconciliation_live.yml`
- `plans/PR-Live-Reconciliation-Open-Threads-Only.md`
- `scripts/check_ai_reconciliation_live.py`
- `tests/test_check_ai_reconciliation_live.py`
- `tests/test_pr_watcher.py`

## Mechanism

The evaluator still checks unresolved scoped bot threads first. It still fails
for current-head Codex `CHANGES_REQUESTED`. If there are no open threads and no
current-head Codex activity yet, the fresh-head window blocks only until the
configured duration expires.

The CLI validates timing inputs after parsing, inside the script's controlled
error path. In live GitHub mode, the default remains nonblocking: a quiet
fresh-head race returns the immediate pending/failure result without sleeping,
which keeps watcher/status-producer invocations fast. The required workflow
passes `--wait-for-review-window`, sleeps until the window closes, refetches
threads/reviews/body, and then evaluates the refreshed state. The workflow
timeout is raised to 10 minutes so the default 300-second wait can complete.
Docs-only file proof remains only for the watcher-owned success signal.

## Intentional

- No branch-protection edit; the existing required check stays in place.
- No watcher merge-policy edit; only the watcher fixture is updated for the new
  `updatedAt` read, and default checker invocations still exit fast.
- No bot identity change; the check still uses exact Codex connector logins.
- The fresh-head window is a race guard, not Codex consent.

## Deferred

- Update the mechanical-enforcement audit note that records the old attestation
  requirement after this policy lands.

Parked hardening: none.

## Verification

- `python -m py_compile scripts/check_ai_reconciliation_live.py && python -m pytest tests/test_check_ai_reconciliation_live.py -q` - 65 passed.
- `python -m pytest tests/test_pr_watcher.py::test_installed_entrypoint_writes_consumer_accepted_snapshot -q --tb=short` - 1 passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/ai_reconciliation_live.yml` | 5 |
| `plans/PR-Live-Reconciliation-Open-Threads-Only.md` | 150 |
| `scripts/check_ai_reconciliation_live.py` | 184 |
| `tests/test_check_ai_reconciliation_live.py` | 225 |
| `tests/test_pr_watcher.py` | 2 |
| **Total** | **566** |
