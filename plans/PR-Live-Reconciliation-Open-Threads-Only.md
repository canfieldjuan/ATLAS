# PR-Live-Reconciliation-Open-Threads-Only

## Why this slice exists

The operator reported that CI/CD is not converging because the live
reconciliation gate can require a final Codex connector approval-style signal
that sometimes never appears. The intended gate is simpler: if the Codex
connector leaves unresolved scoped findings, live reconciliation goes red. A
fresh PR head should not go green before the connector has had a chance to
react, but after that bounded window, no scoped findings should be enough to
merge without waiting for extra Codex consent.

### Problem-derived contract

- Root cause: `scripts/check_ai_reconciliation_live.py` mixes two different
  responsibilities: enforcing real open Codex feedback and requiring a fresh
  current-head clean review/attestation. The second requirement is not the
  desired merge predicate and creates red CI when no open Codex findings exist.
- Correct fix must touch/change: The live reconciliation evaluator must stop
  failing solely because a current-head Codex review or clean-review comment is
  absent; it must still fail for current-head `CHANGES_REQUESTED` reviews and
  unresolved scoped bot threads. It must preserve the docs-only success signal
  consumed by watcher readiness, and it must fail during a short fresh-head
  window when no current-head Codex activity exists yet. The direct unit tests
  must encode those pass and fail cases.
- Must not change: Do not change branch protection, PR opening behavior, review
  bot identities, watcher merge authority, comment-thread filtering, PR body
  reconciliation parsing, product surfaces, or active PR #2251.

## Scope (this PR)

Ownership lane: workflow/live-reconciliation-open-threads-only
Slice phase: Workflow/process

1. Change live reconciliation's quiet-path decision from "current-head Codex
   attestation required" to "no open scoped Codex threads, no current-head
   changes-requested review, and not inside the fresh-head Codex review window."
2. Update the focused live reconciliation tests so the old missing-attestation
   failure becomes a passing quiet-path case while existing open-thread and
   changes-requested failures remain covered.

### Review Contract

- Acceptance criteria:
  1. `evaluate()` returns success when all scoped bot review threads are closed
     and there is no current-head Codex review/clean-comment attestation.
  2. `evaluate()` still returns failure when a current-head Codex connector
     review has state `CHANGES_REQUESTED`, even if threads are closed.
  3. `evaluate()` still returns failure when unresolved scoped Codex review
     threads exist, including when the PR body claims all findings are fixed.
  4. `evaluate()` returns failure during the configured fresh-head review
     window when no current-head Codex activity exists yet, then returns success
     after that window if threads remain clear.
  5. Docs-only PRs with proven Markdown-only diffs still emit the watcher-owned
     docs-only success text when threads are clear and no current-head Codex
     review already satisfies watcher readiness.
  6. The `main()` dry-run path with `--head-sha` and no review attestation
     exits `0` when threads are clear.
- Reachability proof: `python -m pytest tests/test_check_ai_reconciliation_live.py -q`
  exercises the live reconciliation CLI/evaluator entrypoint with injected
  review-thread, review, comment, body, and head-SHA data.
- Affected surfaces: `scripts/check_ai_reconciliation_live.py`,
  `tests/test_check_ai_reconciliation_live.py`, and this plan.
- Risk areas: accidental weakening of open-thread failures, accidental
  weakening of current-head changes-requested failures, stale docs-only-only
  proof paths, and noisy success text that still implies Codex consent is
  required.
- Reviewer rules triggered: R1, R2, R6, R8, R10, R12, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `scripts/check_ai_reconciliation_live.py::evaluate`.
- Replaced-path behaviors: The missing-current-head-attestation failure path is
  replaced with a quiet success path when no scoped bot threads are open and no
  current-head `CHANGES_REQUESTED` review exists, except during the configured
  fresh-head Codex review window.
- Guard-relevant fields: review-thread `isResolved`, first thread comment
  author/path/line/body, review `author.login`, review `commit.oid`, review
  `state`, PR body reconciliation section, and optional `head_sha`.
- Caller x input shape: CI invokes `main()` with live GitHub data; tests invoke
  `evaluate()` and `main()` with injected JSON fixtures.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: Default bot identities remain
  `chatgpt-codex-connector,chatgpt-codex-connector[bot]`.
- Explicit value probe: Focused tests pass explicit `BOTS` and current-head
  review states into `evaluate()`.
- Absent value probe: Focused tests cover no review/clean-comment attestation
  with closed threads both inside and after the fresh-head review window.
- Default-session/default-context probe: N/A - no session-state or environment
  default changes.
- Side-effect ordering: N/A - decision logic remains pure after fetches finish.

### Files touched

- `plans/PR-Live-Reconciliation-Open-Threads-Only.md`
- `scripts/check_ai_reconciliation_live.py`
- `tests/test_check_ai_reconciliation_live.py`

## Mechanism

The evaluator still computes unresolved scoped bot review threads first. If a
current-head Codex connector review requested changes, it records a failure
message. If there are no open threads and no current-head Codex activity yet,
the evaluator fails only while the PR's `updatedAt` timestamp is inside the
configured review window. After the window, no open scoped threads returns
success without requiring a clean Codex approval comment.

Docs-only PRs are not a policy bypass anymore, but their proven Markdown-only
quiet path still emits the prior docs-only success text because
`scripts/pr_watcher.py` consumes that string when calculating watcher readiness.

## Intentional

- No branch-protection edit in this PR. The required check can stay in place;
  its internal pass/fail predicate changes to the desired one.
- No watcher merge-policy edit in this PR. Watcher readiness still has separate
  policy language around Codex attestations and should be handled as its own
  bounded follow-up if the operator wants automated watcher merge readiness to
  match this gate exactly.
- No change to bot identity matching. The scope remains the exact Codex
  connector logins configured for this check.
- The fresh-head review window is a race guard, not Codex consent. It prevents
  immediate green before the connector can react, then quiet no-thread heads
  pass without a final LGTM signal.

## Deferred

- Align watcher/readiness wording and merge-ready predicates that still mention
  Codex review attestations, if the operator wants the watcher layer to use the
  same open-threads-only policy.
- Update the mechanical-enforcement audit note that currently records the old
  attestation requirement, after this policy lands.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_check_ai_reconciliation_live.py -q` - 61 passed.
- `python -m pytest tests/test_content_factory_copy_verification.py::test_scope_lookup_scales_with_negation_scopes_present -q --tb=short` - 1 passed locally after CI unit-gate reported this unrelated node as the only unbaselined failure.
- `python scripts/audit_plan_doc.py plans/PR-Live-Reconciliation-Open-Threads-Only.md` - passed.
- `python scripts/audit_plan_code_consistency.py --base-ref origin/main plans/PR-Live-Reconciliation-Open-Threads-Only.md` - passed.
- `python scripts/sync_pr_plan.py plans/PR-Live-Reconciliation-Open-Threads-Only.md origin/main --check` - passed after syncing generated file/diff-size sections.
- `ATLAS_SESSION_STATE_FILE=SESSION_STATE.codex-live-reconciliation-open-threads.local.md bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas-pr-body-live-reconciliation-open-threads-only.md` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Live-Reconciliation-Open-Threads-Only.md` | 161 |
| `scripts/check_ai_reconciliation_live.py` | 78 |
| `tests/test_check_ai_reconciliation_live.py` | 130 |
| **Total** | **369** |
