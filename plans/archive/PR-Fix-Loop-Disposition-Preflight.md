# PR-Fix-Loop-Disposition-Preflight

## Why this slice exists

The operator asked why Atlas already has drift, compaction, PR ownership, and
hardening/polish deferral rules, yet builders still keep expanding PRs while
chasing Codex connector threads. The local audit at
`docs/audits/agent-check-enforcement-gap-log-2026-08-09.md` identifies one
root loop: `live-reconciliation` is reactive, so a builder can treat every bot
thread as "must edit" before recording the fix-vs-waive decision and bounded
edit set.

This workflow/process slice is justified by recent failed review loops where
red AI reconciliation repeatedly drove PR growth. It adds a small preventative
checkpoint before the next push in a fix loop.

This PR is expected to exceed the 400 LOC target because it adds one new audit
script, its fixture tests, workflow enrollment, and the evidence log that
motivated the change. The behavior is still one narrow gate: require a
structured fix-loop disposition before reconciling non-empty AI findings.

### Problem-derived contract

- Root cause: Atlas requires AI findings to be fixed or waived, but the
  mechanical path does not require the builder to classify the finding, blocking
  predicate, allowed files, and file budget before the next push. That lets
  "red AI reconciliation" become a reflexive edit loop instead of a bounded
  decision.
- Correct fix must touch/change: Add a local audit that inspects the current PR
  body; require `## Fix-loop disposition preflight` only when `## AI
  reconciliation` contains real finding dispositions; require structured
  root-decision, blocking predicate, disposition, allowed files, parked
  hardening, and `Max files: N`; require that `Max files: N` matches the plan's
  Scope budget; wire the audit into `local_pr_review.sh`; add focused tests and
  CI test enrollment; document the new body requirement in `AGENTS.md`.
- Must not change: Do not fetch or classify live GitHub threads in this slice;
  do not change `live-reconciliation` semantics; do not require extra ceremony
  for `no-findings` PR bodies; do not add merge/comment-resolution helpers; do
  not alter product code or customer-visible behavior.

## Scope (this PR)

Ownership lane: workflow/fix-loop-disposition
Slice phase: Workflow/process
Max files: 11

1. Add the local fix-loop disposition audit and run it from the existing local
   PR review bundle when a current PR body is supplied.
2. Prove the audit with synthetic fixtures for no-findings, fixed-in,
   waived-out-of-scope, invalid blocker predicates, missing plan budgets, and
   body/plan budget mismatch.
3. Enroll the new audit tests in the pre-push audit workflow and impacted-test
   selector.

### Review Contract

- Acceptance criteria:
  - A PR body with `## AI reconciliation` containing only `no-findings` passes
    without a fix-loop preflight; settled by
    `tests/test_audit_fix_loop_disposition.py::test_no_findings_needs_no_fix_loop_preflight`.
  - A PR body with a `fixed-in:` AI reconciliation item fails when `## Fix-loop
    disposition preflight` is missing; settled by
    `tests/test_audit_fix_loop_disposition.py::test_fixed_in_reconciliation_requires_preflight`.
  - A valid `fixed-in:` preflight passes only when the body `Max files: N`
    matches the plan Scope `Max files: N`; settled by
    `tests/test_audit_fix_loop_disposition.py::test_valid_fixed_in_preflight_requires_matching_plan_budget`.
  - Waived hardening/polish can pass with `Blocking predicate: not-blocking`,
    while fixed-in findings cannot claim `not-blocking`; settled by
    `tests/test_audit_fix_loop_disposition.py::test_waived_hardening_preflight_uses_not_blocking`
    and
    `tests/test_audit_fix_loop_disposition.py::test_fixed_in_cannot_claim_not_blocking`.
  - `local_pr_review.sh` runs the new audit when a body is supplied; settled by
    `tests/test_local_pr_review.py::test_local_pr_review_runs_fix_loop_disposition_when_body_supplied`.
  - A PR body cannot make the trusted-base audit read an absolute, escaping, or
    symlinked plan path; settled by
    `tests/test_audit_fix_loop_disposition.py::test_absolute_or_escaping_plan_path_is_rejected`
    and
    `tests/test_audit_fix_loop_disposition.py::test_symlinked_plan_path_is_rejected`.
  - The declared `Allowed files` set must cover the changed branch files when a
    base ref is supplied; settled by
    `tests/test_audit_fix_loop_disposition.py::test_allowed_files_must_cover_changed_files`.
  - Each AI reconciliation root needs its own preflight record; settled by
    `tests/test_audit_fix_loop_disposition.py::test_each_ai_root_needs_its_own_preflight_record`
    and
    `tests/test_audit_fix_loop_disposition.py::test_multiple_ai_roots_pass_with_multiple_preflight_records`.
- Reachability proof: The real entrypoint is `bash scripts/local_pr_review.sh
  --current-pr-body-file <body>`, exercised by the local-review integration
  test above. The observable effect is the "Fix-loop disposition preflight"
  check in the local review output.
- Affected surfaces: `scripts/audit_fix_loop_disposition.py`,
  `scripts/local_pr_review.sh`, pre-push audit workflow test enrollment,
  impacted-test selection, `AGENTS.md` fix-loop contract, and the audit log.
- Risk areas: false positives on `no-findings`, body/plan parsing drift,
  workflow test enrollment drift, and accidental expansion into live GitHub
  thread classification.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Boundary-change enumeration

- Boundary path/seam: Local PR review admission for PR bodies with non-empty
  AI reconciliation.
- Replaced-path behaviors: Previously, non-empty AI reconciliation only needed
  a structured fixed/waived record. Now it also needs the fix-loop disposition
  preflight and matching plan file budget.
- Guard-relevant fields: `Plan:`, `## AI reconciliation`, `## Fix-loop
  disposition preflight`, `- Blocking predicate:`, `- Disposition:`, and
  `- Max files:`.
- Caller x input shape: `local_pr_review.sh` passes the current PR body file to
  the audit; direct CLI callers pass `--current-pr-body-file` and `--repo-root`.

### Deployed-config probing

- Deployed/default config values: N/A - no deployed config.
- Explicit value probe: `python scripts/audit_fix_loop_disposition.py
  --repo-root <tmp> --current-pr-body-file <body>` in CLI tests.
- Absent value probe: Missing preflight, missing plan budget, and mismatched
  budgets are negative fixtures.
- Default-session/default-context probe: `no-findings` body passes without a
  preflight.
- Side-effect ordering: The audit is read-only and runs before push through the
  existing local review bundle.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `AGENTS.md`
- `docs/audits/agent-check-enforcement-gap-log-2026-08-09.md`
- `plans/PR-Fix-Loop-Disposition-Preflight.md`
- `scripts/audit_fix_loop_disposition.py`
- `scripts/local_pr_review.sh`
- `scripts/select_impacted_tests.py`
- `tests/test_audit_fix_loop_disposition.py`
- `tests/test_local_pr_review.py`
- `tests/test_pre_push_audit_workflow.py`
- `tests/test_select_impacted_tests.py`

## Mechanism

`scripts/audit_fix_loop_disposition.py` reads the current PR body. If the AI
reconciliation section is absent or contains only `no-findings`, it exits
successfully. If it sees any allowed AI finding disposition (`fixed-in`,
`waived-*`, or `not-applicable`), it requires a `## Fix-loop disposition
preflight` section with the root decision, blocking predicate, disposition,
allowed files, `Max files: N`, and parked hardening target.

For `fixed-in` items, `Blocking predicate: not-blocking` is rejected. For
`waived-*` items, the preflight must use `Blocking predicate: not-blocking`.
Each AI reconciliation root must have its own matching preflight root. The
script accepts only Atlas PR plan paths under `repo_root`, rejects
symlinked plan files, reads the plan's Scope, and requires the plan
`Max files: N` to match the preflight budget. When `--base-ref` is supplied, it
also rejects branch changed files outside the declared `Allowed files` set.

`scripts/local_pr_review.sh` runs the new audit whenever it has a current PR
body file, so `scripts/push_pr.sh` and the trusted
`.github/workflows/pre_push_audit.yml` path get the gate through their existing
local-review call.

## Intentional

- This slice does not classify live GitHub comments. It creates the local body
  contract first, then a later slice can feed live thread summaries into it.
- `no-findings` stays lightweight because there is no fix loop to bound.
- The audit requires a plan Scope budget rather than relying only on the body,
  so the existing plan files-touched audit can enforce the changed-file cap.

## Deferred

- Live Codex-thread classification adapter that maps thread text to
  `fixed-in` vs `waived-*` before edits.
- Shared PR mutation guard for push, merge, and thread-resolution helpers.
- Duplicate-root grouping for live AI reconciliation threads.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_audit_fix_loop_disposition.py tests/test_local_pr_review.py tests/test_select_impacted_tests.py tests/test_pre_push_audit_workflow.py -q`
  - Result: passed, 134 tests.
- `python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**'`
  - Result: passed. Ratchet reported no new brittleness above baseline.
- `ATLAS_SESSION_STATE_FILE=SESSION_STATE.codex-eom-receipts-v2.local.md bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-body-fix-loop-disposition-preflight.md`
  - Result: passed. Local unit mirror reported `baseline=160`, `regressions=0`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 4 |
| `AGENTS.md` | 5 |
| `docs/audits/agent-check-enforcement-gap-log-2026-08-09.md` | 101 |
| `plans/PR-Fix-Loop-Disposition-Preflight.md` | 200 |
| `scripts/audit_fix_loop_disposition.py` | 380 |
| `scripts/local_pr_review.sh` | 14 |
| `scripts/select_impacted_tests.py` | 5 |
| `tests/test_audit_fix_loop_disposition.py` | 504 |
| `tests/test_local_pr_review.py` | 24 |
| `tests/test_pre_push_audit_workflow.py` | 6 |
| `tests/test_select_impacted_tests.py` | 8 |
| **Total** | **1251** |
