# PR-Codex-Review-Scope-Reset

## Why this slice exists

The operator reported that PR throughput collapsed after the GitHub Codex
connector was pointed at the full Atlas reviewer contract. The current workflow
also carries an extra Claude reviewer / `claude-review` gate that the operator no
longer wants. This slice resets Atlas to the actual review model: Codex connector
is the reviewer gate, local scripts remain mechanical checks, and Codex review
threads block only until each in-scope finding is fixed or explicitly waived.

### Problem-derived contract

- Root cause: `AGENTS.md`, watcher scripts, runbooks, and CI tooling still model
  review as a two-gate flow: Codex/bot reconciliation plus a separate Claude
  reviewer status. That is extra process for the current operator workflow, and
  the broad reviewer wording gives Codex too much license to produce large,
  out-of-scope thread sets.
- Diff-budget overage is intentional: most LOC are hard deletions of obsolete
  reviewer-gate scripts/tests/docs and replacement of the large reviewer-session
  bootstrap section with a short Codex connector contract. Splitting the removal
  would leave contradictory review instructions live between PRs.
- Correct fix must touch/change: rewrite the active workflow docs around
  Codex-only review, remove/de-enroll `claude-review` helper surfaces, update
  watcher/readiness logic, and add a deterministic synthetic fixture suite that
  locks the intended Codex finding dispositions.
- Must not change: product behavior, customer-visible surfaces, required
  branch-protection contexts other than removing `claude-review` from local
  readiness logic, `live-reconciliation`, diff-budget/gitleaks gates, plan/body
  admission, or any open PR owned by another lane.

## Scope (this PR)

Ownership lane: workflow/codex-review-scope-reset
Slice phase: Workflow/process

Max files: 32

1. Make Codex connector the only reviewer gate in active workflow docs and
   readiness tooling.
2. Add synthetic fixture tests for Codex review scope and waiver classification.

### Review Contract

- Acceptance criteria:
  - Active docs no longer require a Claude reviewer session or `claude-review`
    status before merge.
  - `scripts/watch_owned_pr.sh` reports readiness from required contexts, zero
    unresolved threads, no `CHANGES_REQUESTED`, and mergeability; it no longer
    reads commit status context `claude-review`.
  - CI/unit-test enrollment no longer includes `check_review_body_r14` or
    `set_claude_review_status`.
  - Codex review rules suppress NITs by default, group duplicate instances,
    distinguish blocker/major/waiver/no-finding dispositions, and keep
    `live-reconciliation` as the scoped Codex thread gate.
  - Synthetic fixtures prove the intended dispositions for docs noise,
    duplicate instances, out-of-scope hardening, missing tests, concrete
    security/data failure, speculative risk, and NIT suppression.
- Reachability proof: N/A - this is workflow/process documentation, script
  readiness logic, and test tooling; no runtime product surface is introduced.
- Affected surfaces: `AGENTS.md`, reviewer-rule docs, watcher/runbook docs,
  pre-push tooling test enrollment, removed Claude-review helper files, and the
  new synthetic fixture checker/tests.
- Risk areas: accidentally weakening Codex thread reconciliation, leaving stale
  `claude-review` readiness dependencies, over-permissive waivers, and creating
  another broad reviewer instruction surface.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `scripts/codex_review_scope_policy.py` classifies
  synthetic Codex review scenarios into blocker/major/waiver/no-finding
  dispositions.
- Replaced-path behaviors: N/A - this is a new test oracle, not a replacement
  for live Codex connector behavior.
- Guard-relevant fields: `duplicate_of`, `category`, `in_scope`,
  `missing_mandatory_proof`, `impact`, `concrete_failure_path`,
  `speculative`, and `one_line_changed_code_fix`.
- Caller x input shape: pytest imports the policy module; CLI self-test invokes
  the built-in fixture set.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - no deployed config or fallback behavior.
- Explicit value probe: N/A.
- Absent value probe: N/A.
- Default-session/default-context probe: N/A.
- Side-effect ordering: N/A.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `AGENTS.md`
- `docs/CURRENT_PRODUCT_DISCIPLINE.md`
- `docs/OVERNIGHT_ARC_WORKFLOW.md`
- `docs/PR_RECONSTRUCTION_PROTOCOL.md`
- `docs/REVIEWER_MERGE_GATE.md`
- `docs/REVIEWER_RULES.md`
- `docs/ai_dev_operating_model.md`
- `docs/ai_dev_operating_model.svg`
- `docs/ci_cd_autonomous_coding_map.md`
- `docs/long_running_agent_monitoring_spec.md`
- `docs/long_running_session_watcher_handoff.md`
- `plans/PR-Codex-Review-Scope-Reset.md`
- `scripts/audit_ai_reconciliation.py`
- `scripts/check_ai_reconciliation_live.py`
- `scripts/check_review_body_r14.py`
- `scripts/codex_review_scope_policy.py`
- `scripts/codex_wake_bridge.py`
- `scripts/local_pr_review.sh`
- `scripts/pr_watcher.py`
- `scripts/set_claude_review_status.py`
- `scripts/watch_owned_pr.sh`
- `tests/maturity_sweep/baseline_scripts.json`
- `tests/test_check_ai_reconciliation_live.py`
- `tests/test_check_review_body_r14.py`
- `tests/test_codex_review_scope_policy.py`
- `tests/test_codex_wake_bridge.py`
- `tests/test_local_pr_review.py`
- `tests/test_pr_watcher.py`
- `tests/test_report_pr_watcher_state.py`
- `tests/test_set_claude_review_status.py`
- `tests/test_watch_owned_pr.py`

## Mechanism

The change removes the second reviewer gate instead of soft-deprecating it:

1. Rewrite the active workflow docs so Codex connector review is the only review
   gate. `live-reconciliation` remains required, but findings can be fixed or
   waived when they are out-of-scope, duplicate, NIT-only, or speculative.
2. Remove/de-enroll the Claude-review helper scripts and tests, and update the
   watcher/runbooks that currently require the `claude-review` commit status.
3. Add a small deterministic classifier over synthetic review fixtures. The
   classifier is not an adapter to Codex; it is a local test oracle for the
   intended policy outcomes.
4. Keep the portable and installed PR watchers on the same execution model:
   non-outdated unresolved Codex threads block readiness, non-Codex threads do
   not, and a current-head Codex review must be proven through paginated review
   data before any ready-for-human-merge handoff.

## Intentional

- Hard-remove the Claude-review gate rather than keeping an obsolete fallback;
  the operator explicitly said it is extra and not needed.
- Keep `live-reconciliation` because the Codex connector is still the reviewer
  gate; the change is scoped waiver/finding discipline, not "ignore Codex."
- Do not create live canary PRs in this slice.

## Deferred

- Optional later canary: after the text and synthetic fixtures are stable, run
  one real low-risk PR through Codex connector and measure thread count/finding
  quality against the synthetic expectations.

Slice parking predicate: waive duplicate instances, out-of-scope hardening,
NIT-only polish, and speculative risks with no concrete failure path from this
PR's diff; do not park a triggered-rule failure, current-head Codex review
freshness failure, required CI failure, or missing mandatory proof.

Parked hardening: none under that predicate.

## Verification

- `python scripts/codex_review_scope_policy.py --self-test` - passed, 9 fixtures.
- `python -m pytest tests/test_codex_wake_bridge.py tests/test_codex_review_scope_policy.py tests/test_check_ai_reconciliation_live.py tests/test_local_pr_review.py tests/test_pre_push_audit_workflow.py tests/test_audit_ai_reconciliation.py tests/test_audit_plan_code_consistency.py tests/test_pr_watcher.py tests/test_watch_owned_pr.py tests/test_report_pr_watcher_state.py -q` - 214 passed.
- `bash -n scripts/watch_owned_pr.sh scripts/local_pr_review.sh` - passed.
- `python -m py_compile scripts/check_ai_reconciliation_live.py scripts/codex_review_scope_policy.py scripts/audit_ai_reconciliation.py scripts/pr_watcher.py scripts/codex_wake_bridge.py` - passed.
- `python scripts/audit_plan_code_consistency.py --base-ref origin/main plans/PR-Codex-Review-Scope-Reset.md` - passed.
- `python scripts/sync_pr_plan.py plans/PR-Codex-Review-Scope-Reset.md origin/main --check` - passed.
- `python scripts/audit_plan_doc.py plans/PR-Codex-Review-Scope-Reset.md` - passed.
- `python scripts/audit_plan_doc_files_touched.py plans/PR-Codex-Review-Scope-Reset.md origin/main` - passed.
- `python scripts/audit_plan_doc_diff_size.py plans/PR-Codex-Review-Scope-Reset.md origin/main` - passed, estimate 2996 actual 2951.
- `git diff --check -- . ':!node_modules'` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 4 |
| `AGENTS.md` | 461 |
| `docs/CURRENT_PRODUCT_DISCIPLINE.md` | 2 |
| `docs/OVERNIGHT_ARC_WORKFLOW.md` | 32 |
| `docs/PR_RECONSTRUCTION_PROTOCOL.md` | 16 |
| `docs/REVIEWER_MERGE_GATE.md` | 84 |
| `docs/REVIEWER_RULES.md` | 98 |
| `docs/ai_dev_operating_model.md` | 60 |
| `docs/ai_dev_operating_model.svg` | 12 |
| `docs/ci_cd_autonomous_coding_map.md` | 4 |
| `docs/long_running_agent_monitoring_spec.md` | 2 |
| `docs/long_running_session_watcher_handoff.md` | 32 |
| `plans/PR-Codex-Review-Scope-Reset.md` | 219 |
| `scripts/audit_ai_reconciliation.py` | 6 |
| `scripts/check_ai_reconciliation_live.py` | 80 |
| `scripts/check_review_body_r14.py` | 202 |
| `scripts/codex_review_scope_policy.py` | 206 |
| `scripts/codex_wake_bridge.py` | 17 |
| `scripts/local_pr_review.sh` | 17 |
| `scripts/pr_watcher.py` | 207 |
| `scripts/set_claude_review_status.py` | 150 |
| `scripts/watch_owned_pr.sh` | 49 |
| `tests/maturity_sweep/baseline_scripts.json` | 7 |
| `tests/test_check_ai_reconciliation_live.py` | 83 |
| `tests/test_check_review_body_r14.py` | 195 |
| `tests/test_codex_review_scope_policy.py` | 73 |
| `tests/test_codex_wake_bridge.py` | 6 |
| `tests/test_local_pr_review.py` | 28 |
| `tests/test_pr_watcher.py` | 276 |
| `tests/test_report_pr_watcher_state.py` | 3 |
| `tests/test_set_claude_review_status.py` | 175 |
| `tests/test_watch_owned_pr.py` | 190 |
| **Total** | **2996** |
