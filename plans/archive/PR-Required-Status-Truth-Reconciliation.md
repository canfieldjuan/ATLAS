# PR-Required-Status-Truth-Reconciliation

## Why this slice exists

Issue #2035 tracks the CI/CD enforcement gap where the merge gate can appear
stronger than the live branch-protection payload actually is. After #2253,
local PR admission and Codex/live-reconciliation are tighter, but live
branch protection still requires only `live-reconciliation` plus the two
Gitleaks checks, while docs and the stale #2130 PR point at a broader required
set. This workflow/process slice is admitted now because that mismatch is a
merge-safety blocker: builders and reviewers cannot know which mechanical
checks are actually merge-blocking unless the audit contract separates the
target required set from the current live GitHub setting.

### Problem-derived contract

- Root cause: the required-status audit/docs conflate desired branch-protection
  policy with current live repository configuration, and the target set is
  missing current PR-shape gates (`plan-admission`, `session-lane`, and
  `review-contract`) that now exist as trusted-base producers.
- Correct fix must touch/change: update the required-status checker default
  contract, branch-protection audit workflow trigger coverage, security docs,
  and focused tests so the intended required set is explicit, source-pinned to
  GitHub Actions, and distinguishable from the current live payload until the
  external branch-protection PATCH lands.
- Must not change: live GitHub branch protection, `claude-review`, product code,
  EOM lanes, Dependabot PRs, #2130's branch, unit-gate enrollment, or any
  workflow execution model beyond the branch-protection audit trigger inputs.

## Scope (this PR)

Ownership lane: workflow/required-status-truth
Slice phase: Workflow/process

1. Reconcile the required-status checker/docs with the current intended merge
   gate set.
2. Add focused regression coverage for the full target set and for the current
   live payload failing until enrollment is patched.

### Review Contract

- Acceptance criteria:
  1. `scripts/check_required_status_checks.py` default required contexts are the
     intended target set: `live-reconciliation`, `diff-budget`,
     `plan-admission`, `session-lane`, `review-contract`, `pr-body-contract`,
     `Gitleaks PR secret scan`, and `Gitleaks baseline growth guard`.
  2. The checker still fails when any target context is absent or not pinned to
     GitHub Actions app id `15368`.
  3. A fixture matching the current live branch-protection payload fails on the
     missing target contexts, proving the audit exposes today's drift instead of
     silently passing it.
  4. `.github/workflows/branch_protection_required_checks.yml` reruns on edits
     to every producer workflow named in the target set.
  5. `docs/SECURITY_GUARDRAILS.md` names the target required set and clearly
     states the live REST branch-protection PATCH remains separate until
     enrollment is applied.
- Reachability proof: `python -m pytest tests/test_security_guardrails_workflow.py
  -q` exercises the checker entrypoint and workflow/docs assertions.
- Affected surfaces: required-status checker, security guardrail docs,
  branch-protection audit workflow trigger paths, focused workflow tests, plan.
- Risk areas: false green branch-protection audit, docs claiming unpatched live
  settings, missing producer trigger coverage, wrong-source required contexts,
  and stale #2130 drift.
- Reviewer rules triggered: R1, R2, R10, R11, R12, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: branch-protection required-status audit.
- Replaced-path behaviors: the target required-context set expands from four
  contexts to eight and keeps source pinning as the pass condition.
- Guard-relevant fields: `contexts[]`, `checks[].context`, `checks[].app_id`,
  default required context order, workflow `paths`, and docs wording that
  distinguishes target policy from live settings.
- Caller x input shape: top-level required-status payload, nested
  `required_status_checks` payload, contexts-only legacy payload, checks-only
  app-pinned payload, wrong-app payload, and current-live-shaped payload.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: live API currently returns only
  `live-reconciliation`, `Gitleaks PR secret scan`, and `Gitleaks baseline
  growth guard` as required.
- Explicit value probe: tests include a full target payload with GitHub Actions
  app id `15368` for every context and expect PASS.
- Absent value probe: tests include missing-context and current-live-shaped
  payloads and expect FAIL.
- Default-session/default-context probe: checker defaults are used when no
  `--required` override is passed.
- Side-effect ordering: no live branch-protection mutation in this PR; REST
  enrollment remains Deferred.

### Files touched

- `.github/workflows/branch_protection_required_checks.yml`
- `docs/SECURITY_GUARDRAILS.md`
- `plans/PR-Required-Status-Truth-Reconciliation.md`
- `scripts/check_required_status_checks.py`
- `tests/test_security_guardrails_workflow.py`

## Mechanism

The checker's default required-context list becomes the explicit target merge
gate set. Focused tests exercise the full app-pinned target payload, legacy and
wrong-source failures, missing-context failures, and the current live payload
that should fail until the branch-protection setting is patched. The
branch-protection audit workflow expands its `push.paths` coverage to include
the trusted-base producer workflows for every target context. Docs name the
target set and call out that the live REST PATCH is still a separate settings
operation.

## Intentional

- No `claude-review` required check; Codex connector threads plus
  `live-reconciliation` remain the reviewer gate.
- No live branch-protection PATCH in this PR; repository settings mutation is an
  external operation that must preserve the current checks.
- No unit-gate enrollment in this slice; #2035 keeps that as a separate
  operator-policy decision.
- Supersedes the stale intent of #2130 from current `main` rather than editing
  that old branch in place.

## Deferred

- Apply the minimal REST branch-protection PATCH after this audit contract lands:
  preserve existing checks and add `diff-budget`, `plan-admission`,
  `session-lane`, `review-contract`, and `pr-body-contract`, then re-fetch the
  live payload and run the checker.
- Close or supersede #2130 after this replacement PR is open.
- #2035 G1.1 unit-gate enrollment remains an operator-policy follow-up.

Default parking predicate: park only repository-settings mutations and
operator-policy enrollments that cannot be represented as tracked code in this
PR.

Parked hardening: none.

## Verification

- PASS: `python -m pytest tests/test_security_guardrails_workflow.py -q` (16
  passed).
- EXPECTED FAIL: current-live-shaped payload probe against
  `scripts/check_required_status_checks.py` exited `1` and reported missing
  `diff-budget`, `plan-admission`, `session-lane`, `review-contract`, and
  `pr-body-contract`.
- PASS: full target payload probe against `scripts/check_required_status_checks.py`
  reported all eight target contexts pinned to GitHub Actions app id `15368`.
- PASS: `gh api repos/canfieldjuan/ATLAS/branches/main/protection/required_status_checks`
  returned live required checks `live-reconciliation`,
  `Gitleaks PR secret scan`, and `Gitleaks baseline growth guard`, matching the
  documented unpatched state.
- PASS: `ATLAS_SESSION_STATE_FILE=SESSION_STATE.codex-required-status-truth.local.md
  ATLAS_CURRENT_PR_BODY_FILE=/tmp/atlas-pr-body-required-status-truth-reconciliation.md
  bash scripts/local_pr_review.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/branch_protection_required_checks.yml` | 6 |
| `docs/SECURITY_GUARDRAILS.md` | 25 |
| `plans/PR-Required-Status-Truth-Reconciliation.md` | 171 |
| `scripts/check_required_status_checks.py` | 4 |
| `tests/test_security_guardrails_workflow.py` | 105 |
| **Total** | **311** |
