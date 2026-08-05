# PR-Required-Workflow-Enrollment-Audit

## Why this slice exists

The mechanical-enforcement audit's next open follow-up is
`PR-Required-Workflow-Enrollment-Audit`: decide whether the existing CI-only
process checks `pre-push-audit` and `unit-gate` should remain visible-but-not
branch-required or become part of live branch protection. This follows the
2026-08-04 required-status alignment, which made the registry's existing
`branch_required` contexts live, but intentionally left these two checks at
`ci_blocking_not_required`.

User request: after repeated red checks and review churn, identify the
mechanical checks that are not being respected and fix the root rather than
patching PR by PR.

### Problem-derived contract

- Root cause: Atlas now has a live branch-required registry, but two checks that
  look merge-relevant in PRs (`pre-push-audit`, `unit-gate`) are still
  classified as CI-blocking-but-not-required without a current evidence record
  explaining why. That ambiguity lets agents treat red or pending checks as
  optional in some threads and blocking in others.
- Correct fix must touch/change: gather current workflow evidence for
  `pre-push-audit` and `unit-gate`; record the enrollment decision and rationale
  in a durable audit doc; update `ci/gates.yml` and related docs only if the
  evidence supports changing their enforcement class; keep live branch
  protection untouched unless the registry changes and the live GitHub payload is
  explicitly verified after the change.
- Must not change: do not change product behavior, EOM lanes, dependency PRs,
  workflow job implementations, test selection logic, or branch protection by
  default. Do not make `unit-gate` branch-required if the evidence shows its
  runtime/flakiness/self-owned workflow semantics would increase churn without
  closing a proven merge-safety gap.

## Scope (this PR)

Ownership lane: dev-workflow/required-workflow-enrollment
Slice phase: Workflow/process

1. Audit current `pre-push-audit` and `unit-gate` workflow behavior, including
   live enforcement class, trust model, recent run outcomes, and runtime/churn
   implications.
2. Record the decision: keep one or both as `ci_blocking_not_required`, or
   promote one or both to `branch_required` with matching documentation.
3. If the decision changes the registry, update only the registry/docs required
   to keep the branch-protection checker and docs consistent.

### Review Contract

- Acceptance criteria:
  1. The audit doc names the current enforcement class for `pre-push-audit` and
     `unit-gate`, settled by `ci/gates.yml`.
  2. The audit doc cites recent GitHub Actions outcome/runtime evidence for both
     workflows, settled by the recorded `gh run list` / `gh run view` commands
     in the audit doc.
  3. The audit doc names the trust model for both workflows, settled by
     `.github/workflows/pre_push_audit.yml`, `.github/workflows/unit_gate.yml`,
     and `docs/SECURITY_GUARDRAILS.md`.
  4. Any registry or docs enforcement-class change is mirrored consistently in
     `ci/gates.yml`, `docs/ci_cd_autonomous_coding_map.md`, and the new audit
     doc; if no registry change is made, the audit doc explicitly says why.
  5. `python scripts/check_required_status_checks.py --payload-file <fresh live
     payload>` still passes after the slice.
- Reachability proof: the real entrypoint is the branch-protection checker
  command against a fresh GitHub payload; the observable effect is PASS after
  the recorded decision.
- Affected surfaces: `ci/gates.yml` if enforcement changes; workflow/process
  docs and the plan.
- Risk areas: accidental branch-protection churn, over-requiring slow/flaky
  checks, stale enforcement docs, and widening outside the workflow lane.
- Reviewer rules triggered: R1, R2, R10, R11, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: live merge gate enrollment for CI contexts.
- Replaced-path behaviors: before this slice, `pre-push-audit` and `unit-gate`
  are visible CI checks but not registry branch-required checks.
- Guard-relevant fields: `ci/gates.yml` `enforcement`, workflow context names,
  branch-protection required check payload, and GitHub Actions app source.
- Caller x input shape: PRs that produce `pre-push-audit` and `unit-gate`
  contexts.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: live `main` branch protection required-status
  payload.
- Explicit value probe: run `scripts/check_required_status_checks.py` against a
  fresh live payload after the decision.
- Absent value probe: audit doc records whether `pre-push-audit` and `unit-gate`
  are absent from the required set by design or promoted.
- Default-session/default-context probe: N/A; no runtime app config.
- Side-effect ordering: do not patch live branch protection unless the registry
  changes and the branch-protection update is explicitly part of the verified
  decision.

### Files touched

- `.github/workflows/unit_gate.yml`
- `docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md`
- `docs/audits/required-workflow-enrollment-audit-2026-08-04.md`
- `docs/ci_cd_autonomous_coding_map.md`
- `docs/ci_cd_runtime_duplication_audit.md`
- `plans/PR-Required-Workflow-Enrollment-Audit.md`
- `tests/test_security_guardrails_workflow.py`

## Mechanism

This slice uses code and live CI as evidence, then records a decision. The
branch-required registry remains the source of truth; the audit doc explains
whether `pre-push-audit` and `unit-gate` should join that registry or remain
visible CI. If the registry changes, documentation and live-payload verification
move with it in the same slice.

The evidence does not support promotion yet, so this slice leaves `ci/gates.yml`
and live branch protection unchanged, records the blockers, updates stale
workflow/process docs, and fixes the stale security-guardrails test assertion
that current CI is already reporting.

## Intentional

- No live branch-protection mutation before evidence: the previous slice already
  aligned the known registry gap, and this slice is about deciding whether the
  registry itself should change.
- `pre-push-audit` and `unit-gate` are evaluated separately; one can be promoted
  without forcing the other.

## Deferred

- Unit-gate selector coverage for branch-protection/security docs so the stale
  docs/test mismatch is caught before merge.
- Trusted-base `pre-push-audit` PR-side docs/test consistency probe, if a safe
  data-only probe can be designed without executing untrusted PR code.
- Re-run enrollment after those blockers close; live branch protection changes
  only if `ci/gates.yml` promotes one of these contexts.

Parked hardening: none.

## Verification

- `gh pr list --state open --json number,title,headRefName,author,isDraft,mergeStateStatus --limit 30`
- `git fetch origin main --prune`
- `git log --oneline -15 origin/main`
- `gh api repos/canfieldjuan/ATLAS/branches/main/protection/required_status_checks > /tmp/atlas-required-status-checks-live-required-workflow.json`
- `python scripts/check_required_status_checks.py --payload-file /tmp/atlas-required-status-checks-live-required-workflow.json` - PASS
- `gh run list --workflow pre_push_audit.yml --limit 20 --json databaseId,status,conclusion,createdAt,updatedAt,event,headBranch,displayTitle`
- `gh run list --workflow unit_gate.yml --limit 20 --json databaseId,status,conclusion,createdAt,updatedAt,event,headBranch,displayTitle`
- `gh run view 30961567962 --log-failed`
- `gh run view 30960553450 --log-failed`
- `python -m pytest tests/test_security_guardrails_workflow.py::test_security_guardrails_docs_name_required_gitleaks_checks` - 1 passed
- `python -m pytest tests/test_security_guardrails_workflow.py` - 35 passed
- `python scripts/audit_pr_body.py /tmp/atlas-pr-body-required-workflow-enrollment.md --base-ref origin/main` - PASS
- `python scripts/audit_ai_reconciliation.py --current-pr-body-file /tmp/atlas-pr-body-required-workflow-enrollment.md --require` - PASS

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/unit_gate.yml` | 5 |
| `docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md` | 15 |
| `docs/audits/required-workflow-enrollment-audit-2026-08-04.md` | 108 |
| `docs/ci_cd_autonomous_coding_map.md` | 6 |
| `docs/ci_cd_runtime_duplication_audit.md` | 26 |
| `plans/PR-Required-Workflow-Enrollment-Audit.md` | 173 |
| `tests/test_security_guardrails_workflow.py` | 4 |
| **Total** | **337** |
