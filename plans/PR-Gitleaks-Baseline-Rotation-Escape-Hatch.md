# PR-Gitleaks-Baseline-Rotation-Escape-Hatch

## Why this slice exists

The PR-Security-Guardrail-CI review parked a security/workflow follow-up:
`gitleaks-baseline-guard` blocks every change to
`docs/security/gitleaks-baseline.json`, but its own failure message tells
builders to use a dedicated security rotation PR. The root cause is that the
baseline rotation policy exists only as prose, so the CI gate has no
machine-checkable path for legitimate post-rotation baseline updates.

This change fixes the root by moving the baseline-change decision into a
tested checker and wiring CI to allow baseline edits only when the PR carries
an explicit rotation label and stays inside a narrow rotation/doc/plan path
allowlist.

The slice is over the 400 LOC target because the review findings expanded the
root fix from a small script extraction into a trusted-code workflow change
with CI-enrolled regression tests. Splitting the tests or workflow enrollment
out would leave the security gate either untrusted or unprotected in CI.

## Scope (this PR)

Ownership lane: security/workflow
Slice phase: Production hardening

1. Add a controlled Gitleaks baseline rotation path guarded by the
   `security-rotation` PR label plus a narrow changed-file allowlist.
2. Preserve the default fail-closed behavior for ordinary PRs that touch
   `docs/security/gitleaks-baseline.json`.
3. Run the baseline decision from trusted base-branch workflow/script code,
   not from PR-controlled code.
4. Re-run the label-dependent guard when `security-rotation` is added or
   removed.
5. Document the rotation ritual and remove the promoted HARDENING item.

### Review Contract
- Acceptance criteria:
  - [ ] Baseline changes without `security-rotation` still fail.
  - [ ] Baseline changes with `security-rotation` pass only when the diff is
        limited to the baseline, security/hardening docs, and a slice plan.
  - [ ] Initial baseline adoption remains allowed when the trusted base has no
        baseline.
  - [ ] The baseline guard uses `pull_request_target`, checks out the trusted
        base, and fetches the PR head only as git data.
  - [ ] Label changes re-trigger the baseline guard, and labels are parsed as
        JSON rather than comma-split text.
  - [ ] Candidate rotated baselines preserve trusted-base fingerprints.
  - [ ] Checker and workflow-shape regression tests are enrolled in CI.
  - [ ] The human-facing security docs describe the exact label and path
        limits.
- Affected surfaces: CI / security workflow / developer docs.
- Risk areas: security / CI correctness / workflow abuse.
- Reviewer rules triggered: R1, R2, R3, R10, R11, R12, R14.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `.github/workflows/security_guardrails.yml`
- `HARDENING.md`
- `docs/SECURITY_GUARDRAILS.md`
- `plans/PR-Gitleaks-Baseline-Rotation-Escape-Hatch.md`
- `scripts/check_gitleaks_baseline_rotation.py`
- `tests/test_check_gitleaks_baseline_rotation.py`
- `tests/test_pre_push_audit_workflow.py`
- `tests/test_security_guardrails_workflow.py`

## Mechanism

`scripts/check_gitleaks_baseline_rotation.py` compares the PR diff against the
trusted base ref. If the base does not yet have the baseline file, it preserves
the original initial-adoption allow path. If the baseline is unchanged, it exits
green. If the baseline changed, the checker requires the `security-rotation`
label and rejects any changed file outside:

- `docs/security/gitleaks-baseline.json`
- `docs/SECURITY_GUARDRAILS.md`
- `HARDENING.md`
- the slice plan under `plans/` using the PR plan naming convention

The `gitleaks-baseline-guard` job runs on `pull_request_target`, checks out
the trusted base commit, fetches the PR head only as git data, and calls the
base-branch copy of this checker with JSON-encoded PR labels. The workflow
listens for `labeled` and `unlabeled` activity so changing
`security-rotation` re-runs the required guard. For rotation PRs, the checker
also parses the proposed baseline and rejects any candidate that drops
fingerprints already present in the trusted-base baseline.

## Intentional

- The escape hatch is label-based rather than actor-based so it is visible in
  the PR UI and can be removed by reviewers if the diff is not a real rotation.
- The label alone is insufficient; the changed-file allowlist prevents using a
  rotation PR to smuggle product-code or workflow changes alongside a baseline
  update.
- The path allowlist permits this plan doc and the security/hardening docs
  because Atlas requires plan/docs alignment for non-trivial workflow changes.
- The baseline guard uses `pull_request_target` only for metadata and git-data
  inspection; it does not execute PR-controlled code.

## Deferred

- Provider-side credential rotation remains deferred to the existing
  `HARDENING.md` item; this PR only makes the post-rotation baseline update
  possible without weakening normal PR protection.

Parked hardening: promoted and removed `Add controlled Gitleaks baseline
rotation escape hatch`.

## Verification

- Focused pytest for checker and workflow enrollment/shape tests - 19 passed.
- Direct checker smoke using `scripts/check_gitleaks_baseline_rotation.py` against `origin/main` with `security-rotation` - passed, reported "Gitleaks baseline unchanged."
- YAML parse smoke for `.github/workflows/security_guardrails.yml` and `.github/workflows/pre_push_audit.yml` - passed.
- Plan sync check for this plan doc - passed.
- Local review bundle with the PR body file - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 2 |
| `.github/workflows/security_guardrails.yml` | 52 |
| `HARDENING.md` | 9 |
| `docs/SECURITY_GUARDRAILS.md` | 13 |
| `plans/PR-Gitleaks-Baseline-Rotation-Escape-Hatch.md` | 131 |
| `scripts/check_gitleaks_baseline_rotation.py` | 189 |
| `tests/test_check_gitleaks_baseline_rotation.py` | 142 |
| `tests/test_pre_push_audit_workflow.py` | 12 |
| `tests/test_security_guardrails_workflow.py` | 40 |
| **Total** | **590** |
