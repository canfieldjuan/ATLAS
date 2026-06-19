# PR-Atlas-Churn-UI-ESLint-10

## Why this slice exists

Dependabot opened this slice to move the Atlas Churn UI lint toolchain from `eslint` 9 to `eslint` 10. The initial review concern was the React Hooks plugin peer range: `eslint-plugin-react-hooks` must advertise ESLint 10 support before this major bump can safely land.

This branch also carries the current Security Guardrails PR-startup fix because that workflow defect is still on `main`; without it, this otherwise dependency-only PR stays red before any package checks can matter.

## Scope (this PR)

Ownership lane: frontend/tooling
Slice phase: production hardening

### Files touched

- `.github/workflows/security_guardrails.yml`
- `atlas-churn-ui/package.json`
- `atlas-churn-ui/package-lock.json`
- `plans/PR-Atlas-Churn-UI-ESLint-10.md`

### Review Contract

Acceptance criteria:

- [ ] `eslint` is updated to `^10.5.0` for `atlas-churn-ui`.
- [ ] The React Hooks ESLint plugin remains on a version whose peer range includes ESLint 10.
- [ ] The lockfile reflects the package manifest without invalid peer dependencies.
- [ ] The inherited Security Guardrails startup failure is fixed without broadening the trusted-base secret scan.
- [ ] No runtime application code changes.

Affected surfaces: Atlas Churn UI lint/test tooling and shared CI guardrails.

Risk areas: ESLint major-version behavior changes, peer dependency compatibility, and PR secret-scan scope.

Reviewer rules triggered: R1, R2, R9, R12.

## Mechanism

The package manifest and lockfile update `eslint` to `^10.5.0`. The branch also keeps `eslint-plugin-react-hooks` on `^7.1.1`, whose published peer dependency range includes `^10.0.0`, so the lint stack is not left in an unsupported peer state.

Security Guardrails gets the same PR-safe fix already used on newer slices: the PR Gitleaks job checks out the PR head, fetches the trusted base ref, and scans only `origin/<base>..HEAD`; the OSV reusable workflow caller declares `actions: read` alongside its existing read/SARIF permissions.

## Intentional

- Dependency slice for `atlas-churn-ui` plus the minimal shared CI startup fix required to make the PR checkable.
- No source, route, test, or product behavior changes.
- Keep the React Hooks plugin compatible with the ESLint major bump instead of waiving the peer warning.
- Keep Gitleaks scoped to PR commits so this dependency PR is not held by trusted-base findings.

## Deferred

- Broader lint rule tuning for any new ESLint 10 behavior remains a follow-up if the project lint command exposes rule-level failures.
- Landing the Security Guardrails fix on `main` directly remains shared follow-up outside this Dependabot slice.

## Parked hardening

None.

## Verification

- `NPM Package Checks` passed on the original Dependabot head before the branch refresh.
- Verified `eslint-plugin-react-hooks@7.1.1` publishes peer support for `eslint` `^10.0.0`.
- Refreshed the branch onto current `main` so the latest merged fixes are included.
- Security Guardrails startup failure matches the OSV reusable workflow permission issue fixed by granting `actions: read` to the OSV caller.
- PR body AI reconciliation uses the local-audit marker `All findings fixed or waived: yes`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/security_guardrails.yml` | ~10 |
| `atlas-churn-ui/package.json` | 1 |
| `atlas-churn-ui/package-lock.json` | ~430 |
| `plans/PR-Atlas-Churn-UI-ESLint-10.md` | ~76 |
| **Total** | **~517** |
