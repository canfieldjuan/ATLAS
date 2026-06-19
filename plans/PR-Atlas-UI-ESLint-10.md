# PR-Atlas-UI-ESLint-10

## Why this slice exists

Dependabot opened this slice to move the Atlas UI lint toolchain from `eslint` 9 to `eslint` 10. The compatibility risk is the lint plugin peer graph: `eslint-plugin-react-hooks` and `typescript-eslint` must advertise ESLint 10 support before the major bump can safely land.

This branch also carries the current Security Guardrails PR-startup fix because that workflow defect is still on `main`; without it, this otherwise dependency-focused PR stays red before the package checks can matter.

## Scope (this PR)

Ownership lane: frontend/tooling
Slice phase: production hardening

### Files touched

- `.github/workflows/security_guardrails.yml`
- `atlas-ui/package.json`
- `atlas-ui/package-lock.json`
- `plans/PR-Atlas-UI-ESLint-10.md`

### Review Contract

Acceptance criteria:

- [ ] `eslint` is updated to `^10.5.0` for `atlas-ui`.
- [ ] React Hooks and TypeScript ESLint lint plugins remain on versions whose peer ranges include ESLint 10.
- [ ] The lockfile reflects the package manifest without invalid peer dependencies.
- [ ] The inherited Security Guardrails startup failure is fixed without broadening the trusted-base secret scan.
- [ ] No runtime application code changes.

Affected surfaces: Atlas UI lint/build tooling and shared CI guardrails.

Risk areas: ESLint major-version behavior changes, peer dependency compatibility, and PR secret-scan scope.

Reviewer rules triggered: R1, R2, R9, R12.

## Mechanism

The package manifest and lockfile update `eslint` to `^10.5.0`. The branch keeps `eslint-plugin-react-hooks` on `^7.1.1` and `typescript-eslint` on `^8.61.1`; the lockfile records those versions with ESLint 10-compatible peer ranges, so the lint stack is not left in the unsupported peer state called out by the automated review.

Security Guardrails gets the same PR-safe fix already used on newer slices: the PR Gitleaks job checks out the PR head, fetches the trusted base ref, and scans only `origin/<base>..HEAD`; the OSV reusable workflow caller declares `actions: read` alongside its existing read/SARIF permissions.

## Intentional

- Dependency slice for `atlas-ui` plus the minimal shared CI startup fix required to make the PR checkable.
- No source, route, test, or product behavior changes.
- Keep React Hooks and TypeScript ESLint plugin versions compatible with the ESLint major bump instead of waiving the peer warning.
- Keep Gitleaks scoped to PR commits so this dependency PR is not held by trusted-base findings.

## Deferred

- Broader lint rule tuning for any new ESLint 10 behavior remains a follow-up if the project lint/build command exposes rule-level failures.
- Landing the Security Guardrails fix on `main` directly remains shared follow-up outside this Dependabot slice.

## Parked hardening

None.

## Verification

- `NPM Package Checks` passed on the original Dependabot head.
- Verified the branch is refreshed onto current `main` so the latest merged fixes are included.
- The resolved automated review finding is addressed by keeping `eslint-plugin-react-hooks@7.1.1` and `typescript-eslint@8.61.1`, which support ESLint 10 in the lockfile peer graph.
- Security Guardrails startup failure matches the OSV reusable workflow permission issue fixed by granting `actions: read` to the OSV caller.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/security_guardrails.yml` | ~12 |
| `atlas-ui/package.json` | 1 |
| `atlas-ui/package-lock.json` | ~440 |
| `plans/PR-Atlas-UI-ESLint-10.md` | ~76 |
| **Total** | **~529** |
