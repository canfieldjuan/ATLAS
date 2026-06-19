# PR-Atlas-Churn-UI-ESLint-10

## Why this slice exists

Dependabot opened this slice to move the Atlas Churn UI lint toolchain from `eslint` 9 to `eslint` 10. The initial review concern was the React Hooks plugin peer range: `eslint-plugin-react-hooks` must advertise ESLint 10 support before this major bump can safely land.

## Scope (this PR)

Ownership lane: frontend/tooling
Slice phase: Dependency maintenance

### Files touched

- `atlas-churn-ui/package.json`
- `atlas-churn-ui/package-lock.json`
- `plans/PR-Atlas-Churn-UI-ESLint-10.md`

### Review Contract

Acceptance criteria:

- [ ] `eslint` is updated to `^10.5.0` for `atlas-churn-ui`.
- [ ] The React Hooks ESLint plugin remains on a version whose peer range includes ESLint 10.
- [ ] The lockfile reflects the package manifest without invalid peer dependencies.
- [ ] No runtime application code changes.

Affected surfaces: Atlas Churn UI lint/test tooling only.

Risk areas: ESLint major-version behavior changes and peer dependency compatibility.

Reviewer rules triggered: R1, R12.

## Mechanism

The package manifest and lockfile update `eslint` to `^10.5.0`. The branch also keeps `eslint-plugin-react-hooks` on `^7.1.1`, whose published peer dependency range includes `^10.0.0`, so the lint stack is not left in an unsupported peer state.

## Intentional

- Dependency-only slice for `atlas-churn-ui`.
- No source, route, test, or product behavior changes.
- Keep the React Hooks plugin compatible with the ESLint major bump instead of waiving the peer warning.

## Deferred

- Broader lint rule tuning for any new ESLint 10 behavior remains a follow-up if the project lint command exposes rule-level failures.

## Parked hardening

None.

## Verification

- `NPM Package Checks` passed on the Dependabot head.
- Verified `eslint-plugin-react-hooks@7.1.1` publishes peer support for `eslint` `^10.0.0`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-churn-ui/package.json` | 1 |
| `atlas-churn-ui/package-lock.json` | ~430 |
| `plans/PR-Atlas-Churn-UI-ESLint-10.md` | ~60 |
| **Total** | **~491** |
