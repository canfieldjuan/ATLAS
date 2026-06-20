# PR-Atlas-Churn-UI-ESLint-10

## Why this slice exists

Dependabot opened this slice to move the Atlas Churn UI lint toolchain from `eslint` 9 to `eslint` 10. The initial review concern was the React Hooks plugin peer range: `eslint-plugin-react-hooks` must advertise ESLint 10 support before this major bump can safely land.

The diff is over the normal 400 LOC budget because npm rewrites the package-lock
dependency graph for the ESLint major; the behavioral surface remains limited to
the `atlas-churn-ui` lint/test/build toolchain.

## Scope (this PR)

Ownership lane: frontend/tooling
Slice phase: Production hardening

1. Update `atlas-churn-ui` from `eslint` 9 to ESLint 10.
2. Keep the React Hooks plugin on a version whose peer range includes ESLint 10.
3. Tighten the package Node engine range to match ESLint 10's Node 22 floor.
4. Add the narrow ESLint config compatibility shim needed for the upgraded React
   Hooks plugin and fix small lint residues exposed by the new lint stack.

### Files touched

- `atlas-churn-ui/eslint.config.js`
- `atlas-churn-ui/package-lock.json`
- `atlas-churn-ui/package.json`
- `atlas-churn-ui/src/auth/AuthContext.tsx`
- `atlas-churn-ui/src/components/DataTable.tsx`
- `atlas-churn-ui/src/components/SubscriptionModal.tsx`
- `atlas-churn-ui/src/pages/BlogReview.tsx`
- `atlas-churn-ui/src/pages/Report.tsx`
- `atlas-churn-ui/src/pages/Watchlists.test.tsx`
- `plans/PR-Atlas-Churn-UI-ESLint-10.md`

### Review Contract

Acceptance criteria:

- [ ] `eslint` is updated to `^10.5.0` for `atlas-churn-ui`.
- [ ] The React Hooks ESLint plugin remains on a version whose peer range includes ESLint 10.
- [ ] `engines.node` no longer allows Node 22.12.x, which ESLint 10 rejects.
- [ ] The lockfile reflects the package manifest without invalid peer dependencies.
- [ ] `npm run lint`, `npm test`, and `npm run build` pass in `atlas-churn-ui`.
- [ ] Source edits are limited to lint-only compatibility fixes with no route,
      workflow, or product behavior changes.

Affected surfaces: Atlas Churn UI lint/test/build tooling.

Risk areas: ESLint major-version behavior changes and peer dependency compatibility.

Reviewer rules triggered: R1, R2, R3, R9, R12.

## Mechanism

The package manifest and lockfile update `eslint` to `^10.5.0`. The branch also keeps `eslint-plugin-react-hooks` on `^7.1.1`, whose published peer dependency range includes `^10.0.0`, so the lint stack is not left in an unsupported peer state.

`eslint-plugin-react-hooks@7.1.1` enables React Compiler-style diagnostics in
its flat recommended config. This PR disables those compiler diagnostics for
existing churn UI render/ref/effect patterns while preserving core hooks and
TypeScript linting. The package Node engine range changes from
`^20.19.0 || >=22.12.0` to `^20.19.0 || ^22.13.0 || >=24` to match ESLint 10's
own engine requirements.

The branch has been refreshed onto current `origin/main`; stale Security
Guardrails workflow drift from the older branch base is intentionally not part
of this PR.

## Intentional

- Dependency slice for `atlas-churn-ui`; source edits are lint unlocks only.
- No route, test, workflow, or product behavior changes.
- Keep the React Hooks plugin compatible with the ESLint major bump instead of waiving the peer warning.

## Deferred

- React Compiler rule adoption for the existing Atlas Churn UI render/effect
  patterns remains deferred to a dedicated app-code cleanup slice.
- The four existing `react-hooks/exhaustive-deps` lint warnings remain deferred;
  this PR keeps lint exit status green without broadening into route-state
  refactors.
- Parked hardening: none.

## Verification

- `NPM Package Checks` passed on the original Dependabot head before the branch refresh.
- Verified `eslint-plugin-react-hooks@7.1.1` publishes peer support for `eslint` `^10.0.0`.
- Verified `eslint@10.5.0` requires Node `^20.19.0 || ^22.13.0 || >=24`, and
  updated the package engine range to match.
- Merged current `origin/main` and resolved stale Security Guardrails workflow
  drift by keeping `origin/main`.
- `npm --prefix atlas-churn-ui ci`
- `npm --prefix atlas-churn-ui run lint` (passes with four existing
  `react-hooks/exhaustive-deps` warnings)
- `npm --prefix atlas-churn-ui test` (85 files / 681 tests passed)
- `npm --prefix atlas-churn-ui run build`
- PR body AI reconciliation uses the local-audit marker `All findings fixed or waived: yes`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-churn-ui/eslint.config.js` | 13 |
| `atlas-churn-ui/package-lock.json` | 434 |
| `atlas-churn-ui/package.json` | 4 |
| `atlas-churn-ui/src/auth/AuthContext.tsx` | 1 |
| `atlas-churn-ui/src/components/DataTable.tsx` | 1 |
| `atlas-churn-ui/src/components/SubscriptionModal.tsx` | 5 |
| `atlas-churn-ui/src/pages/BlogReview.tsx` | 5 |
| `atlas-churn-ui/src/pages/Report.tsx` | 5 |
| `atlas-churn-ui/src/pages/Watchlists.test.tsx` | 8 |
| `plans/PR-Atlas-Churn-UI-ESLint-10.md` | 112 |
| **Total** | **588** |
