# PR-React-Hooks-Patch-Web-Batch

## Why this slice exists

Dependabot PR #1666 combines two very different dependency changes: a small
web-package `eslint-plugin-react-hooks` patch bump and a broad mobile
React/React Native/Expo graph jump. The web patch is low-risk and keeps the
React hooks lint plugin current; the mobile graph belongs in a dedicated Expo
compatibility slice because it touches runtime framework packages and already
has a parked hardening entry.

This slice applies only the web part of #1666 in Atlas PR shape so the safe
security/dependency patch does not wait behind the mobile migration risk.

## Scope (this PR)

Ownership lane: security/dependencies
Slice phase: Production hardening

1. Update `eslint-plugin-react-hooks` from `^7.0.1` to `^7.1.1` in the four
   web UI packages: `atlas-admin-ui`, `atlas-churn-ui`, `atlas-intel-ui`, and
   `atlas-ui`.
2. Apply only directly required lint-compatibility code changes surfaced by
   the updated hook plugin.
3. Keep `atlas-mobile` out of scope because #1666's mobile updates include
   React Native `0.81.5` -> `0.86.0` and related Expo/runtime graph changes.
4. Prove each affected web package still installs, audits clean at high
   severity, and passes its existing lint/build or test/build gate.

### Review Contract

Acceptance criteria:
- The four web package manifests and lockfile root specs carry
  `eslint-plugin-react-hooks@^7.1.1`.
- `atlas-mobile` files are untouched in this PR.
- No ESLint 10 major, TypeScript, React runtime, or React Native runtime
  changes are bundled into this patch slice.
- Any code edits are narrow responses to diagnostics introduced by the updated
  hook plugin.
- Each affected web package check listed in Verification passes locally and in
  GitHub Actions after the PR opens.

Affected surfaces:
- `atlas-admin-ui`
- `atlas-churn-ui`
- `atlas-intel-ui`
- `atlas-ui`

Risk areas:
- Hook lint plugin updates can introduce new lint diagnostics even without app
  runtime changes.
- `atlas-churn-ui` has the largest/timing-sensitive test suite, so its CI
  result matters more than local optimism.
- #1666's mobile graph must not leak into this low-risk web patch.

Reviewer rules triggered: R1, R2, R9, R12, R14.

### Files touched

- `atlas-admin-ui/package-lock.json`
- `atlas-admin-ui/package.json`
- `atlas-admin-ui/src/App.tsx`
- `atlas-churn-ui/package-lock.json`
- `atlas-churn-ui/package.json`
- `atlas-intel-ui/package-lock.json`
- `atlas-intel-ui/package.json`
- `atlas-ui/package-lock.json`
- `atlas-ui/package.json`
- `plans/PR-React-Hooks-Patch-Web-Batch.md`

## Mechanism

Use npm in each affected web package to update only the
`eslint-plugin-react-hooks` dev dependency to `^7.1.1`, then commit the
resulting manifest and lockfile changes. Run each package's existing CI-facing
gate so the reviewer can verify the hook-plugin bump did not introduce lint,
build, or test failures.

The updated plugin enables a stricter `set-state-in-effect` diagnostic in
`atlas-admin-ui`. The dashboard's initial data load still uses the existing
`fetchAll` path, but the first call is scheduled through a zero-delay browser
timer so state updates happen from the callback path instead of synchronously
inside the effect body. Manual refresh and interval refresh still call the same
fetch function.

## Intentional

- Do not merge #1666 as-is; its mobile React Native/Expo updates are a
  different risk class from the web lint-plugin patch.
- Do not batch ESLint 10 majors here; those remain separate Bucket 3 lint
  migration slices.
- Do not touch React, React DOM, TypeScript, Vite, or runtime code in this
  patch slice.
- Keep the admin code change limited to the lint diagnostic introduced by the
  plugin bump; do not refactor the dashboard data model.

## Deferred

- #1666's mobile React/React Native/Expo graph remains deferred to the existing
  `HARDENING.md` entry "Mobile Expo 56 audit cleanup".
- ESLint 10 PRs #1642, #1645, #1649, and #1655 remain separate lint migration
  slices.

Parked hardening: `Mobile Expo 56 audit cleanup` remains parked because this
slice intentionally avoids the mobile runtime dependency graph.

## Verification

- `cd atlas-admin-ui && npm ci && npm audit --audit-level=high && npm run lint && npm run build` - pass.
- `cd atlas-churn-ui && npm ci && npm audit --audit-level=high && npm test && npm run build` - pass, 85 files / 681 tests.
- `cd atlas-intel-ui && npm ci && npm audit --audit-level=high && npm run lint && npm run build` - pass.
- `cd atlas-ui && npm ci && npm audit --audit-level=high && npm run build` - pass.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-admin-ui/package-lock.json` | 10 |
| `atlas-admin-ui/package.json` | 2 |
| `atlas-admin-ui/src/App.tsx` | 5 |
| `atlas-churn-ui/package-lock.json` | 10 |
| `atlas-churn-ui/package.json` | 2 |
| `atlas-intel-ui/package-lock.json` | 10 |
| `atlas-intel-ui/package.json` | 2 |
| `atlas-ui/package-lock.json` | 10 |
| `atlas-ui/package.json` | 2 |
| `plans/PR-React-Hooks-Patch-Web-Batch.md` | 128 |
| **Total** | **181** |
