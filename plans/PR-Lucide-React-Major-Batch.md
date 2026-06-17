# PR-Lucide-React-Major-Batch

## Why this slice exists

The operator asked to move into Bucket 3 major Dependabot review/batching after
the npm CI enrollment slice merged. Bucket 3 contains TypeScript 6, ESLint 10,
and lucide-react 1.x major updates across the npm UI packages.

The root cause for this slice is Dependabot's per-package major-update shape:
lucide-react 1.x landed as four separate PRs (#1644, #1650, #1647, #1652), each
missing Atlas plan/body compliance and several without package-specific CI when
they were opened. This change fixes the review/batch root by superseding those
four lucide-react PRs in one Atlas-shaped batch with the now-enrolled package
checks.

## Scope (this PR)

Ownership lane: security/dependencies
Slice phase: Production hardening

1. Bump `lucide-react` to the Dependabot-requested `1.20.0` line in
   `atlas-admin-ui`, `atlas-churn-ui`, `atlas-intel-ui`, and `atlas-ui`.
2. Keep the batch limited to lucide-react package manifests and lockfiles unless
   package checks expose a required icon API fix.
3. Prove every touched package with its enrolled package-appropriate checks.

### Review Contract

- Acceptance criteria:
  - [ ] The four lucide-react source PRs (#1644, #1650, #1647, #1652) are
        superseded by a single Atlas-shaped branch.
  - [ ] No TypeScript 6 or ESLint 10 upgrades are included.
  - [ ] Each touched package installs cleanly after the lockfile refresh.
  - [ ] Package checks prove the upgrade did not break builds/tests for the
        surfaces that now have npm package CI coverage.
- Affected surfaces: icon component dependency graph for `atlas-admin-ui`,
  `atlas-churn-ui`, `atlas-intel-ui`, and `atlas-ui`.
- Risk areas: icon export/API changes in lucide-react 1.x, package lockfile
  drift, local-vs-CI parity.
- Reviewer rules triggered: R1, R2, R9, R12, R14.

### Files touched

- `atlas-admin-ui/package-lock.json`
- `atlas-admin-ui/package.json`
- `atlas-churn-ui/package-lock.json`
- `atlas-churn-ui/package.json`
- `atlas-intel-ui/package-lock.json`
- `atlas-intel-ui/package.json`
- `atlas-ui/package-lock.json`
- `atlas-ui/package.json`
- `plans/PR-Lucide-React-Major-Batch.md`

## Mechanism

Run `npm install lucide-react@1.20.0 --package-lock-only` in each target
package, preserving the existing manifest style while refreshing the lockfile
to the Dependabot-requested major. Then run the same package checks enrolled by
PR #1664 so this batch does not depend on GitHub's non-compliant Dependabot PR
body shape.

## Intentional

- Do not batch TypeScript 6 or ESLint 10 here. Those are separate compiler and
  lint migrations with known breaking check output.
- Do not touch package code unless the lucide-react 1.x export surface breaks a
  build. Manifest-only is the target shape for this batch.

## Deferred

- TypeScript 6 PRs #1641, #1646, #1648, #1651, and #1654 remain in Bucket 3 for
  a dedicated compiler-migration pass.
- ESLint 10 PRs #1642, #1645, #1649, and #1655 remain in Bucket 3 for a
  dedicated lint-rule migration pass.

Parked hardening: none.

## Verification

- `cd atlas-admin-ui && npm ci && npm audit --audit-level=high && npm run lint && npm run build` - pass.
- `cd atlas-churn-ui && npm ci && npm audit --audit-level=high && npm test && npm run build` - pass, 85 files / 681 tests.
- `cd atlas-intel-ui && npm ci && npm audit --audit-level=high && npm run lint && npm run build` - pass.
- `cd atlas-ui && npm ci && npm audit --audit-level=high && npm run build` - pass.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-admin-ui/package-lock.json` | 8 |
| `atlas-admin-ui/package.json` | 2 |
| `atlas-churn-ui/package-lock.json` | 8 |
| `atlas-churn-ui/package.json` | 2 |
| `atlas-intel-ui/package-lock.json` | 8 |
| `atlas-intel-ui/package.json` | 2 |
| `atlas-ui/package-lock.json` | 8 |
| `atlas-ui/package.json` | 2 |
| `plans/PR-Lucide-React-Major-Batch.md` | 98 |
| **Total** | **138** |
