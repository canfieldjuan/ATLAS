# PR-Npm-Security-Patch-Batch

## Why this slice exists

The operator asked for Dependabot security triage after a large set of npm
alerts landed. The root cause is stale npm lockfile graphs across the Atlas UI
packages: Vite/esbuild, React/Router, Expo/React Native, and related test/build
tooling had drifted far enough that Dependabot could not produce an
Atlas-shaped, reviewable security PR on its own.

This change fixes the root for the high-severity npm audit findings in the
touched npm packages by refreshing their dependency graphs and proving
`npm audit --audit-level=high` after clean installs. It intentionally treats the
remaining `atlas-mobile` moderate Expo-chain findings as a parked follow-up
because npm requires `npm audit fix --force` and an Expo 56 major to clear
them. The diff is over the normal 400 LOC budget because package-lock churn is
the artifact under repair; splitting lockfiles would leave the security graph
partially inconsistent and harder to review.

## Scope (this PR)

Ownership lane: security/dependencies
Slice phase: Production hardening

1. Refresh npm dependency manifests and lockfiles for `atlas-admin-ui`,
   `atlas-churn-ui`, `atlas-intel-ui`, `atlas-mobile`, `atlas-ui`, and
   `portfolio-ui` so high-severity audit findings clear after `npm ci`.
2. Move the Vite apps to the current Vite 8 / `@vitejs/plugin-react` 6 line
   where needed, and move `atlas-churn-ui` to Vitest 4 because Vite 7/Vitest 3
   kept the vulnerable graph alive.
3. Apply the minimal compatibility fixes exposed by the dependency updates:
   `atlas-intel-ui` effect/load hooks, `atlas-churn-ui` Vitest 4 test mocks and
   async assertions, and churn test Node globals in
   `atlas-churn-ui/tsconfig.app.json`.
4. Keep `eslint-plugin-react-hooks` on the prior 7.0.1 line for this batch so
   the PR does not become a broad React compiler lint migration.

### Review Contract

- Acceptance criteria:
  - [ ] `npm ci && npm audit --audit-level=high` exits 0 in every touched npm
        package: `atlas-admin-ui`, `atlas-churn-ui`, `atlas-intel-ui`,
        `atlas-mobile`, `atlas-ui`, and `portfolio-ui`.
  - [ ] The Vite packages that changed build successfully, and the churn
        Vitest suite passes on Vitest 4.
  - [ ] Compatibility edits are limited to dependency-update fallout and do not
        change product behavior.
  - [ ] Remaining mobile Expo moderate audit debt is parked in `HARDENING.md`
        with the force-only Expo 56 follow-up named.
- Affected surfaces: frontend dependency graphs, UI build tooling, mobile
  dependency graph, test tooling.
- Risk areas: security, dependency backcompat, CI/runtime compatibility,
  lockfile reviewability.
- Reviewer rules triggered: R1, R2, R9, R10, R11, R12, R14.

### Files touched

- `HARDENING.md`
- `atlas-admin-ui/package-lock.json`
- `atlas-admin-ui/package.json`
- `atlas-churn-ui/package-lock.json`
- `atlas-churn-ui/package.json`
- `atlas-churn-ui/src/components/AtlasHeroScene.test.tsx`
- `atlas-churn-ui/src/components/AtlasRobotScene.test.tsx`
- `atlas-churn-ui/src/pages/Dashboard.test.tsx`
- `atlas-churn-ui/src/pages/IncidentAlerts.test.tsx`
- `atlas-churn-ui/src/pages/PipelineReview.test.tsx`
- `atlas-churn-ui/tsconfig.app.json`
- `atlas-intel-ui/package-lock.json`
- `atlas-intel-ui/package.json`
- `atlas-intel-ui/src/hooks/useApiData.ts`
- `atlas-intel-ui/src/hooks/useFilterParams.ts`
- `atlas-intel-ui/src/pages/BrandCompare.tsx`
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`
- `atlas-intel-ui/src/pages/b2b/B2BCampaigns.tsx`
- `atlas-intel-ui/src/pages/b2b/B2BDashboard.tsx`
- `atlas-mobile/package-lock.json`
- `atlas-mobile/package.json`
- `atlas-ui/package-lock.json`
- `atlas-ui/package.json`
- `plans/PR-Npm-Security-Patch-Batch.md`
- `portfolio-ui/package-lock.json`
- `portfolio-ui/package.json`

## Mechanism

The package manifests and lockfiles are refreshed from the Dependabot npm batch
and then normalized manually where Dependabot left unresolved security or
compatibility gaps:

- Vite web packages use `vite@^8.0.16` and `@vitejs/plugin-react@^6.0.2` so the
  vulnerable Vite/esbuild chain is removed. `atlas-admin-ui` also resolves the
  optional Vite esbuild peer to `esbuild@0.28.1`, outside the advisory range.
- `atlas-churn-ui` moves to `vitest@^4.1.9`; its robot constructor mocks now
  use constructable functions, async UI assertions wait for loaded content, and
  Node globals are available to tests included by the app tsconfig.
- `atlas-intel-ui` keeps the stricter hook lint pass by moving direct effect
  state updates and initial loads behind scheduled callbacks, while preserving
  existing request-id and unmount guards.
- `atlas-mobile` accepts the Expo 54 / React Native 0.86 Dependabot graph and
  updates `@siteed/expo-audio-studio` to the compatible 3.2.1 shim so clean
  installs resolve. Non-breaking `npm audit fix` clears high/critical mobile
  findings; the remaining moderate Expo-chain findings require Expo 56 and are
  deferred.

## Intentional

- `eslint-plugin-react-hooks` remains pinned at `7.0.1`. Updating beyond that
  line surfaces broad existing lint debt in `atlas-churn-ui` and `atlas-ui`
  unrelated to this security batch, including purity and no-explicit-any
  findings across untouched components.
- `atlas-churn-ui npm run lint` and `atlas-ui npm run lint` are not claimed as
  passing verification for this PR. Their failures are pre-existing lint debt
  exposed by current lint configs and are out of scope for the dependency
  security slice.
- `atlas-mobile` does not run an app build in this slice because its package
  exposes only Expo start scripts. The security proof is clean install plus the
  high-severity audit gate.

## Deferred

- Dedicated Expo 56 mobile compatibility slice to clear the remaining moderate
  `atlas-mobile` audit findings that require `npm audit fix --force`.
- Broad React compiler / ESLint cleanup for `atlas-churn-ui` and `atlas-ui`
  lint debt.

Parked hardening: `Mobile Expo 56 audit cleanup` in `HARDENING.md`.

## Verification

- `cd atlas-admin-ui && npm ci && npm audit --audit-level=high` - pass, 0
  vulnerabilities.
- `cd atlas-admin-ui && npm run lint && npm run build` - pass.
- `cd atlas-churn-ui && npm ci && npm audit --audit-level=high` - pass, 0 high
  vulnerabilities.
- `cd atlas-churn-ui && npm run build` - pass.
- `cd atlas-churn-ui && npm run test` - pass, 85 files / 681 tests.
- `cd atlas-intel-ui && npm ci && npm audit --audit-level=high` - pass, 0
  vulnerabilities.
- `cd atlas-intel-ui && npm run lint && npm run build` - pass.
- `cd atlas-mobile && npm ci && npm audit --audit-level=high` - pass; plain
  audit still reports 15 moderate Expo-chain findings requiring Expo 56.
- `cd atlas-ui && npm ci && npm audit --audit-level=high` - pass, 0
  vulnerabilities.
- `cd atlas-ui && npm run build` - pass.
- `cd portfolio-ui && npm ci && npm audit --audit-level=high` - pass, 0
  vulnerabilities.
- `cd portfolio-ui && npm run build` - pass.
- Known non-passing exploratory checks: `cd atlas-churn-ui && npm run lint`
  fails with existing lint debt; `cd atlas-ui && npm run lint` fails with
  existing purity/no-explicit-any lint debt.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 31 |
| `atlas-admin-ui/package-lock.json` | 1649 |
| `atlas-admin-ui/package.json` | 18 |
| `atlas-churn-ui/package-lock.json` | 2649 |
| `atlas-churn-ui/package.json` | 28 |
| `atlas-churn-ui/src/components/AtlasHeroScene.test.tsx` | 2 |
| `atlas-churn-ui/src/components/AtlasRobotScene.test.tsx` | 2 |
| `atlas-churn-ui/src/pages/Dashboard.test.tsx` | 2 |
| `atlas-churn-ui/src/pages/IncidentAlerts.test.tsx` | 2 |
| `atlas-churn-ui/src/pages/PipelineReview.test.tsx` | 2 |
| `atlas-churn-ui/tsconfig.app.json` | 2 |
| `atlas-intel-ui/package-lock.json` | 1909 |
| `atlas-intel-ui/package.json` | 24 |
| `atlas-intel-ui/src/hooks/useApiData.ts` | 61 |
| `atlas-intel-ui/src/hooks/useFilterParams.ts` | 21 |
| `atlas-intel-ui/src/pages/BrandCompare.tsx` | 4 |
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | 51 |
| `atlas-intel-ui/src/pages/b2b/B2BCampaigns.tsx` | 7 |
| `atlas-intel-ui/src/pages/b2b/B2BDashboard.tsx` | 7 |
| `atlas-mobile/package-lock.json` | 5462 |
| `atlas-mobile/package.json` | 18 |
| `atlas-ui/package-lock.json` | 1932 |
| `atlas-ui/package.json` | 22 |
| `plans/PR-Npm-Security-Patch-Batch.md` | 183 |
| `portfolio-ui/package-lock.json` | 319 |
| `portfolio-ui/package.json` | 18 |
| **Total** | **14425** |
