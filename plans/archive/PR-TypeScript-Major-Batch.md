# PR-TypeScript-Major-Batch

## Why this slice exists

Bucket 3 still has five open TypeScript 5 -> 6 Dependabot PRs (#1641,
#1646, #1648, #1651, #1654). The single-package Dependabot PRs do not satisfy
the Atlas PR contract and do not give one coordinated answer for compiler
compatibility across the UI packages.

This slice batches the web UI TypeScript major updates so compiler breakage is
triaged separately from ESLint 10 rule churn. The attempted mobile inclusion
found that Expo's compatibility checker rejects `typescript@6.0.3` and expects
`~5.9.2`, so mobile remains deferred for an Expo-aligned migration instead of
being forced through a broken package contract.

Review found two follow-up defects before merge. Root cause: the TypeScript
manifest ranges allowed future `6.x` minors beyond the installed
`typescript-eslint` peer range, and one churn UI test queried an async-rendered
witness shortcut synchronously under CI timing. This update fixes both roots by
pinning the affected web packages to the supported TypeScript 6.0 patch line
and awaiting the witness-card link in the failing test.

The local full churn rerun also surfaced a test teardown leak from
`Onboarding.test.tsx`: React still had scheduled work when Vitest tore down the
jsdom `window`. The root is test cleanup that only ran before each test, not
after the final test in the file. This update adds explicit after-test cleanup
so the router tree is unmounted before the environment disappears.

## Scope (this PR)

Ownership lane: security/dependencies
Slice phase: Production hardening

1. Update TypeScript to `~6.0.3` in `atlas-admin-ui`, `atlas-churn-ui`,
   `atlas-intel-ui`, and `atlas-ui`.
2. Keep package code changes limited to compiler-compatibility fixes that are
   required for the TypeScript 6 builds/tests to pass.
3. Prove the batch with each affected package's install, audit, and enrolled
   build/test command.
4. Prove `atlas-mobile` stays out of scope because its Expo compatibility check
   rejects TypeScript 6.0.3 under the current Expo SDK.
5. Address review/CI findings needed for merge: pin the TypeScript range to
   the supported 6.0 line and await the async-rendered churn witness shortcut.
6. Stabilize the churn full-suite rerun by adding explicit after-test cleanup
   to `Onboarding.test.tsx`.

### Review Contract

Acceptance criteria:
- The four web TypeScript 6 source PRs (#1641, #1646, #1648, #1654) are
  superseded by one Atlas-shaped PR.
- Mobile TypeScript PR #1651 is explicitly deferred with the Expo compatibility
  reason named in this plan and PR body.
- TypeScript is bumped consistently to `~6.0.3` in the affected manifests and
  lockfiles.
- Any code edits are direct compiler-compatibility fixes, not lint cleanup or
  product refactors.
- The churn UI test fix is limited to the CI-red async assertion identified in
  review.
- The Onboarding test cleanup fix is limited to unmounting the test router
  after each test; product code is unchanged.
- The affected package checks pass locally before push and in GitHub Actions
  after the PR opens.

Affected surfaces:
- `atlas-admin-ui`
- `atlas-churn-ui`
- `atlas-intel-ui`
- `atlas-ui`

Risk areas:
- TypeScript 6 stricter typechecking may expose real type bugs in UI code.
- `atlas-mobile` currently has an Expo-managed TypeScript compatibility pin and
  must not be forced to TypeScript 6 without an Expo-aligned migration.
- Lockfile collisions with remaining Dependabot PRs are expected and should
  be handled by batching, not by touching unrelated dependency majors.
- Frontend package checks must remain CI-enrolled; passing local commands alone
  is not enough.

Reviewer rules triggered: R1, R2, R9, R12, R14.

### Files touched

- `atlas-admin-ui/package-lock.json`
- `atlas-admin-ui/package.json`
- `atlas-churn-ui/package-lock.json`
- `atlas-churn-ui/package.json`
- `atlas-churn-ui/src/pages/EvidenceExplorer.test.tsx`
- `atlas-churn-ui/src/pages/Onboarding.test.tsx`
- `atlas-intel-ui/package-lock.json`
- `atlas-intel-ui/package.json`
- `atlas-ui/package-lock.json`
- `atlas-ui/package.json`
- `plans/PR-TypeScript-Major-Batch.md`

## Mechanism

Use npm in each affected web package to update only the `typescript` dev
dependency to `~6.0.3`, then run each package's existing compile/build/test
gate. If TypeScript 6 exposes compile errors, fix the narrow typed source
that is actually failing and rerun the package gate that caught it. Keep
`atlas-mobile` pinned to its Expo-compatible TypeScript version because
`npx expo install --check` rejects TypeScript 6.0.3.

The churn UI CI fix does not change product behavior. It changes the failing
test assertion from a synchronous role query to `findByRole`, matching the
component's async witness-card render path.

The Onboarding cleanup fix also stays in test code: it imports Vitest's
`afterEach` and runs Testing Library cleanup after every test to prevent React
work from surviving past jsdom teardown.

## Intentional

- Do not batch ESLint 10 here; lint-rule and formatter churn remains a
  separate Bucket 3 slice.
- Do not touch lucide-react; #1672 already landed that batch.
- Do not include `atlas-mobile` in this TypeScript 6 batch because Expo's
  package compatibility checker expects `typescript@~5.9.2`.
- Do not close the original Dependabot PRs manually in this slice. They are
  source PRs/superseded material unless the operator explicitly asks for
  cleanup.

## Deferred

- ESLint 10 PRs #1642, #1645, #1649, and #1655 remain in Bucket 3 for a
  follow-up lint migration.
- TypeScript 6 mobile PR #1651 remains in Bucket 3 for a future Expo-aligned
  migration.
- Broader frontend lint debt remains out of scope unless TypeScript 6 requires
  a directly related compile fix.

Parked hardening: none.

## Verification

- `cd atlas-admin-ui && npm ci && npm audit --audit-level=high && npm run lint && npm run build` - pass.
- `cd atlas-churn-ui && npm ci && npm audit --audit-level=high && npm test && npm run build` - pass, 85 files / 681 tests.
- `cd atlas-intel-ui && npm ci && npm audit --audit-level=high && npm run lint && npm run build` - pass.
- `cd atlas-ui && npm ci && npm audit --audit-level=high && npm run build` - pass.
- Mobile defer proof: with `typescript@6.0.3`, `cd atlas-mobile && npx expo install --check` fails because Expo expects `typescript@~5.9.2`; after reverting mobile, `cd atlas-mobile && npm ci && npm audit --audit-level=high && npx expo install --check && npx expo config --type public >/tmp/expo-config.out && npm ls react react-dom react-native expo expo-router nativewind react-native-reanimated react-native-worklets @siteed/audio-studio && npm ls @react-native/metro-config @react-native/babel-plugin-codegen @react-native/codegen @react-native/js-polyfills react-native` passes.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-admin-ui/package-lock.json` | 8 |
| `atlas-admin-ui/package.json` | 2 |
| `atlas-churn-ui/package-lock.json` | 8 |
| `atlas-churn-ui/package.json` | 2 |
| `atlas-churn-ui/src/pages/EvidenceExplorer.test.tsx` | 2 |
| `atlas-churn-ui/src/pages/Onboarding.test.tsx` | 6 |
| `atlas-intel-ui/package-lock.json` | 8 |
| `atlas-intel-ui/package.json` | 2 |
| `atlas-ui/package-lock.json` | 8 |
| `atlas-ui/package.json` | 2 |
| `plans/PR-TypeScript-Major-Batch.md` | 158 |
| **Total** | **206** |
