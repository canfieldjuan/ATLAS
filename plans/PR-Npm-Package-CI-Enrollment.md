# PR-Npm-Package-CI-Enrollment

## Why this slice exists

PR #1658 cleared the high-severity npm findings, but its re-review accepted a
specific residual risk: `atlas-admin-ui`, `atlas-churn-ui`, `atlas-ui`, and
`atlas-mobile` were verified locally only. The root cause is that Atlas had CI
coverage for Intel and portfolio npm packages, but no package-specific workflow
for the other npm packages touched by the security batch.

This change fixes that workflow gap by enrolling those four packages in a
path-filtered CI matrix that runs the same security/build/test smoke commands
used to prove PR #1658 locally. It drains the `Enroll npm security package
checks in CI` hardening item from `HARDENING.md`.

## Scope (this PR)

Ownership lane: security/dependencies
Slice phase: Workflow/process

1. Add a dedicated `NPM Package Checks` workflow for `atlas-admin-ui`,
   `atlas-churn-ui`, `atlas-ui`, and `atlas-mobile`.
2. Run `npm ci` and `npm audit --audit-level=high` for every enrolled package.
3. Add package-appropriate proof after audit:
   `atlas-admin-ui` lint/build, `atlas-churn-ui` tests/build, `atlas-ui`
   build, and mobile Expo dependency/config/dependency-tree checks.
4. Remove the completed CI-enrollment hardening item from `HARDENING.md`.

### Review Contract

- Acceptance criteria:
  - [ ] The workflow runs only for the newly enrolled packages and its own
        workflow file.
  - [ ] Every matrix entry runs `npm ci` and `npm audit --audit-level=high`.
  - [ ] Admin, churn, and atlas-ui get build coverage; churn also runs its
        Vitest suite; mobile gets Expo dependency/config and dependency-tree
        smoke coverage without pretending it has a build script.
  - [ ] The completed hardening item is removed while the separate Expo 56
        mobile audit cleanup remains parked.
- Affected surfaces: GitHub Actions workflow coverage for npm packages and the
  security/dependencies hardening queue.
- Risk areas: CI runtime, dependency-cache correctness, local-vs-CI parity,
  workflow scope.
- Reviewer rules triggered: R1, R2, R11, R12, R14.

### Files touched

- `.github/workflows/npm_package_checks.yml`
- `HARDENING.md`
- `plans/PR-Npm-Package-CI-Enrollment.md`

## Mechanism

`.github/workflows/npm_package_checks.yml` defines a single matrix job over the
four previously local-only npm packages. Each matrix row uses Node 20.20.0 so
the Vite 8/Rolldown engine floor is explicit, caches npm by that package's
lockfile, runs a clean install, and fails on high-severity npm audit findings.

The final package-check step switches on `matrix.package` so each package gets
the narrow proof it can actually support:

```sh
atlas-admin-ui  -> npm run lint && npm run build
atlas-churn-ui  -> npm test && npm run build
atlas-ui        -> npm run build
atlas-mobile    -> expo install --check, expo config, and npm ls graph probes
```

The workflow is path-filtered to its own file and the four enrolled package
directories so it does not duplicate the existing Intel or portfolio workflows.

## Intentional

- `atlas-churn-ui` and `atlas-ui` lint are still not enrolled. PR #1658
  documented existing lint debt there; this slice closes the CI coverage gap
  for build/test/audit without turning into a broad lint-cleanup PR.
- `atlas-mobile` does not run an app build because the package exposes Expo
  start scripts, not a build script. The CI smoke checks the Expo dependency
  contract, resolved config plugin, and React Native dependency graph that
  blocked the prior review.
- The existing Intel and portfolio workflows remain separate because they
  already carry package-specific tests and routes.

## Deferred

Parked `Mobile Expo 56 audit cleanup` remains deferred; this workflow enrolls
the current high-severity audit gate and Expo 54 compatibility smoke, not the
future Expo 56/RN upgrade.

Parked hardening: `Mobile Expo 56 audit cleanup` remains in `HARDENING.md`.

## Verification

- `python - <<'PY' ... yaml.safe_load('.github/workflows/npm_package_checks.yml') ... PY` - pass.
- `python scripts/audit_workflow_security_posture.py .github/workflows/npm_package_checks.yml` - pass.
- `cd atlas-admin-ui && npm ci && npm audit --audit-level=high && npm run lint && npm run build` - pass.
- `cd atlas-churn-ui && npm ci && npm audit --audit-level=high && npm test && npm run build` - pass, 85 files / 681 tests.
- `cd atlas-mobile && npm ci && npm audit --audit-level=high && npx expo install --check && npx expo config --type public >/tmp/expo-config.out && npm ls react react-dom react-native expo expo-router nativewind react-native-reanimated react-native-worklets @siteed/audio-studio && npm ls @react-native/metro-config @react-native/babel-plugin-codegen @react-native/codegen @react-native/js-polyfills react-native` - pass; plain audit still reports 21 moderate findings requiring the deferred Expo 56/RN/audio follow-up.
- `cd atlas-ui && npm ci && npm audit --audit-level=high && npm run build` - pass.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/npm_package_checks.yml` | 78 |
| `HARDENING.md` | 9 |
| `plans/PR-Npm-Package-CI-Enrollment.md` | 108 |
| **Total** | **195** |
