# PR-Mobile-NPM-Security-Patches

## Why this slice exists

#1681 is the next security/dependencies queue item after the churn-ui flake
sweep. The raw Dependabot bundle is red because it mixes Expo SDK 54 with React
Native 0.86 / React 19.2 packages. `npx expo install --check` rejects that
graph, and Codex flagged the same issue on `atlas-mobile/package.json`.

Root cause: Dependabot grouped Expo-managed runtime packages with ordinary npm
patches. Expo SDK 54 owns the compatible React, React Native, Reanimated,
screens, safe-area, SVG, worklets, Metro config, and React type versions. This
change fixes the root by keeping the SDK-managed stack on Expo 54-compatible
versions instead of forcing an Expo 56 major upgrade into this security patch
slice.

This PR exceeds the normal LOC budget because the npm lockfile must be
regenerated as one atomic dependency graph. The human-authored diff is limited
to the manifest and plan; the large lockfile churn is mechanical.

## Scope (this PR)

Ownership lane: security/dependencies
Slice phase: Production hardening

1. Keep the safe atlas-mobile npm patch from the Dependabot bundle.
2. Restore Expo SDK 54-managed React Native stack packages to versions accepted
   by `npx expo install --check`.
3. Add an Atlas-formatted PR body and AI reconciliation record so the existing
   #1681 checks can pass without waiving a real compatibility issue.

### Review Contract
- Acceptance criteria:
  - [ ] `npx expo install --check` passes in `atlas-mobile`.
  - [ ] The Codex Expo 54/RN-stack finding is fixed by package versions, not
        hidden or waived.
  - [ ] `npm audit --audit-level=high` still passes; remaining audit output is
        no worse than the Expo-major deferred moderate advisories.
  - [ ] The slice does not upgrade Expo SDK major versions.
- Affected surfaces: mobile dependencies, CI, PR metadata.
- Risk areas: dependency compatibility, mobile runtime, CI stability.
- Reviewer rules triggered: R1, R2, R9, R11, R12, R13, R14.

### Files touched

- `atlas-mobile/package-lock.json`
- `atlas-mobile/package.json`
- `plans/PR-Mobile-NPM-Security-Patches.md`

## Mechanism

The package manifest and lockfile keep `nativewind` at the patched Dependabot
version while restoring SDK-owned packages to Expo 54's compatibility matrix:
React 19.1, React Native 0.81, the matching React Native support packages,
Metro config 0.81, and React types 19.1. The lockfile is regenerated from the
manifest so CI installs the same graph locally verified by Expo.

The PR body will be replaced with the Atlas contract and will record the Codex
finding as fixed by the package correction.

## Intentional

- No Expo 56 upgrade in this slice; that is a separate mobile platform major.
- No broad mobile code changes; this is dependency graph repair.

## Deferred

- Expo 56 / React Native 0.86 platform upgrade remains a dedicated future
  mobile major slice.
- Remaining moderate `npm audit` findings are in Expo/platform-major chains
  (`expo@56`, React Native 0.86, or `@siteed/audio-studio` major/downgrade
  paths). They are not forced into this Expo 54 compatibility repair.

Parked hardening: none added by this slice.

## Verification

- PASS: `npm --prefix atlas-mobile ci`.
- PASS: `cd atlas-mobile && npx expo install --check`.
- PASS: `cd atlas-mobile && npx expo config --type public >/tmp/expo-config.out`.
- PASS: `cd atlas-mobile && npm ls react react-dom react-native expo expo-router nativewind react-native-reanimated react-native-worklets @siteed/audio-studio`.
- PASS: `cd atlas-mobile && npm ls @react-native/metro-config @react-native/babel-plugin-codegen @react-native/codegen @react-native/js-polyfills react-native`.
- PASS: `cd atlas-mobile && npm audit --audit-level=high`.
- Pending before push: `python scripts/sync_pr_plan.py plans/PR-Mobile-NPM-Security-Patches.md --check`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-mobile/package-lock.json` | 1149 |
| `atlas-mobile/package.json` | 4 |
| `plans/PR-Mobile-NPM-Security-Patches.md` | 93 |
| **Total** | **1246** |
