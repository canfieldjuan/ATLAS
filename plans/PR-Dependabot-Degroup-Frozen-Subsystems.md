# PR-Dependabot-Degroup-Frozen-Subsystems

## Why this slice exists

Operator-requested (Juan, 2026-08-18) as part of the Dependabot security-group
triage arc. Both standing security-group PRs are un-mergeable for the same
structural reason, and merging neither means the repo's actual security patches
are 100% blocked:

- **#2397** (`npm-security-and-patches`, 19 updates / 6 dirs) bundles the
  `atlas-mobile` React Native stack (`react-native` 0.81.5 -> 0.87.0 and the
  `react-native-*` / `nativewind` / `@siteed/*` peers) with the web-UI packages.
  atlas-mobile is frozen on Expo SDK 54; those RN bumps are incompatible with it
  (already closed as PRs #2356 mobile leg, #2400, #2401). The one incompatible
  subsystem blocks the 5 web-UI packages' genuine security/patch updates.
- **#2404** (`python-security-and-patches`, 85 updates) bundles the heavy ML/CUDA
  stack (`torch` 2.13.0, `cuda-toolkit` 13.3.1, `cuda-pathfinder`, the
  `nvidia-cuda-*` runtime, `transformers`, `bitsandbytes`, ...) with ~75 real
  security patches. `torch==2.13.0` vs the pinned `cuda-toolkit==13.3.1` produces
  `ERROR: ResolutionImpossible` (verified in the `atlas-main-voice-startup-checks`
  log on head 79dbc1069); the same `torch` 2.13.0 also broke the launcher import
  chain in #2350. One version-locked subsystem blocks all the security patches.

### Problem-derived contract

- Root cause: `.github/dependabot.yml` groups **every** dependency (`patterns: ["*"]`,
  `update-types: [patch, minor]`) across **every** directory into one PR per
  ecosystem, with no `ignore` rules. So a single frozen/version-locked subsystem
  (atlas-mobile RN/Expo; the ML/CUDA set) that cannot take an isolated bump makes
  the whole ecosystem's security group un-mergeable.
- Correct fix must touch/change: `.github/dependabot.yml` only -- add `ignore`
  rules scoped to `version-update:*` types for the frozen subsystems in the npm
  and pip ecosystems, so those deps are excluded from the version-update groups
  while genuine Dependabot **security** updates still surface.
- Must not change: no runtime code, no requirements/constraints files, no other
  ecosystem entries (github-actions, docker, docker-compose), no grouping for the
  packages that must keep flowing (the 5 web-UI packages; the ~75 non-ML python
  security patches). Does not itself merge #2397/#2404 -- those are recreated and
  handled after this lands.

## Scope (this PR)

Ownership lane: dev-workflow/dependabot-config
Slice phase: Workflow/process

1. Add an `ignore` block (version-update-scoped) to the `npm` ecosystem entry for
   `react-native`, `react-native-*`, `@react-native/*`,
   `@react-native-async-storage/*`, `nativewind`, `@siteed/*`, `expo`, `expo-*`.
   (`@react-native/*` and `@react-native-async-storage/*` are scoped SDK-line
   packages that `react-native-*` does not match; `react`/`react-dom`/`@types/react`
   are shared with the web UI and deliberately left updatable.)
2. Add an `ignore` block (version-update-scoped) to the `pip` ecosystem entry for
   `torch`, `torchaudio`, `torchvision`, `transformers`, `accelerate`,
   `bitsandbytes`, `sentence-transformers`, `datasets`, `cuda-toolkit`,
   `cuda-pathfinder`, `nvidia-*`.

### Review Contract

- Acceptance criteria:
  - The file is valid Dependabot v2 config -- settled by `python3 -c "import yaml;
    yaml.safe_load(open('.github/dependabot.yml'))"` (exit 0) and GitHub's own
    dependabot config validation on push.
  - The npm ignore block covers every SDK-line-coupled atlas-mobile dep and no
    web-UI dep -- settled by inspecting the diff: every ignored name
    (`react-native`, `react-native-*`, `@react-native/*`,
    `@react-native-async-storage/*`, `nativewind`, `@siteed/*`, `expo`, `expo-*`)
    exists only in `atlas-mobile/package.json`, not in the other 5 UI packages;
    the scoped `@react-native*` patterns close the gap that `react-native-*` (which
    matches only unscoped names) leaves for `@react-native/metro-config` etc.
    `react`/`react-dom`/`@types/react` are shared with the web UI and are NOT
    ignored (React's own 19.x line, not the RN SDK line).
  - The pip ignore block covers exactly the version-locked ML/CUDA set implicated
    in the resolution conflict / import break -- settled by the #2404
    `ResolutionImpossible` log naming `torch==2.13.0` + `cuda-toolkit==13.3.1`,
    and #2350's torch import-chain regression.
  - Security updates are preserved -- each ignore rule uses `update-types:
    [version-update:semver-major|minor|patch]`, which scopes the ignore to
    version updates only and does not suppress Dependabot security updates.
- Reachability proof: entrypoint is the Dependabot config parser reading
  `.github/dependabot.yml` from the default branch after merge; observable effect
  is that a subsequent `@dependabot recreate` of #2397/#2404 (and all future
  weekly groups) rebuilds the group PRs WITHOUT the ignored subsystem bumps. No
  runtime surface -- this is CI-tooling config.
- Affected surfaces: `.github/dependabot.yml` npm + pip `updates` entries; the
  future/recreated `npm-security-and-patches` and `python-security-and-patches`
  group PRs. No other ecosystem entries touched.
- Risk areas: over-broad ignore that silently drops a web-UI dep (mitigated: the
  ignored names exist only in atlas-mobile) or a needed non-ML python security
  patch (mitigated: the ignore list is the specific version-locked ML/CUDA set,
  not a blanket pattern); accidental suppression of security updates (mitigated:
  version-update-scoped `update-types`).
- Reviewer rules triggered: R11 (dependencies & config), R12 (deployment safety & CI).

### Boundary-change enumeration

N/A - no boundary change. This changes Dependabot update policy (which version
bumps get proposed), not a runtime guard/validator/normalizer/resolver/router or
admission boundary.

### Deployed-config probing

N/A - no guard/config boundary change. `.github/dependabot.yml` is CI-tooling
configuration consumed by Dependabot, not a runtime env/config fallback read by a
guard or admission path.

### Files touched

- `.github/dependabot.yml`
- `plans/PR-Dependabot-Degroup-Frozen-Subsystems.md`

## Mechanism

Dependabot reads `updates[].ignore` from the config on the default branch. Each
`ignore` entry with `update-types` limited to `version-update:semver-{major,minor,
patch}` tells Dependabot to skip **version** updates for that dependency (matched
by exact name or wildcard, e.g. `react-native-*`, `nvidia-*`) while still allowing
**security** updates. Because the group `patterns: ["*"]` only sweeps in deps that
have a proposed update, ignoring the frozen subsystems' version updates removes
them from the group entirely. The remaining group is exactly the mergeable set:
the 5 web-UI packages (npm) and the ~75 non-ML security patches (pip), whose pip
resolution then succeeds because `torch`/`cuda-toolkit` stay at their current,
mutually-compatible pinned versions.

## Intentional

- Version updates for the atlas-mobile RN/Expo stack and the ML/CUDA stack are
  deliberately paused. These subsystems are frozen/version-locked and are upgraded
  as a deliberate whole-set arc, not one package at a time via Dependabot. Genuine
  security updates for them still surface (version-update-scoped ignore).
- Rejected: dropping `/atlas-mobile` from the npm `directories` list entirely --
  that would stop even security updates for atlas-mobile's non-RN deps. The scoped
  ignore keeps those flowing.
- Rejected: manually editing the #2397/#2404 branches to drop the bad bumps --
  pushing a manual commit to a Dependabot branch strips its gate exemptions
  (learned on #2139). The config + `@dependabot recreate` path keeps them as pure
  Dependabot PRs.

## Deferred

- Recreating #2397/#2404 and merging the slimmed security groups happens AFTER
  this config lands (via `@dependabot recreate`), tracked in the same arc.
- A deliberate atlas-mobile Expo-SDK upgrade and a deliberate torch/CUDA-set bump
  remain future arcs, out of scope here.

Parked hardening: none.

## Verification

- Pending before push: `python3 -c "import yaml; yaml.safe_load(open('.github/dependabot.yml'))"` exits 0 (done, prints the 6 npm + 11 pip ignore rules).
- After merge: `@dependabot recreate` on #2397 and #2404; confirm the recreated
  npm PR no longer touches `atlas-mobile/*` RN packages and the recreated pip PR
  no longer bumps torch/cuda-* and resolves (no `ResolutionImpossible`); then
  verify + merge each.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/dependabot.yml` | 61 |
| `plans/PR-Dependabot-Degroup-Frozen-Subsystems.md` | 161 |
| **Total** | **222** |
