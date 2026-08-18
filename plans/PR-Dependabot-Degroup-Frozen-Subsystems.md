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

**Over-budget justification (indivisible):** the slice runs ~490 LOC, over the 400
target, but the actual config change is ~110 lines in one file. The overage is the
mandatory AGENTS.md plan doc plus the CI regression test the reviewer required
(Codex R2), and the test must ship WITH the config it guards -- splitting it into a
separate PR would leave the config unguarded on merge and defeat the review's own
request. The remainder is the plan doc, which is not separable from the slice.

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
Max files: 4

1. Split the single `npm` entry into two: (a) the 5 web-UI directories with NO
   ignores (react/react-dom/@types/react and the toolchain update freely), and
   (b) a separate `/atlas-mobile` entry whose `ignore` freezes the FULL Expo SDK
   stack -- `react-native`, `react-native-*`, `@react-native/*`,
   `@react-native-async-storage/*`, `nativewind`, `@siteed/*`, `expo`, `expo-*`,
   and (Expo-owned on SDK 54) `react`, `react-dom`, `@types/react`. Isolating mobile
   is required to freeze its Expo-owned React without suppressing the shared web
   React updates (Codex R11 on #2411; `expo install --check` rejects React 19.2 on
   SDK 54, per plans/archive/PR-Mobile-NPM-Security-Patches.md).
2. Drop root `/` from the `pip` entry's `directories` and keep a version-update
   `ignore` for the ML/CUDA set (`torch`, `torchaudio`, `torchvision`,
   `transformers`, `accelerate`, `bitsandbytes`, `sentence-transformers`,
   `datasets`, `cuda-toolkit`, `cuda-pathfinder`, `nvidia-*`) on the remaining
   sub-directories. Root `/` is excluded because its `requirements.txt` is bound to
   the GENERATED `constraints.root-asr.txt` via a sha256 pin that
   `compile_root_asr_constraints.py --check` (in `.github/workflows/python_constraints_checks.yml`)
   re-validates; Dependabot cannot re-run the compiler, so any root edit fails that
   check (Codex R12 on #2411). Root Python deps are maintained via the compiler,
   not Dependabot.
3. Add a CI-enrolled regression test (`tests/test_security_policy_docs.py`, already
   run by `.github/workflows/atlas_security_policy_docs_checks.yml` on
   `.github/dependabot.yml` changes) asserting the three policy invariants:
   atlas-mobile is its own npm entry (not grouped with web), the mobile entry's
   ignore covers every Expo-SDK-coupled dep in `atlas-mobile/package.json`, and root
   `/` is excluded from pip. This ends the recurring "missed a sibling package"
   class (Codex R2/R13 on #2411) by failing until a new mobile dep is classified.

### Review Contract

- Acceptance criteria:
  - The file is valid Dependabot v2 config -- settled by `python3 -c "import yaml;
    yaml.safe_load(open('.github/dependabot.yml'))"` (exit 0) and GitHub's own
    dependabot config validation on push.
  - The atlas-mobile npm entry freezes the WHOLE mobile stack and no web-UI update
    is suppressed -- settled by the regression test: every dependency in
    `atlas-mobile/package.json` is matched by an ignore pattern on the atlas-mobile
    entry (react-native / @react-native/* / expo / react / react-dom / @types/react /
    typescript / tailwindcss / zustand, etc.), while `react`/`react-dom`/`@types/react`
    and the toolchain still update via the SEPARATE web-UI npm entry (which carries
    no ignores). Isolating mobile into its own entry is what lets its Expo-owned
    React be frozen without touching the web packages.
  - The pip ignore block covers exactly the version-locked ML/CUDA set implicated
    in the resolution conflict / import break -- settled by the #2404
    `ResolutionImpossible` log naming `torch==2.13.0` + `cuda-toolkit==13.3.1`,
    and #2350's torch import-chain regression.
  - Security updates are preserved -- each ignore rule uses `update-types:
    [version-update:semver-major|minor|patch]`, which scopes the ignore to
    version updates only and does not suppress Dependabot security updates.
- Reachability proof: entrypoint is the Dependabot config parser reading
  `.github/dependabot.yml` from the default branch after merge; observable effect
  is that a subsequent `@dependabot recreate` of #2397 rebuilds it as a web-UI-only
  group (mergeable), and future weekly groups exclude the frozen subsystems. #2404
  is closed (its root updates cannot be Dependabot-merged, per the ASR lock). No
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

The npm mobile-freeze ignore set and the pip root-exclusion are decision-driving
member sets. Closure declaration (R13):

- **atlas-mobile freeze set: CLOSED (whole stack).** atlas-mobile is frozen on
  Expo SDK 54 and not under active development, so its ENTIRE dependency set is
  version-frozen -- React Native / Expo / React are SDK-coupled, and typescript /
  tailwindcss are Expo-migration-coupled too (TS 6 rejected by Expo; Tailwind v4
  needs a coordinated NativeWind migration, per the archived plans). Canonical
  source of truth = the ignore patterns on the atlas-mobile npm entry, cross-checked
  against `atlas-mobile/package.json` by `tests/test_security_policy_docs.py::
  test_atlas_mobile_freezes_full_expo_sdk_stack`; `ATLAS_MOBILE_NON_SDK_DEPS` is
  EMPTY. Every atlas-mobile dependency must be matched by an ignore pattern; a new
  unlisted dependency FAILS the test until added to the freeze -- so nothing can
  silently rejoin the update stream (the second-side guard for the `@react-native/*`
  and typescript/tailwindcss omissions Codex found). Security updates still surface
  (the ignores are version-update-scoped).
- **pip root exclusion: CLOSED.** Root `/` is removed from the pip `directories`;
  `test_root_excluded_from_pip_updates` fails if any pip entry re-adds `/`. The
  ML/CUDA ignore set is a defense-in-depth version-lock guard for the remaining
  sub-dirs, not the primary boundary.

No runtime guard/validator/resolver/router/admission boundary is changed.

### Deployed-config probing

N/A - no guard/config boundary change. `.github/dependabot.yml` is CI-tooling
configuration consumed by Dependabot, not a runtime env/config fallback read by a
guard or admission path.

### Files touched

- `.github/dependabot.yml`
- `.github/workflows/atlas_security_policy_docs_checks.yml`
- `plans/PR-Dependabot-Degroup-Frozen-Subsystems.md`
- `tests/test_security_policy_docs.py`

## Mechanism

Dependabot reads the `updates` list from the config on the default branch.

- **npm:** atlas-mobile becomes its own `updates` entry whose `ignore` (each rule
  scoped to `version-update:semver-{major,minor,patch}`) freezes its ENTIRE stack --
  React Native / Expo / React and the Expo-migration-coupled tooling (typescript,
  tailwindcss, zustand). `react`/`react-dom`/`@types/react` and the toolchain still
  update through the SEPARATE web-UI entry (no ignores). The recreated
  `npm-security-and-patches` group is therefore the web-UI packages only, and is
  mergeable; the mobile entry proposes no version updates (only security updates,
  which the `version-update:*` scoping leaves untouched).
- **pip:** dropping root `/` from `directories` stops Dependabot touching the
  generated `constraints.root-asr.txt`/`requirements.txt` surface (which it cannot
  recompile), so it no longer opens root PRs that fail the constraints check. The
  remaining sub-directories have no ASR binding and are Dependabot-safe; the
  version-update `ignore` for the ML/CUDA set still guards those sub-dirs (e.g.
  `atlas_video-processing/requirements.txt` pins `torch`) from the resolution
  conflict seen in #2404.

`update-types` scoping means genuine Dependabot **security** updates are not
suppressed for any ignored dependency.

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

- Recreating #2397 (npm) after this lands (via `@dependabot recreate`): the web-UI
  entry rebuilds without atlas-mobile and is mergeable. Tracked in the same arc.
- #2404 (pip) is NOT recreated -- with root `/` excluded, its root updates cannot
  be a mergeable Dependabot PR (the generated ASR lock). It is closed, and a
  follow-up issue tracks a proper root-Python-security-update path (a scheduled
  `compile_root_asr_constraints.py` run that picks up CVEs, or a Dependabot-
  compatible regeneration step).
- A deliberate atlas-mobile Expo-SDK upgrade and a deliberate torch/CUDA-set bump
  remain future arcs, out of scope here.

Parked hardening: none.

## Verification

- `python3 -c "import yaml; yaml.safe_load(open('.github/dependabot.yml'))"` exits 0
  (2 npm entries: web with 0 ignores, mobile with 11; pip with root excluded, 11).
- `python3 -m unittest tests.test_security_policy_docs.DependabotFrozenSubsystemPolicyTest`
  passes (3 tests), and negative controls confirm it fails when atlas-mobile is
  regrouped, an SDK dep is dropped from the freeze, or root pip is re-enabled.
- After merge: `@dependabot recreate` on **#2397** only; confirm the recreated npm
  PR is web-UI packages only (no `atlas-mobile/*`). **#2404 is NOT recreated** -- it
  is closed (root updates cannot be Dependabot-merged, per the ASR lock), with a
  follow-up issue for a proper root-Python-security path.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/dependabot.yml` | 111 |
| `.github/workflows/atlas_security_policy_docs_checks.yml` | 2 |
| `plans/PR-Dependabot-Degroup-Frozen-Subsystems.md` | 227 |
| `tests/test_security_policy_docs.py` | 161 |
| **Total** | **501** |
