# PR-Dependabot-Major-Compatibility-Guard

## Why this slice exists

Juan asked to process the open Dependabot PRs carefully because dependency
bumps may break adjacent systems. The current queue contains nine PRs and none
is merge-ready. Five independently move TypeScript 6 to 7, two raise the NumPy
floor from 1.24 to 2.5, one moves the pinned Atlas Edge NeMo release from 2.6.2
to 3.0.0, and one stale 66-package group still edits the compiler-owned root
Python lock that current `main` has already excluded from Dependabot.

Those eight major-version PRs expose one policy defect: compatibility-coupled
runtime/toolchain majors are admitted as unattended version maintenance. The
TypeScript PRs either exceed `typescript-eslint`'s supported compiler range or
lack mandatory build proof; NumPy 2.5 drops the Python 3.11 audit baseline; and
NeMo 3 contradicts the code-owned Edge Python 3.10/3.11 pin contract. Closing
the individual PRs leaves the weekly schedule free to recreate the same class.

### Problem-derived contract

- Root cause: the web npm and leaf-service pip entries group patch/minor updates
  but leave every major version update eligible as an individual PR. Major
  compiler/runtime lines require coordinated toolchain, interpreter, platform,
  and CI evidence that Dependabot cannot supply through an isolated manifest
  edit.
- Correct fix must touch/change: constrain ordinary version maintenance in the
  two existing entries to wildcard patch/minor `allow.update-types`; extend the
  enrolled policy parser/test to prove the exact admitted levels; add this plan.
  After the guard lands, close the nine stale/incompatible PRs and let current
  `main` generate only eligible maintenance groups.
- Must not change: no package manifest, lockfile, requirements/constraints file,
  runtime code, Python/Node baseline, dependency version, UI, API, deployment,
  security-update eligibility, existing mobile/ML ignore policy, or PR outside
  the operator-assigned Dependabot queue changes in this slice.

## Scope (this PR)

Ownership lane: dev-workflow/dependabot-config
Slice phase: Workflow/process
Max files: 3

1. Admit only semver-minor and semver-patch ordinary version updates for the
   five web UI directories and five leaf-service Python directories.
2. Extend the existing policy parser/test so deleting, widening, or mis-scoping
   either wildcard allow rule fails the CI-enrolled policy test.

### Review Contract

- Acceptance criteria:
  - The web npm entry's `allow_update_types` parses exactly as wildcard
    semver-minor plus semver-patch, settled by
    `test_routine_version_policy_requires_deliberate_major_migrations`.
  - The leaf-service pip entry has the same exact wildcard policy, settled by
    the same test.
  - GitHub documents `allow.update-types` as affecting version updates, not
    security updates; the configured allow therefore preserves security-update
    eligibility across a major boundary. Settled by GitHub's official
    Dependabot options reference and controlling-updates documentation.
  - Existing groups, mobile freezes, ML/CUDA ignores, root pip exclusion, and
    every other ecosystem entry remain unchanged, settled by the focused policy
    class plus the cold diff.
  - `.github/dependabot.yml` remains valid YAML, settled by `yaml.safe_load`.
- Reachability proof: Dependabot reads `.github/dependabot.yml` from default
  `main`; after merge, its next scheduled/recreated ordinary version-update
  queue omits semver majors for these entries while security updates remain
  separately eligible. This PR adds no runtime surface.
- Affected surfaces: ordinary Dependabot version-update admission for the five
  web UI directories and five leaf-service Python directories; the existing
  security-policy test.
- Risk areas: accidentally suppressing security updates, changing patch/minor
  grouping, applying the policy to atlas-mobile or root pip, and parser false
  positives that pass while the allow rule has a wrong update type.
- Reviewer rules triggered: R1, R2, R11, R12, R13, R14.

### Boundary-change enumeration

- Boundary path/seam: ordinary Dependabot version-update admission for the web
  npm and leaf-service pip entries.
- Replaced-path behaviors: semver-major ordinary version proposals change from
  admitted to unallowed; semver-minor, semver-patch, and the separately governed
  security-update path remain eligible.
- Guard-relevant fields: `package-ecosystem`, `directories`, wildcard
  `allow.dependency-name`, and `allow.update-types`.
- Caller x input shape: Dependabot reading default-branch config x ordinary
  version updates for any dependency in either declared directory set.

Closure declaration: the admitted ordinary-version class is **CLOSED by SemVer
level**, not by a dependency-name list. Minor and patch are admitted; major is
not. The reviewable surface is the complete web npm entry and complete
leaf-service pip entry. Any future dependency follows the same default without
growing a member list. Security updates are a separate GitHub-governed path and
are not restricted by `allow.update-types`.

### Deployed-config probing

- Deployed/default config values: current default-branch entries omit `allow`,
  so the eight live major PRs demonstrate the ordinary-major admitted default.
- Explicit value probe: the focused test parses both wildcard allow rules and
  requires exactly semver-minor plus semver-patch.
- Absent value probe: deleting either allow block makes the focused test fail;
  the open major PRs are live evidence of the absent-policy behavior.
- Default-session/default-context probe: YAML validation and the test read the
  repository-default `.github/dependabot.yml`, matching Dependabot's consumer.
- Side-effect ordering: N/A; this is scheduler admission configuration and has
  no runtime mutation path.

### Files touched

- `.github/dependabot.yml`
- `plans/PR-Dependabot-Major-Compatibility-Guard.md`
- `tests/test_security_policy_docs.py`

## Mechanism

Each applicable ecosystem entry gets one wildcard `allow` rule containing only
`version-update:semver-minor` and `version-update:semver-patch`. Dependabot then
does not create unattended major version PRs for those directories. This uses
`allow`, not `ignore`: GitHub's official documentation states that
`allow.update-types` only controls version updates and security updates are
still created independently. The existing patch/minor groups and frozen-stack
ignores remain byte-for-byte unchanged.

The policy parser now records the update types attached to `allow` and `ignore`
dependency rules only when the section is a direct child of its update entry,
instead of proving only that an ignored dependency name is present. The regression
test requires the exact wildcard patch/minor set on both entries, rejecting a
missing rule, an accidentally admitted major, a policy narrowed to selected
package names, or a rule incorrectly nested under a group.

## Intentional

- All ordinary majors in the two active maintenance entries require a deliberate
  migration PR. This avoids a growing dependency-name denylist and covers future
  compilers/runtimes by the same safe default.
- Security updates are not traded away to stop version-update noise. The first
  attempted `ignore` mechanism was rejected after official documentation showed
  that ignore filters can affect security updates.
- The stale 66-package PR is not repaired by hand. Its root files are no longer
  Dependabot-owned on current `main`, so recreation from current policy is the
  truthful path.

## Deferred

- Deliberate major migrations after the affected package supplies toolchain,
  resolver, build/lint, and owned-platform runtime proof.
- Close #2443/#2430/#2429/#2426/#2425/#2421/#2419/#2416/#2413 after this guard
  is merged; permit Dependabot to recreate only the eligible current-main queue.
- The pre-existing atlas-mobile and ML/CUDA `ignore` policy is unchanged. Its
  security-update semantics are not widened into this ordinary-major admission
  slice.

Parking predicate: dependency migrations that require a new runtime/toolchain
baseline or new CI surface are separate slices rather than inline additions.

Parked hardening: none.

## Verification

- `python3 -c "import yaml; yaml.safe_load(open('.github/dependabot.yml', encoding='utf-8'))"`
  — passed.
- `python3 -m pytest tests/test_security_policy_docs.py::DependabotFrozenSubsystemPolicyTest -q`
  — `5 passed in 0.37s`.
- `python scripts/sync_pr_plan.py plans/PR-Dependabot-Major-Compatibility-Guard.md origin/main --check`
  — passed after synchronizing the generated diff-size table.
- `python scripts/audit_plan_doc.py plans/PR-Dependabot-Major-Compatibility-Guard.md`
  — passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Dependabot-Major-Compatibility-Guard.md --base-ref origin/main`
  — passed.
- `git diff --check` — passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/dependabot.yml` | 16 |
| `plans/PR-Dependabot-Major-Compatibility-Guard.md` | 178 |
| `tests/test_security_policy_docs.py` | 109 |
| **Total** | **303** |
