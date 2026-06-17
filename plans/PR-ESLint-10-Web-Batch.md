# PR-ESLint-10-Web-Batch

## Why this slice exists

The dependency triage lane is draining the remaining Dependabot security and
toolchain backlog in small, reviewable batches. After the TypeScript 6 and
React-hooks slices landed, four open Dependabot PRs still try to move web
frontends from ESLint 9 to ESLint 10 (#1642, #1645, #1649, #1655). A trial
batch showed only the CI-linted packages (`atlas-admin-ui` and
`atlas-intel-ui`) can prove lint compatibility now; `atlas-churn-ui` and
`atlas-ui` have pre-existing lint debt that needs its own cleanup slice.

This slice lands the safe, lint-verified ESLint 10 subset while keeping the
risky mobile TypeScript/Expo graph, the broad #1666 npm security group, and the
churn/ui lint-debt cleanup out of scope.

The diff exceeds the usual 400 LOC soft cap because npm lockfile rewrites are
indivisible: both touched packages need their lockfiles refreshed with the
manifest bump, and hand-trimming generated lockfile churn would be less safe
than letting npm own the package graph.

Review follow-up: CI caught a plan/code consistency false path claim because
the original plan described generic `package.json` / `package-lock.json` files
instead of package-qualified paths. The root cause was ambiguous plan wording,
not a missing repository file; this update fixes the wording class by avoiding
bare root-looking file names in prose.

## Scope (this PR)

Ownership lane: security/dependencies
Slice phase: Production hardening

1. Update `eslint` to the Dependabot-requested ESLint 10 line in
   `atlas-admin-ui` and `atlas-intel-ui`.
2. Refresh only those two package manifests/lockfiles unless ESLint 10 exposes
   a compatibility fix required for their existing lint/build gates.
3. Prove the existing lint/build surfaces still pass for both touched web
   packages.

### Review Contract

- Acceptance criteria:
  - [ ] `eslint` is updated consistently in `atlas-admin-ui` and
        `atlas-intel-ui`.
  - [ ] No `atlas-churn-ui`, `atlas-ui`, `atlas-mobile` package graph, Docker
        image, workflow guardrail, or broad #1666-only dependency change is
        included.
  - [ ] Existing flat ESLint config remains compatible, or any required config
        edit is minimal and explained here.
  - [ ] `atlas-admin-ui` and `atlas-intel-ui` lint/build gates pass locally
        before push.
- Affected surfaces: frontend lint toolchains and lockfiles for the two
  CI-linted web packages.
- Risk areas: dependency/toolchain backcompat, CI runtime compatibility, and
  lockfile drift with open Dependabot PRs.
- Reviewer rules triggered: R1, R2, R9, R12, R13, R14.

### Files touched

- `HARDENING.md`
- `atlas-admin-ui/package-lock.json`
- `atlas-admin-ui/package.json`
- `atlas-intel-ui/package-lock.json`
- `atlas-intel-ui/package.json`
- `plans/PR-ESLint-10-Web-Batch.md`

## Mechanism

Each touched package is updated with npm so the package manifest and lockfile
remain in sync (`atlas-admin-ui/package.json`,
`atlas-admin-ui/package-lock.json`, `atlas-intel-ui/package.json`, and
`atlas-intel-ui/package-lock.json`). The existing apps already use flat config
via `eslint/config`, `@eslint/js`, `typescript-eslint`,
`eslint-plugin-react-hooks`, and `eslint-plugin-react-refresh`; this slice does
not rewrite lint policy because the existing configs pass under ESLint 10 for
the touched packages.

## Intentional

- Keep the standalone Dependabot PRs as source material rather than merging
  them directly; the admin and intel updates are reapplied on current main.
- Do not include `atlas-churn-ui` or `atlas-ui` even though Dependabot has
  ESLint 10 PRs for them. Their `npm run lint` scripts currently fail on
  existing source debt, so including them would make this package-manager slice
  claim compatibility it cannot prove.
- Do not include `atlas-mobile`. Its Expo/React Native graph has a separate
  parked hardening entry and a different runtime verification burden.
- Do not merge or close #1666 in this PR. #1666 is broad overlap/noise and
  remains an operator decision unless explicitly reassigned.

## Deferred

- Standalone Dependabot PR cleanup for #1642 and #1649 after this batch lands.
- #1645 and #1655 remain deferred until the churn/ui lint-debt slice lands.
- #1666 cleanup remains separate; the safe React-hooks portion has already
  landed, and the remaining mobile graph belongs to the Expo hardening slice.

Parked hardening: `Mobile Expo 56 audit cleanup` remains parked because this
slice deliberately excludes `atlas-mobile`; `Churn and Atlas UI lint debt
blocks ESLint 10` is added because the trial batch exposed pre-existing
churn/ui lint failures that block their ESLint 10 PRs.

## Verification

- `cd atlas-admin-ui && npm ci && npm audit --audit-level=high && npm run lint && npm run build` -- passed.
- Discovery only: `cd atlas-churn-ui && npm ci && npm audit --audit-level=high && npm test && npm run lint && npm run build` -- `npm test` passed (85 files / 681 tests) and audit passed, then lint failed on 118 existing errors; package reverted out of scope and debt parked.
- `cd atlas-intel-ui && npm ci && npm audit --audit-level=high && npm run lint && npm run build` -- passed.
- Discovery only: `cd atlas-ui && npm ci && npm audit --audit-level=high && npm run lint && npm run build` -- audit passed, then lint failed on 24 existing errors; package reverted out of scope and debt parked. `npm run build` passed after reverting.
- Pending before push: `bash scripts/push_pr.sh tmp/pr-body-eslint-10-web-batch.md -u origin HEAD`.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 9 |
| `atlas-admin-ui/package-lock.json` | 440 |
| `atlas-admin-ui/package.json` | 2 |
| `atlas-intel-ui/package-lock.json` | 446 |
| `atlas-intel-ui/package.json` | 2 |
| `plans/PR-ESLint-10-Web-Batch.md` | 121 |
| **Total** | **1020** |
