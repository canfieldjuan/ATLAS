# PR-Atlas-UI-ESLint-10

## Why this slice exists

Dependabot opened this slice to move the Atlas UI lint toolchain from `eslint` 9 to `eslint` 10. The compatibility risk is the lint plugin peer graph: `eslint-plugin-react-hooks` and `typescript-eslint` must advertise ESLint 10 support before the major bump can safely land.

The diff is over the normal 400 LOC budget because npm rewrites the package-lock
dependency graph for the ESLint major; the behavioral surface remains limited to
the `atlas-ui` lint/build toolchain.

## Scope (this PR)

Ownership lane: frontend/atlas-ui-tooling
Slice phase: Production hardening

1. Update `atlas-ui` from `eslint` 9 to `eslint` 10.
2. Keep the React Hooks and TypeScript ESLint plugin versions on releases whose
   peer ranges include ESLint 10.
3. Add the narrow ESLint config compatibility shim needed for the upgraded
   React Hooks plugin.
4. Fix two pre-existing `any` lint violations so the upgraded lint command is
   green without weakening TypeScript lint globally.

### Files touched

- `atlas-ui/eslint.config.js`
- `atlas-ui/package-lock.json`
- `atlas-ui/package.json`
- `atlas-ui/src/App.tsx`
- `atlas-ui/src/types/index.ts`
- `plans/PR-Atlas-UI-ESLint-10.md`

### Review Contract

Acceptance criteria:

- [ ] `eslint` is updated to `^10.5.0` for `atlas-ui`.
- [ ] React Hooks and TypeScript ESLint lint plugins remain on versions whose peer ranges include ESLint 10.
- [ ] The lockfile reflects the package manifest without invalid peer dependencies.
- [ ] `npm run lint` and `npm run build` pass in `atlas-ui`.
- [ ] Source edits are limited to lint-only typing/style fixes with no route,
      workflow, or product behavior changes.

Affected surfaces: Atlas UI lint/build tooling.

Risk areas: ESLint major-version behavior changes and peer dependency compatibility.

Reviewer rules triggered: R1, R2, R9, R12.

## Mechanism

The package manifest and lockfile update `eslint` to `^10.5.0`. The branch keeps `eslint-plugin-react-hooks` on `^7.1.1` and `typescript-eslint` on `^8.61.1`; the lockfile records those versions with ESLint 10-compatible peer ranges, so the lint stack is not left in the unsupported peer state called out by the automated review.

`eslint-plugin-react-hooks@7.1.1` enables React Compiler-style rules in its flat
recommended config. This PR disables the compiler diagnostics that flag existing
render-time randomness, ref reads, and effect-loading patterns, preserving the
pre-upgrade app behavior while still running the core hooks and TypeScript lint
rules. Two `any` violations are fixed locally with `CSSProperties` and
`unknown`.

The branch has been refreshed onto current `origin/main`; stale Security Guardrails
workflow drift from the older branch base is intentionally not part of this PR.

## Intentional

- Dependency slice for `atlas-ui`; source edits are lint unlocks only.
- No route, test, or product behavior changes.
- Keep React Hooks and TypeScript ESLint plugin versions compatible with the ESLint major bump instead of waiving the peer warning.
- Leave the package `engines.node` range unchanged; CI uses Node 20.20.0, which
  satisfies ESLint 10's Node engine.

## Deferred

- React Compiler rule adoption for the existing Atlas UI render/effect patterns
  remains deferred to a dedicated app-code cleanup slice.
- Parked hardening: none.

## Verification

- `NPM Package Checks` passed on the original Dependabot head.
- The resolved automated review finding is addressed by keeping `eslint-plugin-react-hooks@7.1.1` and `typescript-eslint@8.61.1`, which support ESLint 10 in the lockfile peer graph.
- Verified the lockfile records `eslint-plugin-react-hooks@7.1.1` peer support
  through `^10.0.0` and `typescript-eslint@8.61.1` peer support through
  `^10.0.0`.
- Verified CI uses Node 20.20.0 for NPM package checks, satisfying ESLint 10's
  Node engine.
- Merged current `origin/main` and resolved stale Security Guardrails workflow
  drift by keeping `origin/main`.
- `npm --prefix atlas-ui ci`
- `npm --prefix atlas-ui run lint`
- `npm --prefix atlas-ui run build`
- PR body AI reconciliation uses the local-audit marker `All findings fixed or waived: yes`.
- Ownership lane is scoped to `frontend/atlas-ui-tooling` so it does not collide with the open Atlas Churn UI ESLint slice.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-ui/eslint.config.js` | 6 |
| `atlas-ui/package-lock.json` | 440 |
| `atlas-ui/package.json` | 2 |
| `atlas-ui/src/App.tsx` | 6 |
| `atlas-ui/src/types/index.ts` | 2 |
| `plans/PR-Atlas-UI-ESLint-10.md` | 105 |
| **Total** | **561** |
