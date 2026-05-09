# PR: Content Ops contract test harness (static type-level)

## Why this slice exists

PR #403 landed `atlas-intel-ui/src/api/contentOps.ts` -- typed
fetch wrappers + 9 wire interfaces. Without a checked-in
verification artifact, the next four frontend slices (domain
layer, screens 1-3) build on those types with no automatic check
that they still match the backend response shape. If a backend
PR renames a field or changes an enum value, the failure mode
is "first user runs the screen and sees a runtime error."

This PR pins the TS wire types to canonical fixture JSON
representing each route's response shape. When the backend
changes a dataclass field, the fixture goes stale, the
`satisfies` assertion fails at compile time, and `npx tsc -b`
flags it. No test runner needed today -- the type system is
the gate.

## Scope (this PR)

Compile-time-only contract verification. No new dependencies.

### Files touched

1. `atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json`
2. `atlas-intel-ui/src/api/__fixtures__/contentOps/preview-can-run.json`
3. `atlas-intel-ui/src/api/__fixtures__/contentOps/preview-blocked.json`
4. `atlas-intel-ui/src/api/__fixtures__/contentOps/plan-runnable.json`
5. `atlas-intel-ui/src/api/__fixtures__/contentOps/plan-blocked.json`
6. `atlas-intel-ui/src/api/__fixtures__/contentOps/execution-completed.json`
7. `atlas-intel-ui/src/api/__fixtures__/contentOps/execution-partial.json`
8. `atlas-intel-ui/src/api/__fixtures__/contentOps/execution-failed.json`
9. `atlas-intel-ui/src/api/__fixtures__/contentOps/execution-blocked.json`

   Each fixture is a JSON file representing the canonical wire
   shape for that route response, with field values that mirror
   real backend semantics (e.g. catalog has all 6 outputs and 5
   presets; preview-blocked has at least one `missing_inputs`
   entry; execution-partial has both succeeded and failed steps).

10. `atlas-intel-ui/src/api/contentOps.contract.ts`
    - Imports each fixture as a typed JSON.
    - Asserts each against the corresponding wire interface via
      `satisfies` with `as const`.
    - Defines a tiny `Exact<A, B>` type-equality helper used in
      compile-time-only `_assertSatisfies<...>` placeholders to
      catch fixture-vs-interface drift.
    - The file is type-only; nothing runs at runtime.

11. `atlas-intel-ui/tsconfig.json` -- add `"resolveJsonModule": true`
    if not already enabled, so the fixture imports type-check.

12. `plans/PR-Content-Ops-Contract-Tests.md` (this file).

## Mechanism

The fixture JSON files come from the backend's actual response
shape, not invented. Each fixture sits under
`__fixtures__/contentOps/` so the location signals "test data,
not runtime payload."

`contentOps.contract.ts` imports each fixture and feeds it to a
`satisfies` clause:

```ts
import catalogFixture from './__fixtures__/contentOps/catalog.json'
import type { ContentOpsCatalogResponse } from './contentOps'

const _catalog: ContentOpsCatalogResponse = catalogFixture
//   ^-- fails to compile if fixture drifts from interface.
```

A complementary `Exact<A, B>` helper catches the inverse drift
(extra fields in the fixture that the interface doesn't model):

```ts
type Exact<A, B> = (<T>() => T extends A ? 1 : 2) extends
  (<T>() => T extends B ? 1 : 2)
  ? true
  : false

const _catalogExact: Exact<typeof catalogFixture, ContentOpsCatalogResponse> = true
//                                                                            ^--
//   compile error if the fixture has fields the interface doesn't define.
```

`tsc -b --noEmit` is the gate. CI (or the developer running
`npm run build`) catches drift in either direction.

The `executeContentOpsRun` discriminator union is checked via
shape-only fixtures for each outcome kind -- the *runtime* code
that maps HTTP codes onto the union is not exercised here (that's
the next slice; runtime tests need Vitest).

## Intentional

- **Static type-level only; no test runner.** Adding Vitest
  changes the dev-deps surface and is its own slice. Static
  checks via `tsc -b` cover type drift, which is 80% of the
  contract risk. Runtime tests can land on top.
- **Fixtures as JSON, not `.ts` const objects.** JSON is the
  on-the-wire shape; `.ts` const objects subtly let TypeScript
  infer narrower types than the JSON parser would produce
  (e.g. literal types instead of `string`). JSON forces the
  shape to be exactly what `JSON.parse` returns.
- **Hand-authored fixtures, not generated from Python.** The
  backend dataclass + `as_dict()` shapes are stable enough that
  a generation script is overkill for v0. If fixtures bit-rot
  (e.g. the contract changes 3+ fields), a generator script can
  land as a follow-up.
- **No fixture for `executeContentOpsRun`'s `request_invalid` /
  `validation_error` / `services_unavailable` outcomes.** Those
  are HTTP-code-driven, not body-shape-driven; their detail
  field is `unknown` or `string`, so a fixture wouldn't add type
  coverage. Runtime tests cover those.

## Deferred

- Vitest + runtime tests for `executeContentOpsRun` HTTP-code
  branching. Separate slice.
- A Python `scripts/dump_content_ops_fixtures.py` that imports
  the backend dataclasses and emits canonical JSON. Converts
  hand-authored fixtures into auto-generated ones, removing
  bit-rot risk. Useful but separable.
- Domain-layer types (camelCase) and their own contract checks.
- Snapshot of step-result shapes per output (e.g.
  `email_campaign` step result has different fields from
  `signal_extraction`). Today's fixtures cover the envelope; the
  step-result-payload coverage is a follow-up slice.

## Verification

- `cd atlas-intel-ui && npx tsc -b --noEmit` -- clean (the
  fixtures + interface assertions all type-check).
- `cd atlas-intel-ui && npx eslint src/api/contentOps.contract.ts
  src/api/__fixtures__/contentOps/` -- clean.
- `cd atlas-intel-ui && npm run build` -- builds.
- Sanity: temporarily mutate a field in `catalog.json`
  (e.g. rename `outputs` to `outputs_typo`); confirm `tsc -b`
  fails. Revert before commit.

## Estimated diff size

- 9 JSON fixtures × ~30 LOC each = ~270 LOC.
- `contentOps.contract.ts`: ~80 LOC.
- `tsconfig.json` tweak: ~1 LOC.
- Plan doc: ~140 LOC.

Total: ~490 LOC. Marginally over the 400 soft cap; the JSON
fixtures dominate (~55% of the diff) and they're the value-
bearing artifact. Reviewer can flag if the fixture-density
should split into "envelope-only" vs "per-output" slices.
