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

11. `atlas-intel-ui/tsconfig.app.json` -- add `"resolveJsonModule": true`
    so the fixture imports type-check.

12. `plans/PR-Content-Ops-Contract-Tests.md` (this file).

## Mechanism

The fixture JSON files come from the backend's actual response
shape, not invented. Each fixture sits under
`__fixtures__/contentOps/` so the location signals "test data,
not runtime payload."

The naive approach -- typed assignment, e.g.
`const _catalog: ContentOpsCatalogResponse = catalogFixture` --
fails because TypeScript widens JSON-imported literals to
`string`. Wire interfaces use literal-string unions
(e.g. `status: "completed" | "partial" | "failed" | "blocked"`)
that JSON's inferred `status: string` does not satisfy. A naive
`as <Type>` cast also doesn't work: it's too permissive (initial
testing showed it accepted a JSON missing `ingestion_profiles`
without complaint).

The shipped harness uses two complementary gates:

**Gate 1 -- structural drift via `Loosen<T>` + `satisfies`:**

```ts
type Loosen<T> = T extends string
  ? string
  : T extends Array<infer U>
    ? Loosen<U>[]
    : T extends object
      ? { [K in keyof T]: Loosen<T[K]> }
      : T

export const __catalogContract = catalogFixture satisfies
  Loosen<ContentOpsCatalogResponse>
```

`Loosen<T>` recursively widens literal-string unions in `T` to
`string`, so JSON's inferred `status: string` is compatible at
the structural layer. `satisfies` then enforces "no missing
required fields, no extra properties." Removing `ingestion_profiles`
from a fixture trips this gate (verified manually: produces
`TS1360 Property 'ingestion_profiles' is missing`).

**Gate 2 -- literal vocabulary via `Record<UnionType, true>`:**

```ts
const _executionStatusCoverage: Record<
  ContentOpsExecutionResult['status'],
  true
> = { completed: true, partial: true, failed: true, blocked: true }
```

`Record<UnionType, true>` requires exactly the keys in the union.
Adding a new literal to the wire union (`"queued"`) makes the
assignment fail because `queued` isn't in the object. Removing a
literal makes the existing key excess. This gate catches enum
drift in both directions.

The two gates close the contract envelope: Gate 1 covers shape,
Gate 2 covers enum vocabulary. `tsc -b --noEmit` is the runner;
CI (or `npm run build`) catches drift in either gate.

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

Initial estimate undershot the JSON fixtures; real backend output
is more verbose than the rough mental model.

- 9 JSON fixtures dumped from real backend code: ~590 LOC actual
  (initial estimate ~270 LOC; backend dataclasses serialize more
  verbosely than expected).
- `contentOps.contract.ts`: ~108 LOC actual (initial estimate ~80).
- `tsconfig.app.json` tweak: 1 LOC.
- Plan doc: ~150 LOC.

Total actual: ~852 LOC. Over the 400 soft cap, justified because
the JSON fixtures dominate (~70% of the diff) and the contract
gate is only useful end-to-end -- splitting at any layer leaves
the harness half-formed. Reviewer can flag if the fixture-density
should split into "envelope-only" vs "per-output" slices in the
follow-up that adds step-result-payload coverage.
