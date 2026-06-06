# PR: Content Ops domain layer (camelCase types + fromWire mappers)

## Why this slice exists

PR #403 landed the API adapter with snake_case wire types
(matching `client.ts` / `b2bClient.ts` convention). PR #404
locked the wire types against backend drift. Screens 1-3 from
the contract need camelCase types (TS-idiomatic) and a
`ContentOpsRun` aggregate to model a complete run lifecycle.
The contract doc explicitly defines this layer:

> ### Domain layer
> Owns the typed models above. No HTTP, no React.

Without it, screens would either repeat the snake_case wire
shape (uncomfortable in JSX) or do ad-hoc translation in each
component (drift risk).

## Scope (this PR)

The domain layer for Content Ops:

1. **camelCase domain types** mirroring the API adapter's
   snake_case wire interfaces. One TS file with all the
   types -- they're small and tightly coupled.
2. **`fromWire*` mappers** that translate API adapter responses
   into domain types. One mapper per top-level wire interface.
3. **`ContentOpsRun` aggregate** type modeling a complete run
   lifecycle (catalog snapshot → request → preview → plan →
   optional execution).
4. **Barrel export** so screens import from a single
   `src/domain/contentOps` path.
5. **Contract-test extension** that pins each domain type to
   a `fromWire*(fixture)` invocation via the same `Loosen<T>` +
   `satisfies` pattern used in PR #404.

### Files touched

1. `atlas-intel-ui/src/domain/contentOps/types.ts` (new) -- all
   camelCase domain types in one file. ~180 LOC.
2. `atlas-intel-ui/src/domain/contentOps/fromWire.ts` (new) --
   one mapper per wire interface. ~140 LOC.
3. `atlas-intel-ui/src/domain/contentOps/index.ts` (new) -- re-
   exports types + mappers. ~15 LOC.
4. `atlas-intel-ui/src/domain/contentOps/contract.ts` (new) --
   compile-time pins from wire fixtures through `fromWire*` to
   each domain type. ~50 LOC.
5. `plans/PR-Content-Ops-Domain-Layer.md` (this file).

### What's NOT in this slice

- Selectors (`canExecuteRun(run)`, `outputsByExecution(run)`,
  `missingInputsForOutput(run, outputId)`). Selectors belong in
  the view-model layer per the contract; they ship with screen
  1's slice when it's clear which selectors are actually used.
- Reducers / state management. React's `useState` / `useReducer`
  is the screen's concern; the domain layer is pure data + pure
  mappers.
- Reasoning-context view-model (`CampaignReasoningContextView`).
  The wire shape lives under `step.result` in the execution
  response; a dedicated view-model lands when the reasoning
  drawer screen does.
- Signal extraction view types beyond the basic envelope; the
  contract calls these out as a special case that lands with
  its own screen.
- Step-result-payload domain types per output. Today's wire-level
  `result: Record<string, unknown>` is fine for screens 1-3 to
  display; per-output result shapes are a separate slice.

## Mechanism

The mappers convert snake_case wire fields to camelCase domain
fields. They're pure data transforms:

```ts
export function fromWireCatalog(
  wire: ContentOpsCatalogResponse,
): ContentOpsCatalog {
  return {
    outputs: wire.outputs.map(fromWireOutputDefinition),
    presets: wire.presets.map(fromWirePreset),
    execution: {
      configured: wire.execution.configured,
      configuredOutputs: [...wire.execution.configured_outputs],
    },
    ingestionProfiles: [...wire.ingestion_profiles],
  }
}
```

The contract-test extension (`contract.ts`) feeds each fixture
JSON through the mapper and asserts the result satisfies the
domain type via the same `Loosen<T>` + `satisfies` gate from PR
#404. If a wire field is renamed, the mapper trips at compile
time; if a domain field is added without a mapper update, the
satisfies check trips.

`ContentOpsRun` is a small aggregate type modeling the lifecycle:

```ts
export interface ContentOpsRun {
  catalog: ContentOpsCatalog       // snapshot at run start
  request: ContentOpsRequest       // user's input
  preview?: ControlSurfacePreview  // populated after preview
  plan?: GenerationPlan            // populated after plan
  execution?: ContentOpsExecutionResult  // populated after execute
}
```

Optional fields reflect the lifecycle: a run starts with catalog
+ request; preview / plan / execution accumulate.

## Intentional

- **One `types.ts`, not seven per-type files.** The contract
  suggested `contentOpsCatalog.ts`, `contentOpsRequest.ts`, etc.,
  but the existing repo uses single-file conventions
  (`client.ts`, `b2bClient.ts`). One file keeps imports tidy
  and matches house style. If the file grows past ~400 LOC in
  a follow-up slice, splitting is mechanical.
- **Mappers separate from types.** Wire-to-domain mappers are
  the boundary's discipline; keeping them in `fromWire.ts`
  makes the boundary explicit and screens never import from
  there directly (they import from `index.ts`).
- **No selectors / reducers.** Those are view-model concerns
  per the contract's layering. Skipping here keeps the slice
  tightly scoped; selectors land with their first consumer
  (screen 1).
- **`ContentOpsCatalog['execution']` uses an inline object type**,
  not a named alias. The shape (`{configured, configuredOutputs}`)
  is small enough that a name adds noise; if it grows, the
  alias is a one-line refactor.
- **Domain types use `readonly` on array fields where natural**
  (e.g. `readonly outputs: OutputDefinitionView[]`)? **No** --
  matches the existing TS style in the repo (mutable arrays).
  Pure-functional immutability isn't established; not introducing
  it in this slice.

## Deferred

- Selectors (`canExecuteRun`, `outputsByExecution`, etc.) --
  ship with screen 1's slice.
- Reducers / `useReducer` skeletons -- ship with screen 1.
- `CampaignReasoningContextView` domain type -- ships with the
  reasoning-drawer screen.
- Per-output `step.result` domain types (e.g.
  `EmailCampaignResult`, `BlogPostResult`) -- separate slice.
- Vitest runtime tests for the mappers -- adding Vitest is its
  own slice (deferred from PR #404).
- Snapshot of ContentOpsRun lifecycle transitions -- tied to
  the screen 1 reducer.

## Verification

- `cd atlas-intel-ui && npx tsc -b --noEmit` -- clean (the
  domain types + fromWire mappers + contract pins all
  type-check).
- `cd atlas-intel-ui && npx eslint src/domain/contentOps/` --
  clean.
- `cd atlas-intel-ui && npm run build` -- builds.
- Sanity: temporarily rename `outputs` to `outputs_typo` on
  `ContentOpsCatalog` (domain type); confirm `tsc -b` fails on
  the mapper return statement. Revert before commit.

## Estimated diff size

- `types.ts`: ~180 LOC.
- `fromWire.ts`: ~140 LOC.
- `index.ts`: ~15 LOC.
- `contract.ts`: ~50 LOC.
- Plan doc: ~150 LOC.

Total: ~535 LOC. Marginally over the 400 soft cap; the contract
test pin (~50 LOC) and the plan doc (~150 LOC) are documentation
overhead that justify themselves by catching drift and explaining
the boundary discipline. Splitting types + mappers into separate
PRs would leave one half useless without the other.
