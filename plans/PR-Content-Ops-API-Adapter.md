# PR: Content Ops API adapter — typed fetch wrappers + wire types

## Why this slice exists

PR #401 landed `docs/frontend/content_ops_frontend_contract.md` --
the load-bearing artifact that defines the AI Content Ops backend
surface for the frontend. Every type and route is cited file:line
against backend HEAD `a4020c1`. The contract's *Deferred* section
explicitly names the next slice:

> Concrete TypeScript type generation (e.g. `openapi-codegen` or
> hand-written `src/api/contentOps.ts`). Lands when the frontend
> repo scaffolds.

`atlas-intel-ui/` is already scaffolded (Vite + React 19 + TS).
Without typed fetch wrappers, screens 1-3 from the contract can't
land -- they'd hand-roll fetch + retype the same response shapes.
This PR is the foundation; the next four frontend slices stack on
it.

## Scope (this PR)

One file: `atlas-intel-ui/src/api/contentOps.ts`.

- 9 wire-shape `interface`s mirroring the backend dataclasses
  cited in `docs/frontend/content_ops_frontend_contract.md`. Field
  names stay snake_case to match the backend response (matches
  the existing `client.ts` / `b2bClient.ts` convention).
- 4 typed fetch wrappers:
  `fetchContentOpsControlSurfaces`,
  `previewContentOpsRun`,
  `planContentOpsRun`,
  `executeContentOpsRun`.
- HTTP status code mapping helper that translates 200/207/400/502
  responses into a discriminated union `ContentOpsExecuteOutcome`
  the UI can render without needing to know HTTP semantics.
- Reuse the existing fetch / auth / 401-refresh / 402-403 plumbing
  via a thin internal helper (or shared module) -- do not duplicate
  the entire `client.ts` boilerplate.

### What's NOT in this slice

- Domain layer (camelCase translation, `ContentOpsRun` aggregate,
  reducers). Belongs in `src/content/contentOps/` per the
  contract's "Domain layer" section. Next slice.
- View-model / UI / screens.
- A contract test harness pinning the TS types to canonical
  fixtures. The slice immediately after this one.

### Files touched

1. `atlas-intel-ui/src/api/contentOps.ts` (new) -- ~250 LOC.
2. `plans/PR-Content-Ops-API-Adapter.md` (this file).

## Mechanism

The four routes mount under a host-configurable prefix; the
contract uses `/content-ops` as the canonical default (set in
`ContentOpsControlSurfaceApiConfig`). The adapter targets that
prefix relative to `API_BASE`.

Wire types are 1:1 with the backend response JSON. The contract
doc shows camelCase example types in its "Frontend domain model"
section, but those camelCase types are the *domain layer* shapes
(per its "Frontend layering" section). The API adapter's job is
to mirror the wire format and let the domain layer translate. This
matches the existing `b2bClient.ts` pattern: snake_case interfaces
that mirror backend response keys directly.

The execute route is special: the backend maps the
`ContentOpsExecutionResult.status` field onto HTTP codes (200 /
207 / 400 / 502), so a simple `await res.json()` is insufficient
-- the HTTP code itself is part of the contract. The adapter
returns a discriminated `ContentOpsExecuteOutcome`:

```ts
type ContentOpsExecuteOutcome =
  | { kind: "completed"; result: ContentOpsExecutionResult }
  | { kind: "partial"; result: ContentOpsExecutionResult }
  | { kind: "failed"; result: ContentOpsExecutionResult }
  | { kind: "blocked"; result: ContentOpsExecutionResult }
  | { kind: "validation_error"; detail: unknown }       // 422
  | { kind: "services_unavailable"; detail: string }    // 503
  | { kind: "request_invalid"; detail: string };        // 400 (ValueError-shape)
```

The UI renders one banner per outcome kind. No string-matching on
HTTP codes elsewhere in the codebase.

## Intentional

- **snake_case wire types**, NOT camelCase. The existing
  `client.ts` and `b2bClient.ts` mirror the backend literal keys.
  Translation happens at the domain layer. Avoids accidentally
  introducing two conventions in `src/api/`.
- **One file, not four.** The contract suggested
  `contentOpsControlSurfaces.ts` / `contentOpsPreview.ts` /
  `contentOpsPlan.ts` / `contentOpsExecute.ts` -- that's more
  ceremony than the existing repo uses (`client.ts` and
  `b2bClient.ts` each carry their full surface). Single-file keeps
  imports tidy and matches convention. If the file grows past
  ~400 LOC in a future slice, splitting is mechanical.
- **HTTP-code-aware execute outcome** instead of raising on 207.
  The existing `client.ts` `handleResponse` throws on `!res.ok`
  -- but 207 is a partial-success signal, not an error. The
  adapter's execute wrapper handles 207 / 400 / 502 specially and
  surfaces them as outcome kinds; the rest of `client.ts`'s
  401/402/403/4xx behavior stays unchanged.
- **Discriminated union for execute outcome.** The UI banner is
  one switch on `kind`. No HTTP-code knowledge leaks into screens.
- **No domain transforms here.** Resist the temptation to compute
  `canExecute && executionConfigured` or normalize warnings here;
  those are domain-layer concerns. This slice is wire-shape only.

## Deferred

- Domain layer (`src/content/contentOps/`) -- camelCase types,
  `ContentOpsRun` aggregate, reducers/selectors. Next slice.
- Contract test harness -- fixture JSON + tests pinning the TS
  shapes to backend dataclass shapes. Should land right after the
  domain layer (or before -- either order works).
- Screen 1 / 2 / 3 implementations.
- Reasoning context view-model (`CampaignReasoningContextView`)
  belongs in the domain layer; the wire shape is already part of
  the execute response under `step.result`.

## Verification

- `cd atlas-intel-ui && npm run lint` -- clean.
- `cd atlas-intel-ui && npx tsc -b --noEmit` -- type-checks.
- `bash scripts/check_ascii_python.sh` -- N/A (no Python touched).
- Manual: import the new module from `App.tsx` to confirm no
  tooling complaints; revert the import before commit (don't ship
  unused screens; this slice is types-only).
- `git diff main --stat` -- 2 files, ~250 LOC code + ~150 LOC plan.

## Estimated diff size

- `atlas-intel-ui/src/api/contentOps.ts`: ~250 LOC.
- `plans/PR-Content-Ops-API-Adapter.md`: ~150 LOC.

Total: ~400 LOC. Right at the budget.
