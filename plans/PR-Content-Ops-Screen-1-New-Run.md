# PR: Content Ops Screen 1 — New Run / Control Surface (v0)

## Why this slice exists

PR #403 / #404 / #405 landed the API adapter, contract test
harness, and domain layer for Content Ops. The v0 frontend
needs a user-visible entry point: pick a preset, pick outputs,
provide inputs, see the preview verdict. Without this screen,
the backend's `/content-ops/*` surface has no UI to drive it.

This PR is the first of three screens defined in the contract
doc:

> 1. **New Run / Control Surface**
>    - Loads `GET /content-ops/control-surfaces`.
>    - Renders preset picker, output picker, required-input
>      form, options.
>    - On submit: `POST /content-ops/preview`, render
>      `ControlSurfacePreview`.

A v0 ships first; per-output dynamic input form, sticky options
panel, and screen polish are follow-up slices.

## Scope (this PR)

A new page at `/content-ops/new` that:

1. **Loads the catalog** via `fetchContentOpsControlSurfaces` →
   `fromWireCatalog` → `ContentOpsCatalog` domain type.
2. **Renders a preset picker** as a radio group (5 options;
   selecting one auto-fills the output set).
3. **Renders an output multi-select** as checkboxes (6 outputs;
   deselects/auto-pops as the user customizes).
4. **Renders an "inputs" JSON textarea** for v0 -- the user
   types `{"target_account": "Acme", "offer": "Audit"}` directly.
   The dynamic per-output input form ships in a follow-up.
5. **Renders an options panel** with `limit`, `maxCostUsd`,
   `requireQualityGates`, `allowUnimplementedOutputs`,
   `ingestionProfile`.
6. **Submit** posts to `/content-ops/preview` via
   `previewContentOpsRun` → `fromWirePreview` → renders the
   verdict (canRun, missing_inputs, blocked_outputs, warnings,
   estimated_cost_usd, normalized_request).

### Files touched

1. `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` (new) -- the
   v0 page component. ~280 LOC.
2. `atlas-intel-ui/src/App.tsx` -- register the route at
   `/content-ops/new` and the lazy import. ~3 LOC.
3. `plans/PR-Content-Ops-Screen-1-New-Run.md` (this file).

### What's NOT in this slice

- **Dynamic per-output required-input form.** v0 uses a JSON
  textarea so the user can type inputs in raw form. The
  per-output dynamic form lands in a follow-up slice once we
  see the screen in use.
- **Plan / Execute screens.** Those are screens 2 and 3 of the
  contract; their own PRs.
- **Per-input client-side validation** (length caps, etc.).
  Screen-1's validation comes from the backend via
  `missing_inputs` / `warnings`; v0 trusts the backend.
- **Persisted run state.** No `useReducer` / context / route
  state preservation -- just local `useState`. Persistence
  ships when needed.
- **Selectors in domain layer.** Computed-state derivations
  like `canSubmit(catalog, request)` live inline in this
  screen for now. Once a second consumer needs them, they
  graduate into the domain layer.
- **Loading / error skeleton components.** v0 uses inline
  conditional rendering; existing pages (`Brands.tsx`) use
  the same pattern.

## Mechanism

Standard React patterns matching the existing repo:

- `useApiData` hook from `src/hooks/useApiData.ts` for catalog
  loading.
- Local `useState<ContentOpsRequest>` for the form state, seeded
  with sane defaults from `fromWireRequest({})`.
- Local `useState<ControlSurfacePreview | null>` for the preview
  response.
- Submit handler: build a domain request, call
  `toWireRequest(domain) → previewContentOpsRun(wire)`, map the
  response with `fromWirePreview`, store in state.

The preset → outputs auto-fill is one-way: selecting a preset
clones the preset's outputs into the form. Subsequently
unchecking an output drops it from the form's outputs without
clearing the preset selection (the contract notes the backend
warns about this and falls back to explicit outputs).

The JSON textarea uses a try/catch parse on submit: parse
errors surface as a toast-style inline error; valid parses go
through. No live syntax highlighting -- v0.

## Intentional

- **JSON textarea instead of dynamic per-output inputs form.**
  The dynamic form is the most complex part of the screen and
  has multiple design questions (which inputs are shared across
  selected outputs? per-output sub-sections? a flat union with
  source labels?). Shipping a JSON textarea unblocks the rest
  of the screen and lets us see the screen in use before
  committing to a shape. Plan-doc explicitly defers.
- **Outputs / preset interaction is one-way.** Picking a preset
  fills outputs; tweaking outputs doesn't clear the preset
  display. Keeps the UI predictable; matches the backend's
  behavior of "explicit outputs override preset."
- **No URL state for the form.** v0 doesn't survive page
  reload; that's a follow-up. The contract doc doesn't promise
  URL-state preservation.
- **No Tailwind-component extraction.** Every existing page
  inlines Tailwind classes; a `<RadioCard>` extraction is its
  own slice once we have ≥3 callers.
- **`Promise<void>` form submit.** No `useTransition` /
  optimistic state -- v0 keeps the UI honest about loading.

## Deferred

- Dynamic per-output required-input form (follow-up).
- Inline per-input validation (the backend's `missing_inputs`
  is the source of truth in v0).
- Sticky options panel / save-as-template (out of v0 scope).
- Screen 2 (Plan Preview) and Screen 3 (Execute / Run Result)
  -- separate PRs.
- URL-state preservation of the form across reloads.
- Component extraction (`<RadioCard>`, `<OutputChip>`,
  `<PreviewVerdictPanel>`) once ≥3 callers exist.

## Verification

- `cd atlas-intel-ui && npx tsc -b --noEmit` -- clean.
- `cd atlas-intel-ui && npx eslint src/pages/ContentOpsNewRun.tsx
  src/App.tsx` -- clean.
- `cd atlas-intel-ui && npm run build` -- builds.
- Manual: `npm run dev` and exercise the page -- pick a preset,
  select outputs, type inputs, submit, see preview verdict
  render.

## Estimated diff size

- `ContentOpsNewRun.tsx`: ~280 LOC.
- `App.tsx`: ~3 LOC.
- Plan doc: ~150 LOC.

Total: ~435 LOC. Marginally over the 400 soft cap. Splitting
the screen into "page skeleton" vs "form" leaves an unusable
half-screen; the screen is a structurally indivisible vertical
slice.
