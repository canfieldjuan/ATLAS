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
   v0 page component. **~497 LOC** (initial estimate ~280 LOC
   undershot; the form + verdict panel + race-condition guard
   came in heavier than projected).
2. `atlas-intel-ui/src/App.tsx` -- register the route at
   `/content-ops/new` and the lazy import. ~3 LOC.
3. `atlas-intel-ui/src/api/contentOps.ts` -- update `BASE` to
   `${API_BASE}/api/v1/content-ops` so the dev Vite proxy
   handles the request and the mount aligns with the existing
   `client.ts` / `b2bClient.ts` convention. ~5 LOC delta.
4. `atlas_brain/api/__init__.py` -- mount the
   `extracted_content_pipeline` content-ops router into the
   host's aggregate `api_router`, gated behind the
   `require_b2b_plan("b2b_growth")` dependency (same auth gate
   as `b2b_campaigns_router`; the frontend's `ProtectedRoute`
   is UI-only and does not protect the API surface). The
   `extracted_content_pipeline` import lives inside the
   `try/except` block so the host doesn't crash startup when
   the package is absent (the lazy-import + Dockerfile copy
   pattern; see item 5). The router is mounted with no
   `execution_services_provider` for v0 -- preview / plan /
   GET control-surfaces work; execute correctly returns 503
   until execution services are wired in a follow-up slice.
   ~32 LOC delta (added across Codex P1 review rounds 4, 7,
   and 8).
5. `Dockerfile` -- copy the four extracted packages
   (`extracted_content_pipeline`, `extracted_quality_gate`,
   `extracted_reasoning_core`, `extracted`) into the prod
   image so the host's lazy-import in
   `atlas_brain/api/__init__.py` resolves at runtime. Added
   in fix-up after Codex P1 round 9. Without this the route
   ships disabled in prod (lazy-import catches
   `ModuleNotFoundError` and logs the warning). ~12 LOC delta.
6. `plans/PR-Content-Ops-Screen-1-New-Run.md` (this file).

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
- Single `SubmitState` discriminated union state for the
  preview lifecycle:
  ```ts
  type SubmitState =
    | { kind: 'idle' }
    | { kind: 'submitting' }
    | { kind: 'invalid_inputs_json'; message: string }
    | { kind: 'invalid_max_cost'; message: string }
    | { kind: 'error'; message: string }
    | { kind: 'success'; preview: ControlSurfacePreview }
  ```
  Co-locates loading / parse-error / max-cost-error / API-error
  / success in one variable; the verdict panel switches on
  `kind`.
- `submitRequestIdRef` + `markStale()` race-condition guard:
  any form mutation bumps the request id; `handleSubmit`
  captures the id at submit time and drops the response if a
  newer mutation has happened. Mirrors the pattern in
  `src/hooks/useApiData.ts`.
- Submit handler: build a domain request, call
  `toWireRequest(domain) → previewContentOpsRun(wire)`, map the
  response with `fromWirePreview`, store in state -- but only
  if the captured request id still matches.

The preset → outputs interaction is bidirectional: selecting a
preset clones the preset's outputs into the form (visual hint
that the preset will resolve to those outputs). Toggling any
output then clears the preset (`toggleOutput` sets `preset:
null`) so the form moves into "explicit outputs" mode -- the
backend's `preview_control_surface` flags the dual-set state
with a "Preset ignored because explicit outputs were provided"
warning otherwise. At submit time, when a preset is still
active (user never toggled), `outputs` is sent as `[]` so the
backend resolves the preset itself; that mirrors the contract's
preset XOR outputs semantic.

The JSON textarea uses a try/catch parse on submit: parse
errors surface as a toast-style inline error; valid parses go
through. No live syntax highlighting -- v0.

`max_cost_usd` is parsed at submit time, not on every keystroke,
so the user can type sub-dollar values like `0.50` without the
input clobbering itself when the leading `0` is typed. The
submit handler:
- Blank input -> `null` (no cap; matches the backend's optional
  field semantic).
- Non-empty unparseable / non-positive (`$10`, `0`, negative,
  `NaN`) -> the verdict panel renders an inline
  `invalid_max_cost` error and submission is blocked.
  Silent-drop on a spend cap is unsafe: the user thinks they
  capped, the backend would otherwise interpret it as no cap.

The API mount aligns with the existing repo convention: the
adapter targets `${API_BASE}/api/v1/content-ops`, matching
`client.ts` (`/api/v1/consumer/dashboard`) and `b2bClient.ts`
(`/api/v1/b2b/tenant`). The Vite dev proxy handles `/api/*`
unchanged.

## Intentional

- **JSON textarea instead of dynamic per-output inputs form.**
  The dynamic form is the most complex part of the screen and
  has multiple design questions (which inputs are shared across
  selected outputs? per-output sub-sections? a flat union with
  source labels?). Shipping a JSON textarea unblocks the rest
  of the screen and lets us see the screen in use before
  committing to a shape. Plan-doc explicitly defers.
- **Outputs / preset interaction enforces preset XOR outputs.**
  Picking a preset auto-fills the form's outputs as a visual
  hint. The moment the user toggles any output, the preset is
  cleared (the form moves into explicit-outputs mode). Submit
  sends `outputs=[]` when the preset is still active so the
  backend resolves the preset itself; sending both would
  trigger the backend's "Preset ignored because explicit
  outputs were provided" warning.
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
- **Wiring `execution_services_provider` into the host mount.**
  v0 mounts the content-ops router with no provider, so
  `/execute` returns 503 by design. A follow-up slice wires
  the host's existing `IntelligenceRepository` /
  `CampaignRepository` / `BlogPostRepository` / etc. (already
  present in the host for `/api/v1/b2b/*` routes) into a
  `ContentOpsExecutionServices` factory and passes it through.
  Preview and plan work without it.

## Verification

- `cd atlas-intel-ui && npx tsc -b --noEmit` -- clean.
- `cd atlas-intel-ui && npx eslint src/pages/ContentOpsNewRun.tsx
  src/App.tsx` -- clean.
- `cd atlas-intel-ui && npm run build` -- builds.
- `bash scripts/check_ascii_python.sh` -- clean (the script is
  scoped to `extracted_content_pipeline`; this slice doesn't
  touch that package, but running for completeness per AGENTS.md
  §3b).
- `python -c "open('atlas_brain/api/__init__.py').read().encode('ascii')"`
  -- clean (the host change `atlas_brain/api/__init__.py` is
  outside the package script's scope; verified ASCII-clean
  separately).
- Manual: `npm run dev` and exercise the page -- pick a preset,
  select outputs, type inputs, submit, see preview verdict
  render.

## Estimated diff size

Initial estimate undershot the page component by ~75% -- the
form + verdict panel + race-condition guard + `SubmitState`
discriminated union added more than the rough mental model
projected. Updated for transparency.

- `ContentOpsNewRun.tsx`: ~529 LOC actual (initial estimate
  ~280 LOC; grew further across the Codex review rounds for the
  `SubmitState` race guard, `markStale`, max-cost string draft,
  `invalid_max_cost` validation state, preset / outputs XOR
  enforcement).
- `App.tsx`: ~3 LOC.
- `api/contentOps.ts`: ~5 LOC delta (BASE path realignment,
  added in fix-up commit after the Codex P1 review).
- `atlas_brain/api/__init__.py`: ~32 LOC delta (mount the
  content-ops router into the host's aggregate `api_router`
  with `require_b2b_plan("b2b_growth")` auth gate and lazy
  import; added across Codex P1 review rounds 4, 7, and 8).
- `Dockerfile`: ~12 LOC delta (copy the four extracted packages
  into the prod image; added after Codex P1 round 9).
- Plan doc: ~245 LOC actual (post-update).

Total actual: **~826 LOC**. Over the 400 soft cap. The screen
is a structurally indivisible vertical slice -- splitting at
"page skeleton" vs "form" leaves an unusable half-screen and
the race-condition / max-cost / state-machine logic depends on
the full form being present.
