# PR: Content Ops Screen 2 — Plan Preview

## Why this slice exists

Screen 1 (PR #406) lets the user pick a preset / outputs / inputs
and submit to `/content-ops/preview`. The contract's screen 2
takes the next step:

> 2. **Plan Preview**
>    - Triggered when preview `canRun=true`; calls
>      `POST /content-ops/plan`.
>    - Renders `GenerationPlan.steps` with config panels.
>    - "Execute" button enabled when `plan.canExecute=true` and
>      host execution services are configured.

Without Screen 2, the user can preflight the request but can't
see the runnable plan (per-output runner names, per-step config
shapes, blocked-step reasons). Screen 3 (Execute) builds on top
of this.

## Scope (this PR)

Extends `ContentOpsNewRun.tsx` with a "Build plan" step instead
of shipping a separate route. After a preview returns
`canRun=true`, a CTA appears that triggers
`POST /content-ops/plan` and renders the plan steps panel
beneath the preview verdict. This avoids duplicating the
preset / outputs / inputs / options form.

### Files touched

1. `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` -- add
   a `planState` discriminated union, a `handlePlan()` handler,
   a `PlanPanel` component that renders
   `GenerationPlan.steps` + per-step config tables, and a
   "Build plan" CTA inside the existing `PreviewVerdict`. ~150
   LOC delta.
2. `plans/PR-Content-Ops-Screen-2-Plan-Preview.md` (this file).

### What's NOT in this slice

- **A separate `/content-ops/plan` route.** Adding a separate
  page would duplicate the preset / outputs / inputs / options
  form (which is ~250 LOC). Better to wait until 3+ pages need
  the form, then extract a shared `<RequestForm>` component.
- **Execute submit.** Screen 3's territory. The plan panel
  shows a disabled "Execute" placeholder with a tooltip
  ("Screen 3 ships next") -- not a functioning button yet.
- **Plan refresh / re-plan after edits.** Editing the form
  invalidates both preview and plan via the same
  `markStale()` plumbing already in Screen 1.
- **Per-step config shape adapters** (e.g. render
  `email_campaign`'s `channels` array as chips, render
  `signal_extraction`'s `max_text_chars` as a chart-cap badge).
  v0 renders all per-step config as a generic
  key-value JSON table -- humanizing is a follow-up slice
  per the contract doc's view-model layer.

## Mechanism

The state model gains a parallel `planState`:

```ts
type PlanState =
  | { kind: 'idle' }
  | { kind: 'submitting' }
  | { kind: 'error'; message: string }
  | { kind: 'success'; plan: GenerationPlan }
```

`handlePlan()` mirrors `handleSubmit()` (parse inputs JSON,
normalize max cost, build domain request, call
`planContentOpsRun(toWireRequest(request))`, map response with
`fromWirePlan`), but writes to `planState`. The same
`submitRequestIdRef` race guard already in place is reused to
drop stale plan responses.

The "Build plan" CTA appears inside the existing
`PreviewVerdict` only when `preview.canRun=true`. Clicking it
fires `handlePlan()`. The plan panel renders below the preview
verdict; the user sees both at once.

`markStale()` already invalidates the preview verdict on form
mutation. Extending it to also reset `planState` keeps the
two panels in sync with the form.

## Intentional

- **Single route, single form.** The contract's "MVP screens 1 /
  2 / 3" framing is sequential, not separate-page. The cleanest
  match is one wizard-style flow on `/content-ops/new`.
- **Plan panel is read-only.** No "edit step config" affordance
  in v0; the plan reflects what the backend will run, not
  what the user can tweak. Editing belongs upstream in the
  options panel.
- **Disabled "Execute" placeholder, not a missing button.**
  Communicates Screen 3's pending arrival without committing
  to a UX shape. Removed cleanly when Screen 3 lands.
- **No `useReducer`.** Two parallel states (`submitState` and
  `planState`) are simple enough that a reducer adds friction.
  Reducer ships when the lifecycle has 3+ states or screen 3
  introduces dependent transitions.
- **Per-step config rendered as JSON.** v0 readability is
  acceptable; per-output adapters are a clear follow-up slice
  per the contract's "view-model layer" section.

## Deferred

- Per-output `step.config` view adapters (humanize `channels`,
  `default_report_type`, etc.). Separate slice per the
  contract's view-model layer.
- Screen 3 (Execute / Run Result) -- next PR.
- Component extraction (`<RequestForm>`, `<StatusBadge>`,
  `<JsonTable>`) once 3+ callers exist.
- URL state / shareable plan-preview links.

## Verification

- `cd atlas-intel-ui && npx tsc -b --noEmit` -- clean.
- `cd atlas-intel-ui && npx eslint src/pages/ContentOpsNewRun.tsx`
  -- clean.
- `cd atlas-intel-ui && npm run build` -- builds.
- Manual: `npm run dev`; pick `email_only` preset, type
  `{"target_account": "Acme", "offer": "Audit"}`, submit
  Preview, click "Build plan", see plan panel render with one
  step (`email_campaign` runner + config).

## Estimated diff size

- `ContentOpsNewRun.tsx`: ~150 LOC delta (PlanState +
  handlePlan + PlanPanel + Build-plan CTA).
- Plan doc: ~140 LOC.

Total: ~290 LOC. Under the 400 LOC soft cap.
