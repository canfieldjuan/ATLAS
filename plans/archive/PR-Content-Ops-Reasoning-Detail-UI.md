# PR: Content Ops Reasoning Detail UI

## Why this slice exists

AI Content Ops execution responses already carry bounded
`reasoning.consumed_contexts` payloads. The current Atlas Intel page only shows
a compact summary for each context, so operators can see that reasoning was
used but cannot inspect the theses, proof points, timing, coverage limits, or
reference ids without opening the raw JSON result.

This PR closes the deferred "full reasoning context drawer/detail UX" item with
the smallest frontend-only surface: expandable detail panels inside the
existing execution step reasoning badge.

## Scope

1. Replace the compact-only consumed-context list in
   `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` with expandable context
   detail cards.
2. Render the existing consumed-context fields already defined by the frontend
   domain contract; do not add a backend shape.
3. Keep the first three context cap and "+N more" behavior so the execution
   panel stays bounded.
4. Update coordination and plan docs for this slice.

### Files touched

- `plans/PR-Content-Ops-Reasoning-Detail-UI.md`
- `docs/extraction/coordination/inflight.md`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`

## Mechanism

`ReasoningContextList` keeps its existing input contract and still renders at
most three consumed contexts. Each rendered context becomes a `details` element
with the current summary/count header as the collapsed state. Opening the panel
shows bounded sections for top theses, proof points, account signals, timing
windows, reference ids, coverage limits, and optional scope/delta metadata.

The implementation uses local formatting helpers in the same page component so
the UI can display unknown object shapes without introducing a new dependency or
changing domain types.

## Intentional

- No backend change. The current `reasoning.consumed_contexts` payload is
  sufficient for this UI pass.
- No modal or global drawer state. Inline expandable panels fit the current
  execution card layout and keep the change surgical.
- No component test is added because the Atlas Intel UI currently has no React
  test runner or nearby component-test pattern. Verification uses the existing
  TypeScript/Vite build.
- This branch is independent of the intervention-provider PR. If that PR lands
  first, this slice may need a coordination-ledger rebase only.

## Deferred

- A full-screen drawer with search/filtering across all consumed contexts.
- Rich semantic rendering for every possible context sub-shape.
- Frontend component tests once the UI has an established React test harness.

## Verification

- Atlas Intel production build passes.
- Focused ESLint check passes for the touched page.
- Diff whitespace check passes.
- Local PR review passes before push.

## Estimated diff size

| Area | Estimated LOC |
| --- | ---: |
| Frontend detail UI | ~260 |
| Plan and coordination docs | ~90 |
| **Total** | **~350** |
