# Content Ops Reasoning Capability UI Parity

## Goal

Expose the per-mode reasoning capability readiness shape through the
`/content-ops/control-surfaces` catalog and render it cleanly in the New Run
screen. PR #545 added the detailed shape to campaign operations status; this
slice makes the existing Content Ops catalog path preserve and display the same
contract instead of reducing it to scalar-only hints.

## Why this slice exists

The Content Ops UI reads `/content-ops/control-surfaces`, not
`/campaigns/operations/status`. Without this slice, hosts can expose detailed
reasoning capability readiness through campaign operations while the New Run
screen only sees a coarse configured flag or scalar hints.

## Scope

- Preserve nested `reasoning.capabilities` readiness maps in the control-surface
  reasoning status sanitizer.
- Keep legacy scalar capability lists supported for hosts that already pass
  simple strings.
- Update frontend wire/domain types and the Content Ops New Run reasoning hint
  to summarize object-shaped capabilities.
- Update the canonical catalog fixture so TypeScript contract checks cover the
  object shape.

## Mechanism

- Add a narrow sanitizer branch for object-shaped `reasoning.capabilities`.
- Keep the existing scalar-list sanitizer for legacy capability hints.
- Type `reasoning.capabilities` as either a scalar list or a per-mode status map.
- Summarize active/ready/configured capability statuses in the New Run badge.

## Intentional

- No frontend call to campaign operations status; the catalog remains the
  screen's single readiness source.
- Unknown nested capability fields are dropped by the API sanitizer.
- Legacy scalar capability arrays remain valid.

## Deferred

- Adding a new frontend request to `/campaigns/operations/status`.
- Changing campaign operations readiness logic.
- Changing execution behavior or provider wiring.

## Verification

- Focused control-surface API tests.
- Atlas Intel UI production build.
- Local PR review wrapper.

### Files Touched

| File | LOC |
|---|---:|
| `atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json` | 25 |
| `atlas-intel-ui/src/api/contentOps.ts` | 11 |
| `atlas-intel-ui/src/domain/contentOps/fromWire.ts` | 28 |
| `atlas-intel-ui/src/domain/contentOps/types.ts` | 11 |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | 23 |
| `docs/extraction/coordination/inflight.md` | 2 |
| `extracted_content_pipeline/api/control_surfaces.py` | 29 |
| `plans/PR-Content-Ops-Reasoning-Capability-UI-Parity.md` | 78 |
| `tests/test_extracted_content_control_surface_api.py` | 47 |

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Backend sanitizer | ~29 |
| Backend tests | ~47 |
| Frontend wire/domain/page and fixture | ~98 |
| Coordination | ~2 |
| Plan doc | ~78 |
| **Total** | ~253 |
