# Content Ops Reasoning Status UI Parity

## Why This Slice Exists

PR #537 widened the Content Ops control-surface `reasoning` payload so hosts
can expose bounded scalar lists such as available modes, packs, and
capabilities. The Atlas Intel adapter still only keeps `configured` and
`source`, so the UI cannot show those host-provided capability hints.

## Scope

1. Preserve `reasoning.modes`, `reasoning.packs`, and
   `reasoning.capabilities` in the Content Ops wire and domain types.
2. Map the three optional list fields through `fromWireCatalog`.
3. Render a compact hint in the New Run reasoning badge when those fields are
   present.
4. Clean the stale in-flight row from the merged reasoning status capabilities
   slice and claim this slice.

## Mechanism

The backend already sanitizes the list fields. The frontend keeps them as
display-only scalar arrays and limits the badge text to the first few values so
long host capability lists do not overwhelm the header.

## Intentional

- No backend route changes.
- No reasoning execution behavior changes.
- No new capability taxonomy or per-output capability matching.

## Deferred

- Per-output reasoning capability matching.
- Fuller reasoning-provider details drawer.
- Standardized reasoning capability vocabulary.

## Verification

- Atlas Intel UI production build.
- Git diff whitespace check.
- Local PR review wrapper.

### Files Touched

- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/domain/contentOps/types.ts`
- `atlas-intel-ui/src/domain/contentOps/fromWire.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Reasoning-Status-UI-Parity.md`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Wire and domain types | ~15 |
| Wire mapper | ~15 |
| New Run badge rendering | ~20 |
| Coordination | ~5 |
| Plan doc | ~45 |
| **Total** | ~100 |
