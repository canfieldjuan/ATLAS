# Content Ops Reasoning UI Parity

## Why this slice exists

Backend slices now expose two operator-facing reasoning signals:

1. Catalog-level `reasoning.source` from PR #472 (`db`, `file`, or `none`).
2. Step-level `reasoning.consumed_contexts` from the execution parity work.

The Atlas Intel UI still only typed/rendered `reasoning.configured` and the
compact `contexts_used` count, so operators could not distinguish DB-backed
reasoning from file/no provider and could not inspect what context a generated
asset actually consumed.

## Scope (this PR)

1. Update Content Ops wire/domain types and mappers for `reasoning.source` and
   `reasoning.consumed_contexts`.
2. Refresh the committed fixtures so compile-time contract checks cover both
   fields.
3. Render provider source in the output picker badges.
4. Render consumed reasoning context summaries in execution result cards.
5. Clean up the merged #472 coordination row and claim this slice.

### Files touched

- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/domain/contentOps/types.ts`
- `atlas-intel-ui/src/domain/contentOps/fromWire.ts`
- `atlas-intel-ui/src/domain/contentOps/index.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json`
- `atlas-intel-ui/src/api/__fixtures__/contentOps/execution-completed.json`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Reasoning-UI-Parity.md`

## Mechanism

The API wire type keeps `source` optional so older hosts remain compatible:

```ts
reasoning: { configured: boolean; source?: 'db' | 'file' | 'none' | string }
```

Step reasoning audits gain `consumed_contexts?: CampaignReasoningContextView[]`.
The mapper copies the array defensively when present. The page renders source
as `reasoning ready (db)` / `reasoning ready (file)` and shows a compact list
of consumed context summaries under the existing reasoning badge.

## Intentional

- No new backend behavior. This is UI/domain parity only.
- The consumed-context renderer stays compact and read-only. It surfaces
  summary/proof-point counts without adding a drawer or detailed inspection
  workflow.
- `source` stays optional to preserve compatibility with older catalog
  responses.

## Deferred

- A full context drawer remains deferred. This PR makes the execution evidence
  visible enough for operator validation without adding a larger UI surface.
- No screenshot automation; this page already relies on TypeScript contract
  checks rather than browser tests.

## Verification

- `npm ci` in `atlas-intel-ui` -> installed dependencies; npm reported 6
  pre-existing audit findings (2 moderate, 4 high)
- `npm run build` in `atlas-intel-ui` -> passed
- `git diff --check` -> passed

## Estimated diff size

9 files, about +250 / -5 including this plan.
