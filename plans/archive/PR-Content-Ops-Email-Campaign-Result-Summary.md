# PR: Content Ops email campaign result summary

## Why This Slice Exists

PR #410 made the Content Ops execute panel functional, but every
step result still rendered as raw JSON. That is acceptable as a
fallback, but the first shipped execution shape already has stable,
user-facing fields for `email_campaign`: `generated` and `saved_ids`.

This slice adds the smallest useful result adapter so users can see
the common email draft outcome without opening the raw payload.

## Scope

Files touched:

1. `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
   - Add `ExecutionStepSummary`.
   - Render the summary for `output === "email_campaign"`.
   - Summarize `generated` and `saved_ids` only when those fields are
     present with the expected runtime shape.
   - Keep raw result JSON rendering for every output.

2. `docs/frontend/content_ops_frontend_contract.md`
   - Document the `email_campaign` summary rule.

3. `docs/extraction/coordination/inflight.md`
   - Claim the slice while the PR is open.

## Mechanism

`ExecutionPanel` delegates per-step result display to
`ExecutionStepSummary` before rendering the existing raw JSON block.

`ExecutionStepSummary` is intentionally narrow:

- It returns `null` for every output except `email_campaign`.
- It reads `generated` only when it is a number.
- It reads `saved_ids` only when it is an array of strings.
- It returns `null` when neither field is present.

This keeps malformed or future payload shapes from producing
misleading UI while preserving the raw payload for inspection.

## Intentional

- No new domain result types yet. This is one small view adapter for
  one stable result shape.
- No changes to API adapters, execution behavior, backend routes, or
  fixtures.
- No attempt to summarize `report`, `landing_page`, `sales_brief`, or
  `signal_extraction` in this slice.
- Raw JSON remains the universal fallback.

## Deferred

- Per-output result adapters for reports, landing pages, sales briefs,
  and signal extraction.
- Dedicated component tests for `ContentOpsNewRun`; the repo does not
  currently have a component-test harness for this page.
- A richer saved-draft link/display model once the execution payload
  carries stable URLs or user-facing draft metadata.

## Verification

- `cd atlas-intel-ui && npm ci`
- `cd atlas-intel-ui && npm run build`
- `cd atlas-intel-ui && npx eslint src/pages/ContentOpsNewRun.tsx src/api/contentOps.ts src/domain/contentOps/types.ts src/domain/contentOps/fromWire.ts src/api/contentOps.contract.ts src/domain/contentOps/contract.ts`
- `git diff --check`
- no non-ASCII added lines

## Diff Size

Expected production delta is small:

- `ContentOpsNewRun.tsx`: about 35 lines.
- Docs/coordination: about 5 lines plus this plan.

The slice stays under the soft 400-line PR budget.
