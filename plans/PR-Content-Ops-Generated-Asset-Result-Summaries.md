# PR: Content Ops generated asset result summaries

## Why this slice exists

The execute panel already summarizes `email_campaign` and
`signal_extraction`, but the implemented generated-asset outputs still
fall through to raw JSON. That makes successful report, landing-page,
and sales-brief runs harder to scan even though their backend result
contracts already share the same small status shape.

This closes the next frontend-contract gap for Content Ops execution
results without changing backend behavior.

## Scope (this PR)

1. Add a shared generated-asset result summary for `report`,
   `landing_page`, and `sales_brief`.
2. Preserve the existing `email_campaign` summary behavior while
   routing it through the same display component.
3. Keep raw JSON details available for every execution step.
4. Update the frontend contract doc with the shared result shape.
5. Claim this slice in the extraction coordination table while the PR
   is open.

### Files touched

- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `docs/frontend/content_ops_frontend_contract.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Generated-Asset-Result-Summaries.md`

## Mechanism

`ExecutionStepSummary` now routes:

- `signal_extraction` to the existing signal summary.
- `email_campaign` to `GeneratedAssetSummary` with the label
  `Drafts generated`.
- `report`, `landing_page`, and `sales_brief` to
  `GeneratedAssetSummary` with the label `Assets generated`.

`GeneratedAssetSummary` defensively reads only stable primitive fields:
`requested`, `generated`, `skipped`, `saved_ids`, and `errors`. Missing
or malformed fields are ignored, and the raw JSON details remain below
the summary.

## Intentional

- No backend changes. The UI consumes the existing `as_dict()` result
  contracts.
- No new per-output custom cards for `report`, `landing_page`, or
  `sales_brief`; their result contracts are identical enough for one
  shared summary.
- No blog-post adapter in this slice. Its result shape needs a separate
  read before adding a summary.
- No component-test harness added; this page currently validates through
  the frontend build path.

## Deferred

- Blog-post execution result summary after its backend result contract
  is read and mapped.
- Rich generated-asset details, such as titles or preview snippets,
  once those fields are stable across all asset services.
- Component-level tests for `ContentOpsNewRun` if the frontend test
  harness is introduced.

## Verification

- `cd atlas-intel-ui && npm ci`
- `cd atlas-intel-ui && npm run build`
- `git diff --check`
- Confirmed no non-ASCII added lines in the TypeScript diff.

## Estimated diff size

- 4 files.
- About 90 inserted lines and 4 deleted lines.
- Well below the 400-line soft PR budget.
