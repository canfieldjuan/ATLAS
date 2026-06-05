# PR: Content Ops signal extraction result summary

## Why This Slice Exists

The execute panel now summarizes `email_campaign`, but
`signal_extraction` is the other result shape explicitly called out
in the frontend contract. Its payload is a pipeline artifact rather
than generated content, so users need the count, target mode, warning
count, and opportunity identity fields before the raw JSON block.

## Scope

Files touched:

1. `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
   - Extend `ExecutionStepSummary` with a `signal_extraction` branch.
   - Summarize `generated`, `target_mode`, `warnings.length`, and a
     short list of extracted opportunities.
   - Review follow-up: include `vendor_name` in the opportunity label
     fallback, surface `email_campaign` `errors.length`, and render
     saved IDs as wrapping chips instead of one long string.
   - Keep raw result JSON rendering for every output.

2. `docs/frontend/content_ops_frontend_contract.md`
   - Document the `signal_extraction` summary rule.

3. `docs/extraction/coordination/inflight.md`
   - Claim the slice while the PR is open.

## Mechanism

The summary remains display-only and defensive:

- `generated` renders only when it is a number.
- `target_mode` renders only when it is a string.
- `warnings` renders as a count only when it is an array.
- `opportunities` renders a short list only when it is an array of
  objects, using best-effort identity fields already documented in the
  contract.

Unknown payload fields still render under the raw JSON block.

## Intentional

- No new API or domain result types.
- No backend changes.
- No attempt to render full opportunity details; this slice only
  makes the execution result scannable.
- No backend or API changes to `email_campaign` behavior. The UI
  summary may still surface existing `errors` and wrap existing
  `saved_ids` more readably.

## Deferred

- Rich opportunity cards with evidence and raw metadata sections.
- Dedicated component tests for `ContentOpsNewRun`; the repo does not
  currently have a component-test harness for this page.
- Result adapters for reports, landing pages, and sales briefs.

## Verification

- `cd atlas-intel-ui && npm ci`
- `cd atlas-intel-ui && npm run build`
- `cd atlas-intel-ui && npx eslint src/pages/ContentOpsNewRun.tsx src/api/contentOps.ts src/domain/contentOps/types.ts src/domain/contentOps/fromWire.ts src/api/contentOps.contract.ts src/domain/contentOps/contract.ts`
- `git diff --check`
- no non-ASCII added lines

## Diff Size

Expected production delta:

- `ContentOpsNewRun.tsx`: about 70 lines.
- Docs/coordination: about 5 lines plus this plan.

The slice stays under the soft 400-line PR budget.
