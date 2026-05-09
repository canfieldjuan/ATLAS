# PR: Content Ops Generated Asset Result Summaries

Date: 2026-05-09
Owner: codex-content-ops-asset-summaries

## Goal

Extend the Content Ops execution panel beyond `email_campaign` and
`signal_extraction` so implemented generated-asset outputs also show a
useful summary before the raw JSON payload.

## Scope

- Add a shared result summary for `report`, `landing_page`, and
  `sales_brief`.
- Preserve the existing `email_campaign` summary behavior.
- Keep raw JSON details available for every step.
- Update the frontend contract doc to name the shared result shape.

## Verified Contracts

`ReportGenerationResult`, `LandingPageGenerationResult`, and
`SalesBriefGenerationResult` all expose:

- `requested`
- `generated`
- `skipped`
- `saved_ids`
- `errors`

The UI can summarize these fields without changing backend contracts.

## Out Of Scope

- Blog post summary adapter. Its result contract needs a separate read.
- Backend result-shape changes.
- Competitive-intelligence files currently claimed by another session.

## Validation

- `npm run build` in `atlas-intel-ui`.
