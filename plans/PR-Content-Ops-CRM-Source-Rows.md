# Content Ops CRM Source Rows

## Why This Slice Exists

AI Content Ops already accepts reviews, tickets, surveys, calls, meetings, and
generic documents as source rows. Many hosts will start with CRM deal exports
and account notes before they have cleaner review or support-ticket exports.
This slice lets those rows enter the same deterministic source-to-opportunity
path without adding a CRM provider integration.

## Scope

- Recognize CRM deal and note collection keys in source bundle JSON.
- Recognize CRM deal/note identifiers on individual rows.
- Infer source types for CRM deals and CRM notes.
- Preserve the existing normalized opportunity contract.
- Document the new accepted row shapes.

## Mechanism

The source adapter extends the existing static key lists and source-type
inference. Rows with `deal_id` or `opportunity_id` are treated as `crm_deal`
evidence. Rows with `note_id` or `activity_id` are treated as `crm_note`
evidence. Existing source text precedence remains unchanged, so CRM rows use
`summary`, `notes`, `description`, `message`, or any other already-supported
text field.

## Intentional

- No new CRM API client.
- No schema migration.
- No changes to generation prompts or the campaign draft contract.
- No special handling for provider-specific fields beyond preserving them as
  normal custom opportunity metadata.

## Deferred

- Provider-specific CRM importers for HubSpot, Salesforce, or Pipedrive.
- Rich account/contact reconciliation across multiple CRM rows.
- Source-bundle examples that include every supported CRM shape.

## Verification

- Focused source-adapter tests.
- Python compile check for the adapter and tests.
- Diff whitespace check.
- Local PR review script.

### Files Touched

- `extracted_content_pipeline/campaign_source_adapters.py`
- `tests/test_extracted_campaign_source_adapters.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `extracted_content_pipeline/STATUS.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-CRM-Source-Rows.md`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Adapter keys | ~20 |
| Tests | ~100 |
| Docs and coordination | ~35 |
| Plan doc | ~70 |
| **Total** | ~225 |
