# PR: Content Ops Source Field Aliases

## Why this slice exists

AI Content Ops source adapters support reviews, calls, CRM rows, tickets,
surveys, renewals, and bundles, but the row-key lookup is exact. Real host CSV
and JSON exports commonly use labels such as "Ticket ID", "Account Name",
"Pain Category", and "Open Ended Response". Those rows should enter the same
source-to-opportunity path without host-side pre-renaming.

## Scope

1. Add tolerant source-row key lookup for case, spacing, dash, and underscore
   differences.
2. Canonicalize common buyer/vendor/contact fields inside the source adapter
   before the existing opportunity normalizer runs.
3. Add focused tests for ticket and survey/provider-style field labels.
4. Refresh host-facing docs and coordination state.

## Mechanism

The source adapter will keep exact-key precedence, then fall back to normalized
key comparisons. Normalization is local to source-row ingestion and does not
change the general customer-opportunity adapter.

## Intentional

- No new source format.
- No provider API clients.
- No database, generation, or prompt changes.
- No change to the normalized campaign opportunity contract.

## Deferred

- Provider-specific importers for Salesforce, HubSpot, Zendesk, Intercom, or
  Gong exports.
- Complex nested object flattening beyond the existing source-bundle path.

## Verification

- Run the focused source-adapter tests.
- Compile the touched Python files.
- Run the local PR review script.
- Run diff whitespace checks.

### Files Touched

- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/campaign_source_adapters.py`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `plans/PR-Content-Ops-Source-Field-Aliases.md`
- `tests/test_extracted_campaign_source_adapters.py`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Source adapter | ~125 |
| Tests | ~80 |
| Docs and coordination | ~40 |
| Plan doc | ~55 |
| **Total** | ~300 |
