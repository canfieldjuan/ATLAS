# PR-Content-Ops-Support-Ticket-Examples

## Why this slice exists

Content Ops can already ingest support-ticket-like rows through the generic
source adapter, and the CFPB smoke proves a public complaint source can feed
that path. Hosts still do not have a packaged support-ticket CSV or JSON bundle
they can run locally when wiring Zendesk, Intercom, Freshdesk, Salesforce case,
or help desk exports.

This slice adds concrete packaged support-ticket examples and locks them
through the existing conversion and offline smoke commands.

## Scope (this PR)

1. Add a packaged support-ticket CSV example with provider-style column labels.
2. Add a packaged support-ticket JSON bundle with shared account metadata and
   multiple ticket rows.
3. Add tests proving both packaged examples load and the CLI can convert them.
4. Document the support-ticket examples in the README and host install runbook.
5. Remove the merged source-smoke helper coordination row while claiming this
   slice.

### Files touched

- `extracted_content_pipeline/examples/support_ticket_sources.csv`
- `extracted_content_pipeline/examples/support_ticket_bundle.json`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `tests/test_extracted_campaign_source_adapters.py`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Support-Ticket-Examples.md`

## Mechanism

The examples use existing source adapter semantics:

```text
Ticket ID / Account Name / Vendor Name / Subject / Description / Pain Category
```

The CSV path verifies lenient provider-style field labels. The JSON bundle path
verifies shared account/contact metadata inherited by multiple
`support_tickets` rows.

## Intentional

- No adapter code changes. This is a packaged example and regression-test
  slice around behavior that already exists.
- Examples avoid product/seller data coupling. They model customer support
  evidence only.
- The examples are small and synthetic fixture data, not a fabricated live
  customer dataset.

## Deferred

- Live customer support-ticket import from Zendesk/Intercom/Freshdesk APIs
  remains a future connector slice.
- Postgres import smoke coverage for packaged support-ticket examples remains
  out of scope; the current offline conversion/generation path is enough for
  schema guidance.

## Verification

- `pytest tests/test_extracted_campaign_source_adapters.py -q` -> 58 passed.
- `python scripts/build_extracted_campaign_opportunities_from_sources.py extracted_content_pipeline/examples/support_ticket_sources.csv --format csv --output /tmp/support_ticket_opportunities.json` -> passed; wrote 2 support-ticket opportunities.
- `python scripts/build_extracted_campaign_opportunities_from_sources.py extracted_content_pipeline/examples/support_ticket_bundle.json --format json --output /tmp/support_ticket_bundle_opportunities.json` -> passed; wrote 2 support-ticket opportunities.
- `python scripts/smoke_extracted_content_pipeline_host.py extracted_content_pipeline/examples/support_ticket_sources.csv --source-rows --source-format csv --min-drafts 2` -> passed.
- `python -m py_compile tests/test_extracted_campaign_source_adapters.py` -> passed.
- `grep -nP '[^\x00-\x7F]' tests/test_extracted_campaign_source_adapters.py` -> no matches.
- `rg -n "TODO|FIXME|pass$|acct_support_demo|Acme|HubSpot|LegacyCRM|example.com|ops@example.com" extracted_content_pipeline/examples/support_ticket_sources.csv extracted_content_pipeline/examples/support_ticket_bundle.json extracted_content_pipeline/README.md extracted_content_pipeline/docs/host_install_runbook.md tests/test_extracted_campaign_source_adapters.py` -> only fixture/demo values and no TODO/FIXME/pass placeholders.
- `git diff --check` -> passed.
- `bash scripts/local_pr_review.sh` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Examples | ~35 |
| Tests | ~90 |
| Docs + plan + coordination | ~115 |
| **Total** | **~250** |
