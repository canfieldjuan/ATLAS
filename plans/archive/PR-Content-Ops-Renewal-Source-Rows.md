# Content Ops Renewal Source Rows

## Why This Slice Exists

AI Content Ops can already turn reviews, transcripts, support tickets, CRM
deals, and CRM notes into normalized campaign opportunities. Customer revenue
teams also export contract, renewal, and subscription records as the source of
truth for expansion and retention plays. Those rows should enter the same
source-row adapter instead of forcing hosts to pre-map them into generic CRM
deal rows.

This slice keeps the existing source-row contract and extends the taxonomy to
cover commercial lifecycle records.

## Scope

- Accept contract, renewal, and subscription collection keys in JSON bundles.
- Preserve contract, renewal, and subscription identifiers as source ids.
- Infer stable source types for rows carrying those identifiers.
- Cover direct rows and bundled rows with focused tests.
- Update the host-facing docs and coordination ledger.

## Mechanism

The adapter continues to normalize source rows through the existing campaign
opportunity path. New commercial row shapes only affect source discovery and
source-type inference:

- bundle collections: contracts, contract notes, renewals, renewal notes,
  subscriptions, and subscription notes
- identifier fields: contract id, renewal id, and subscription id
- inferred types: contract, renewal, and subscription

Text extraction, evidence shaping, parent metadata inheritance, warning
behavior, and downstream opportunity normalization are unchanged.

## Intentional

- Contract, renewal, and subscription rows are source evidence, not separate
  generation modes.
- Review and transcript text still wins over commercial ids when both appear in
  one row, preserving the existing source-type precedence.
- Existing CRM deal and CRM note behavior stays unchanged.

## Deferred

- No new packaged example file is added in this slice. Existing tests cover the
  bundle shape without expanding the starter data surface.
- No database import schema changes are needed because the source adapter emits
  the existing normalized opportunity payload.
- No UI changes are included; this is host ingestion support only.

## Verification

- Focused source-adapter tests cover direct contract, renewal, and subscription
  rows plus a bundled renewal export.
- Python compile and diff checks cover the changed adapter and tests.
- Local PR review covers repository policy checks before the branch is pushed.

### Files Touched

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-Renewal-Source-Rows.md` | 80 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `extracted_content_pipeline/README.md` | 25 |
| `extracted_content_pipeline/STATUS.md` | 12 |
| `extracted_content_pipeline/campaign_source_adapters.py` | 19 |
| `extracted_content_pipeline/docs/host_install_runbook.md` | 13 |
| `tests/test_extracted_campaign_source_adapters.py` | 164 |

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Adapter keys and precedence comment | ~20 |
| Tests | ~165 |
| Docs and coordination | ~35 |
| Plan doc | ~80 |
| **Total** | ~315 |
