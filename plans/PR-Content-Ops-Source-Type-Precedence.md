# PR: Content Ops Source-Type Precedence Table

## Why this slice exists

The source adapter now supports reviews, transcripts, calls, meetings, CRM
rows, lifecycle rows, tickets, surveys, bundles, and tolerant field aliases.
The current source-type inference order is product behavior, but it lives as
an implicit if-chain.

This slice makes that order explicit without adding new source families.

## Scope

1. Replace the implicit `_infer_source_type` if-chain with a local ordered
   source-type precedence table.
2. Add focused tests that lock the table order and preserve ambiguous-row
   behavior.
3. Keep the normalized opportunity contract unchanged.

## Mechanism

`_SOURCE_TYPE_PRECEDENCE` maps ordered key groups to the source type they
produce. `_infer_source_type` iterates the table and returns the first match,
falling back to `document`.

## Intentional

- No new source-type labels.
- No provider-specific source adapters.
- No changes to generated asset behavior.
- No public API or function signature changes.

## Deferred

- Data-driven source-family registry.
- Provider-specific source importers.
- End-to-end generated-asset quality tests by source type.

## Verification

- Run pytest tests/test_extracted_campaign_source_adapters.py -q.
- Run python -m py_compile extracted_content_pipeline/campaign_source_adapters.py tests/test_extracted_campaign_source_adapters.py.
- Run bash scripts/local_pr_review.sh.

### Files Touched

- `extracted_content_pipeline/campaign_source_adapters.py`
- `tests/test_extracted_campaign_source_adapters.py`
- `plans/PR-Content-Ops-Source-Type-Precedence.md`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Source adapter | ~30 |
| Tests | ~25 |
| Plan | ~40 |
| **Total** | ~145 |
