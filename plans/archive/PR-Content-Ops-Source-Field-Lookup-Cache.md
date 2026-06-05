# PR: Content Ops Source Field Lookup Cache

## Why this slice exists

The source adapter now accepts provider-style field aliases by normalizing row
keys during lookup. That is useful for host CSV/JSON exports, but each source
row currently repeats the same normalized-key scan across text, id, title,
company, vendor, pain, and source-type helpers.

This slice keeps the adapter behavior unchanged while caching per-row field
lookups.

## Scope

1. Add a private source-row lookup wrapper that caches normalized field
   lookups for one row.
2. Use the wrapper inside source-row conversion and bundle collection lookup.
3. Add focused tests that preserve exact-key precedence and provider-style
   alias behavior.

## Mechanism

The wrapper keeps the raw mapping as the source of truth, precomputes normalized
and compact forms for raw keys once, and caches lookup results by requested
key. `_field_value` detects the wrapper and delegates to it; raw mappings keep
the existing fallback path.

## Intentional

- No public API changes.
- No new source types or aliases.
- No generated-asset behavior changes.
- No change to exact-key precedence.

## Deferred

- Full data-driven source-family registry.
- End-to-end generated-asset quality tests by source type.
- Provider-specific importers for named CRM/support/call platforms.

## Verification

- Run pytest tests/test_extracted_campaign_source_adapters.py -q.
- Run python -m py_compile extracted_content_pipeline/campaign_source_adapters.py tests/test_extracted_campaign_source_adapters.py.
- Run bash scripts/local_pr_review.sh.

### Files Touched

- `extracted_content_pipeline/campaign_source_adapters.py`
- `tests/test_extracted_campaign_source_adapters.py`
- `plans/PR-Content-Ops-Source-Field-Lookup-Cache.md`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Source adapter | ~55 |
| Tests | ~55 |
| Plan | ~55 |
| **Total** | ~165 |
