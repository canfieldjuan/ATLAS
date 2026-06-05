# PR: Admin Costs Scraping Saved Calls Default

## Why this slice exists

The #1040 review found two pre-existing red tests in `tests/test_admin_costs.py`:
`test_scraping_summary_quality_uses_canonical_reviews` and
`test_scraping_summary_exposes_source_tier_operational_state_and_maintenance`.
Both fail with `KeyError: 'saved_calls_today'` because the scraping summary route
directly indexes one optional aggregate key while its existing tests and helper
style already allow partial fake rows.

This is a production-hardening cleanup for the admin cost surface before adding
more cost/caching UI on top.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Production hardening

1. Make `/admin/costs/scraping/summary` read today's aggregate fields through
   `_record_value(...)` defaults instead of direct row indexing.
2. Keep the SQL query and response shape unchanged.
3. Pin the missing `saved_calls_today` fixture case as a valid `saved_calls: 0`
   response in the existing route tests.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Admin-Costs-Scraping-Saved-Calls-Default.md` | Plan doc for the scraping summary hardening slice. |
| `atlas_brain/api/admin_costs.py` | Default today's aggregate values through the existing safe row accessor. |
| `tests/test_admin_costs.py` | Assert omitted saved-calls input maps to zero in the response. |

## Mechanism

The `today` payload already handles optional `maintenance_row` fields with
`_record_value(...)`. This slice applies the same pattern to `today_row`:
`runs_today`, `inserted_today`, `errors_today`, and `saved_calls_today` all
default to zero when a fake row or partial DB row omits them. The SQL still
selects `saved_calls_today`, so real production rows continue to report saved
pre-scrape calls.

## Intentional

- This does not change the scraping summary SQL or semantics.
- This does not broaden the admin-costs test gate; it fixes the red route tests
  first. A CI-enrollment change would be a separate workflow/process slice.
- This leaves #1041's cache-health section untouched while it waits for review.

## Deferred

- Future PR: decide whether the full `tests/test_admin_costs.py` suite should be
  enrolled in CI once the known pre-existing failures are drained.
- Parked hardening: none. Root `HARDENING.md` was scanned; this slice directly
  promotes the #1040 reviewer FYI instead of parking it.

## Verification

- Initial reproduction command: python -m pytest tests/test_admin_costs.py::test_scraping_summary_quality_uses_canonical_reviews tests/test_admin_costs.py::test_scraping_summary_exposes_source_tier_operational_state_and_maintenance -q — failed with `KeyError: 'saved_calls_today'`.
- Focused pytest command: python -m pytest tests/test_admin_costs.py::test_scraping_summary_quality_uses_canonical_reviews tests/test_admin_costs.py::test_scraping_summary_exposes_source_tier_operational_state_and_maintenance -q — 2 passed, 1 warning.
- Compile command: python -m compileall -q atlas_brain/api/admin_costs.py tests/test_admin_costs.py — passed.
- Full admin-costs pytest command: python -m pytest tests/test_admin_costs.py -q — 21 passed, 1 warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~70 |
| Route hardening | ~5 |
| Test assertions | ~4 |
| **Total** | **~80** |
