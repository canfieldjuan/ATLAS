# PR: Support Ticket Source Context Live Validation

## Why this slice exists

PR #930 fixed landing-page support-ticket source context, but the live smoke
needed one more end-to-end proof after merge. The first validation pass also
showed the support-ticket blog smoke helper could still claim a 90-day source
window when the CSV rows did not carry parseable dates. That is the same
evidence-truthfulness issue fixed for landing inputs, so this slice fixes the
smoke helper and records the live proof.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: functional validation

1. Gate the support-ticket blog smoke blueprint's `review_period`,
   `source_period`, and `source_window_days` on parseable ticket dates.
2. Keep undated support-ticket CSV smoke runs on neutral `Uploaded support
   tickets` wording.
3. Add focused tests for undated and dated blog blueprint source-period
   behavior.
4. Document the Haiku live landing-page and blog-post smoke/export validation
   after #930 merged.

### Files touched

- `plans/PR-Support-Ticket-Source-Context-Live-Validation.md`
- `scripts/smoke_content_ops_live_generation.py`
- `tests/test_smoke_content_ops_live_generation.py`
- `docs/extraction/validation/support_ticket_source_context_live_validation_2026-05-24.md`

## Mechanism

The smoke helper now inspects common ticket date columns before claiming a
`Last 90 days` window. If every included row has a parseable date, the
blueprint keeps the 90-day source-period language and includes
`source_window_days=90`. Otherwise it uses neutral uploaded-ticket wording and
omits the window stat.

## Intentional

- This changes only the live smoke helper and validation docs. Hosted FAQ,
  file-ingestion, landing generation, and blog generation services are not
  changed.
- The live smoke still uses the packaged support-ticket CSV and Haiku override
  for low-cost validation.

## Deferred

- Larger customer CSV validation remains a later robust-testing slice.
- Parked hardening: none. `HARDENING.md` was scanned; current entries are FAQ
  scale and file-ingestion concurrency work outside this support-ticket
  provider validation slice.

## Verification

- Focused smoke tests:
  `pytest tests/test_smoke_content_ops_live_generation.py::test_support_ticket_blog_blueprint_payload_uses_csv_counts tests/test_smoke_content_ops_live_generation.py::test_support_ticket_blog_blueprint_payload_uses_date_window_when_dates_validate tests/test_smoke_content_ops_live_generation.py::test_live_generation_smoke_packages_support_ticket_csv_for_landing_page -q`
  - 3 passed.
- Expanded support-ticket/source-context suites:
  `pytest tests/test_extracted_support_ticket_input_package.py tests/test_extracted_support_ticket_input_provider.py tests/test_smoke_content_ops_live_generation.py tests/test_extracted_content_ops_live_execute_harness.py tests/test_extracted_content_ops_execution.py tests/test_extracted_landing_page_generation.py -q`
  - 148 passed.
- Py compile for changed Python files - passed.
- Git whitespace check - passed.
- Manual Haiku landing-page live smoke with `--support-ticket-csv` and
  `--export-saved-draft` - passed; exported draft
  `d181fc92-1711-40dd-98e0-e79bcdb1c304`.
- Manual Haiku blog-post live smoke with `--support-ticket-csv` and
  `--export-saved-draft` after the source-period fix - passed; exported draft
  `90cf80e1-baad-478a-8b25-98394c509279`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Smoke helper | ~45 |
| Tests | ~20 |
| Validation doc | ~90 |
| **Total** | **~230** |
