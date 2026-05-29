# PR: FAQ Macro Writeback Idempotency

## Why this slice exists

PR `FAQ-Macro-Writeback-Preview` proved that approved, verified FAQ items can
be turned into platform-agnostic support macro drafts without network calls.
The next safety requirement is idempotency: when a future Zendesk, Intercom, or
Freshdesk adapter publishes the same FAQ item more than once, Atlas must update
the existing external macro instead of creating duplicates in the customer's
support tool. This slice adds the tenant-scoped mapping contract and Postgres
storage needed before any live outbound adapter exists.
The diff is over the 400-LOC soft cap because the storage contract, migration,
and scoped upsert tests need to ship together; separating the tests from the
idempotency table would leave the live-adapter prerequisite under-proven.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add a `MacroWritebackMapping` DTO and repository port for storing the
   external macro id that belongs to one FAQ item on one platform.
2. Add a Postgres repository that upserts by account, platform, FAQ draft id,
   and FAQ item id, with scoped lookup for later adapters.
3. Add the `ticket_faq_macro_writebacks` migration with tenant, platform,
   FAQ-item, and external-id uniqueness constraints.
4. Add focused tests for scoped lookup, idempotent upsert SQL, JSON metadata
   round-trip, and migration shape.
5. Enroll the new tests in the extracted pipeline check runner.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Idempotency.md` — plan for this slice.
- `extracted_content_pipeline/faq_macro_writeback.py` — mapping DTO and repository port.
- `extracted_content_pipeline/faq_macro_writeback_postgres.py` — Postgres mapping repository.
- `extracted_content_pipeline/storage/migrations/328_ticket_faq_macro_writebacks.sql` — idempotency mapping table.
- `tests/test_extracted_ticket_faq_macro_writeback_postgres.py` — repository and migration tests.
- `scripts/run_extracted_pipeline_checks.sh` — extracted check runner enrollment for the new test file.

## Mechanism

The mapping key is:

```text
account_id + platform + faq_draft_id + faq_item_id
```

The value is the help-desk `external_id`, optional `external_url`, and JSONB
metadata. `PostgresFAQMacroWritebackMappingRepository.upsert_mapping(...)`
uses `ON CONFLICT (account_id, platform, faq_draft_id, faq_item_id) DO UPDATE`
so repeated publishes update the existing mapping. Lookup always filters by
`TenantScope.account_id`, which keeps tenant isolation at the source before the
future live adapter reads credentials or external ids.

## Intentional

- No live outbound API adapter is added here. Idempotency must be a stable
  storage contract before Zendesk/Freshdesk/Intercom request code depends on it.
- The mapping uses `faq_item_id` instead of macro title because customer-facing
  titles can be edited while the source FAQ item identity should remain stable.
- The table also constrains `(account_id, platform, external_id)` so one
  platform macro id cannot silently map to multiple FAQ items for the same
  tenant.

## Deferred

- Future PR `PR-FAQ-Macro-Writeback-Zendesk-Adapter`: use this mapping repo to
  choose create vs update in the first live adapter.
- Future PR `PR-FAQ-Macro-Writeback-Publish-Trigger`: wire approval workflow and
  publish trigger around the provider plus mapping repository.
- Parked hardening: none

## Verification

- `python -m py_compile extracted_content_pipeline/faq_macro_writeback.py extracted_content_pipeline/faq_macro_writeback_postgres.py tests/test_extracted_ticket_faq_macro_writeback_postgres.py` -- passed.
- `python -m pytest tests/test_extracted_ticket_faq_macro_writeback_postgres.py tests/test_extracted_ticket_faq_macro_writeback.py -q` -- 11 passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py` -- passed, 128 matching tests enrolled.
- `bash scripts/validate_extracted_content_pipeline.sh` -- passed.
- `bash scripts/check_ascii_python.sh` -- passed.
- `python scripts/check_extracted_imports.py` -- passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -- passed.
- `python scripts/smoke_extracted_pipeline_imports.py` -- passed.
- `python scripts/smoke_extracted_pipeline_standalone.py` -- passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-idempotency.md` -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~97 |
| Mapping DTO/port | ~39 |
| Postgres repository | ~105 |
| Migration | ~27 |
| Tests | ~170 |
| Runner enrollment | ~1 |
| Total | ~439 |
