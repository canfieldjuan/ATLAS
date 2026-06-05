# PR: FAQ Macro Writeback Zendesk Adapter

## Why this slice exists

The macro writeback preview and idempotency mapping now exist, but no outbound
support-tool adapter can use them. The next vertical slice is the first real
platform adapter: Zendesk macro create/update. This keeps the live API surface
behind injectable credentials and transport so CI remains no-network while the
production request shape, auth boundary, idempotency lookup, and per-item
result handling are proven before any publish trigger is wired.
The diff is over the 400-LOC soft cap because the live-adapter request shape,
credential redaction, idempotency behavior, pending-mapping retry safety, and
failure isolation need to ship with tests in the same slice.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add a Zendesk credentials DTO and scope-aware credentials provider protocol.
2. Add a Zendesk macro transport protocol plus an HTTP transport implementation.
3. Add `ZendeskMacroPublishProvider` that uses the idempotency mapping repo:
   existing mapping -> update macro; missing mapping -> reserve pending mapping
   -> create macro -> complete mapping.
4. Map a `SupportMacroDraft` to Zendesk's macro payload using `comment_value`
   plus public comment mode.
5. Add focused no-network tests for create, update, pending-row retry refusal,
   create-then-mapping failure, missing credentials, per-item failure isolation,
   auth header shape, and token redaction.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Zendesk-Adapter.md` — plan for this slice.
- `extracted_content_pipeline/faq_macro_writeback.py` — mapping reservation port/status fields used by the adapter.
- `extracted_content_pipeline/faq_macro_writeback_postgres.py` — pending mapping reservation implementation.
- `extracted_content_pipeline/faq_macro_writeback_zendesk.py` — Zendesk credentials, transport, and publish provider.
- `extracted_content_pipeline/storage/migrations/329_ticket_faq_macro_writebacks_pending.sql` — migration allowing pending mapping rows before external ids exist.
- `tests/test_extracted_ticket_faq_macro_writeback_postgres.py` — pending mapping storage and migration tests.
- `tests/test_extracted_ticket_faq_macro_writeback_zendesk.py` — focused adapter tests with fake credentials, mapping repo, and transport.
- `scripts/run_extracted_pipeline_checks.sh` — extracted check runner enrollment for the new test file.

## Mechanism

`ZendeskMacroPublishProvider.publish(...)` asks a scoped credentials provider
for credentials, then handles each macro independently:

```text
mapping exists with external_id -> PUT /api/v2/macros/{external_id}
mapping exists without external_id -> failed pending-reconcile, no POST
mapping missing -> reserve pending mapping -> POST /api/v2/macros
successful response -> complete mapping -> MacroPublishResult(status)
failure -> MacroPublishResult(status="failed") and continue
```

The payload follows Zendesk's macro API shape: `{"macro": {"title": ...,
"actions": [...]}}`, with `comment_value` carrying the approved FAQ answer.
The HTTP transport builds the Basic auth header from `email/token:api_token`
but never stores the token in mapping metadata or error strings. A successful
create whose mapping completion fails returns `zendesk_mapping_persist_failed`
with the external id and leaves a pending mapping row; a retry sees that pending
row and refuses to POST again until reconciliation clears or completes it.

## Intentional

- No API route, UI, scheduler, or publish trigger is added here. The adapter is
  callable but not product-wired until the approval/publish-trigger slice.
- Credentials are provided through a protocol rather than read from global env.
  The host app can back this with encrypted tenant storage later without
  changing the adapter contract.
- The adapter catches per-item boundary failures and returns a failed result
  instead of aborting the whole batch.

## Deferred

- Future PR `PR-FAQ-Macro-Writeback-Publish-Trigger`: wire approved FAQ drafts
  to the provider and decide when a customer can run live publish, including
  operator handling for pending rows that need reconciliation.
- Future PR `PR-FAQ-Macro-Writeback-Credential-Storage`: provide encrypted
  tenant credential storage if the existing host secret pattern is not enough.
- Parked hardening: none

## Verification

- `python -m py_compile extracted_content_pipeline/faq_macro_writeback.py extracted_content_pipeline/faq_macro_writeback_postgres.py extracted_content_pipeline/faq_macro_writeback_zendesk.py tests/test_extracted_ticket_faq_macro_writeback_zendesk.py tests/test_extracted_ticket_faq_macro_writeback_postgres.py` -- passed.
- `python -m pytest tests/test_extracted_ticket_faq_macro_writeback_zendesk.py tests/test_extracted_ticket_faq_macro_writeback_postgres.py tests/test_extracted_ticket_faq_macro_writeback.py -q` -- 20 passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py` -- passed, 129 matching tests enrolled.
- `bash scripts/validate_extracted_content_pipeline.sh` -- passed.
- `bash scripts/check_ascii_python.sh` -- passed.
- `python scripts/check_extracted_imports.py` -- passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -- passed.
- `python scripts/smoke_extracted_pipeline_imports.py` -- passed.
- `python scripts/smoke_extracted_pipeline_standalone.py` -- passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-zendesk-adapter.md` -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~108 |
| Mapping DTO/port | ~9 |
| Mapping repository | ~46 |
| Pending migration | ~24 |
| Zendesk adapter | ~313 |
| Postgres tests | ~54 |
| Zendesk tests | ~304 |
| Runner enrollment | ~1 |
| Total | ~859 |
