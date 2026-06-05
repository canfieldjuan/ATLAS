# PR-FAQ-Macro-Writeback-Functional-Validation

## Why this slice exists

The FAQ macro writeback operator path is now reachable end to end from the
dashboard: tenant credentials can be provisioned, approved FAQ Markdown drafts
can be reviewed, and the review UI can call the hosted publish route. The lane
has good unit coverage at each layer, but it still needs one focused functional
validation that proves the core publish chain works as a chain: approved FAQ
draft -> publish service -> tenant credential lookup -> Zendesk provider ->
idempotency mapping -> Zendesk transport -> draft status update.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Functional validation

1. Add a focused no-network functional test for the FAQ macro writeback publish
   chain.
2. Validate tenant-scoped credentials are used instead of config fallback when
   the request has an account id.
3. Validate the first publish creates one Zendesk macro and marks the FAQ draft
   published.
4. Validate a second publish reuses the saved mapping and updates the existing
   macro instead of creating a duplicate.
5. Enroll the new validation test in the extracted pipeline check runner.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Functional-Validation.md` -- plan for this slice.
- `scripts/run_extracted_pipeline_checks.sh` -- CI enrollment for the validation test.
- `tests/test_content_ops_faq_macro_writeback_flow.py` -- functional publish-chain validation.

## Mechanism

The new test builds the real `FAQMacroWritebackPublishService` with the real
`ZendeskMacroPublishProvider` and `TenantZendeskMacroCredentialsProvider`.
Only the outer dependencies are faked:

1. A fake FAQ repository returns one approved, resolution-backed FAQ draft and
   records status updates.
2. A fake tenant credential lookup returns a Zendesk credential only for the
   expected tenant account.
3. An in-memory mapping repository records idempotency reservations and upserts.
4. A fake Zendesk transport records POST/PUT calls and returns realistic macro
   response envelopes.

The first service call must POST one macro, persist the mapping, and mark the
FAQ draft published. The second service call over the same mapping repository
must PUT the existing macro id and avoid a second POST.

## Intentional

- No live Zendesk or database access. This is functional validation of our
  integration contracts, not an external smoke test.
- No UI/browser automation in this slice. PR #1152 already pinned the UI route
  wrapper; this slice validates the publish chain behind that route.
- No product behavior changes. If this test fails, the source chain is broken
  and should be fixed in the failing layer.

## Deferred

- `PR-FAQ-Macro-Writeback-Live-Smoke`: optional operator-only smoke against a
  sandbox Zendesk account once live credentials and safe test data are available.
- `PR-FAQ-Macro-Writeback-Publish-History`: persist prior publish attempts if
  operators need an audit trail beyond the latest publish response.

Parked hardening: none

## Verification

- python -m pytest tests/test_content_ops_faq_macro_writeback_flow.py -q -- 1 passed.
- python -m pytest tests/test_atlas_content_ops_macro_writeback.py tests/test_content_ops_zendesk_credentials.py tests/test_content_ops_faq_macro_writeback_flow.py -q -- 15 passed.
- python -m py_compile tests/test_content_ops_faq_macro_writeback_flow.py -- passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- passed; 135 matching tests enrolled.
- git diff --check -- passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-functional-validation.md -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~82 |
| Test | ~265 |
| CI enrollment | ~1 |
| Total | ~349 |
