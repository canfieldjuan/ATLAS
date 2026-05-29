# PR: FAQ Macro Writeback Preview

## Why this slice exists

The FAQ pipeline can now ingest support tickets, generate verified FAQ items,
and reuse those FAQ outputs as source material for blogs and landing pages.
The next product unlock is closing the loop back into the support tool: turning
approved, verified FAQ question/resolution pairs into support macros, saved
replies, or canned responses. This first slice builds the safe core only: a
provider-agnostic macro DTO, deterministic mapping, the verified-and-approved
double gate, and a dry-run preview with no live help-desk API calls. The diff is
over the 400-LOC soft cap because the product contract and its negative tests
need to ship together; splitting the tests from the gate would weaken the slice.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add platform-agnostic macro writeback DTOs and a `MacroPublishProvider` port.
2. Map `TicketFAQDraft.items` into macro drafts when the FAQ draft is approved
   and the item has verified resolution evidence.
3. Return skipped preview rows with explicit reasons for unapproved drafts,
   unverified items, missing questions, and missing resolution bodies.
4. Add a dry-run provider that returns per-item results without network calls.
5. Enroll focused tests in the extracted pipeline check runner.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Preview.md` — plan for this slice.
- `extracted_content_pipeline/faq_macro_writeback.py` — macro DTOs, mapping, gates, dry-run provider.
- `tests/test_extracted_ticket_faq_macro_writeback.py` — focused mapping, gate, and dry-run tests.
- `scripts/run_extracted_pipeline_checks.sh` — extracted check runner enrollment for the new test file.

## Mechanism

The new `faq_macro_writeback` module accepts persisted `TicketFAQDraft` rows and
builds a `MacroWritebackPreview`. Each FAQ item is evaluated independently:

```python
draft.status == "approved"
item["answer_evidence_status"] == "resolution_evidence"
question is present
resolution_text / answer / steps produce a body
```

Items that pass become `SupportMacroDraft` values with a title, body, category,
FAQ draft id, FAQ item id, source ticket ids, and small metadata. Items that do
not pass become `MacroWritebackSkippedItem` values with a machine-readable
reason. `DryRunMacroPublishProvider.publish(...)` mirrors the future outbound
provider port but returns `dry_run` results only.

## Intentional

- No live Zendesk, Intercom, or Freshdesk adapter is added in this slice. The
  customer-facing safety contract must be proven before credentials, rate
  limits, and platform-specific request shapes enter the system.
- No idempotency table is added yet. The preview DTO carries `faq_draft_id` and
  `faq_item_id` so the later mapping table has stable keys, but live upsert is a
  separate slice.
- Draft items marked `draft_needs_review` are skipped even when they contain
  usable-looking steps. Macro writeback is customer-facing, so the gate is
  stricter than blog/landing ingestion.

## Deferred

- Future PR `PR-FAQ-Macro-Writeback-Idempotency`: persist per-tenant platform
  mapping from FAQ item ids to external macro ids.
- Future PR `PR-FAQ-Macro-Writeback-Zendesk-Adapter`: add the first live
  outbound provider adapter with scoped credentials and transport tests.
- Future PR `PR-FAQ-Macro-Writeback-Publish-Trigger`: wire approval workflow
  and publish trigger around the provider port.
- Parked hardening: none

## Verification

- `python -m py_compile extracted_content_pipeline/faq_macro_writeback.py tests/test_extracted_ticket_faq_macro_writeback.py` -- passed.
- `python -m pytest tests/test_extracted_ticket_faq_macro_writeback.py -q` -- 5 passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py` -- passed, 127 matching tests enrolled.
- `bash scripts/validate_extracted_content_pipeline.sh` -- passed.
- `bash scripts/check_ascii_python.sh` -- passed.
- `python scripts/check_extracted_imports.py` -- passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- passed.
- `python scripts/smoke_extracted_pipeline_imports.py` -- passed.
- `python scripts/smoke_extracted_pipeline_standalone.py` -- passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -- passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-preview.md` -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~97 |
| Macro writeback module | ~230 |
| Tests | ~180 |
| Runner enrollment | ~1 |
| Total | ~508 |
