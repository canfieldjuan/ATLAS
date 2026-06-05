## Why this slice exists

The live FAQ deflection flow collects `contact_email` during report submission,
but the paid-gated report row only persists snapshot/artifact/paid state. That
blocks the next delivery step: after a buyer unlocks the report, ATLAS needs a
durable delivery address to send the report link or follow-up conversion email.

This slice adds the durable metadata only. It does not send email, expose the
address in the free snapshot, or change the Stripe/paywall trust boundary.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection-backend

Slice phase: Production hardening

1. Add a nullable delivery email column for persisted deflection report rows.
2. Thread a cleaned `contact_email` from Content Ops execute inputs into the
   deflection report store when the gated report artifact is saved.
3. Preserve existing paid/payment state across report regenerations.
4. Keep snapshots and customer-facing result payloads free of delivery email.
5. Add focused tests for in-memory submit persistence and Postgres store
   round-trip behavior.

### Files touched

- `atlas_brain/storage/migrations/331_content_ops_deflection_report_delivery_email.sql`
- `extracted_content_pipeline/deflection_report_access.py`
- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_extracted_content_deflection_submit.py`
- `plans/PR-Deflection-Delivery-Email-Metadata.md`

## Mechanism

`DeflectionReportArtifactStore.save_report(...)` accepts an optional
`delivery_email`. The control-surface execution path extracts
`inputs.contact_email`, cleans it, and passes it only at the storage boundary
when a `faq_deflection_report` artifact is gated.

The Postgres adapter persists the value in `content_ops_deflection_reports` via
a new nullable `delivery_email` column. On conflict, a non-empty new delivery
email replaces the prior value; a missing/blank value leaves the prior value in
place, so regenerating a report without delivery metadata does not erase the
address needed for later delivery.

The free snapshot route still returns only the stored snapshot JSON. The paid
artifact route still returns only the full artifact after the verified paid
gate. The new delivery email is store metadata for future delivery workers, not
a customer payload field.

## Intentional

- No email is sent in this slice. Delivery requires a separate worker/template
  slice after the persistence contract exists.
- The delivery email is not added to `DeflectionSnapshot`,
  `FAQDeflectionReportArtifact`, or the portfolio result page response.
- This does not alter Stripe metadata, payment verification, paid unlock, or
  checkout behavior.
- The field is named `delivery_email`, not `contact_email`, because it records
  the address ATLAS may use for post-purchase delivery and follow-up workflows.

## Deferred

- Future slice: post-webhook report delivery email using the persisted
  `delivery_email` and canonical result URL.
- Future slice: abandoned/checkout-cancel follow-up capture policy and opt-out
  rules.
- Parked hardening: none.

## Verification

- Py compile for `extracted_content_pipeline/deflection_report_access.py`,
  `extracted_content_pipeline/api/control_surfaces.py`,
  `tests/test_content_ops_deflection_report.py`, and
  `tests/test_extracted_content_deflection_submit.py` - passed.
- Focused pytest for `tests/test_content_ops_deflection_report.py` and
  `tests/test_extracted_content_deflection_submit.py` - 39 passed in 0.61s;
  rechecked after rebase, 39 passed in 0.60s.
- Extracted package guardrails - passed:
  `scripts/validate_extracted_content_pipeline.sh`,
  `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py`,
  `scripts/audit_extracted_standalone.py`, and
  `scripts/check_ascii_python.sh`.
- Full extracted mirror with
  `EXTRACTED_DATABASE_URL=postgresql://atlas@localhost:5433/atlas` and hosted
  deflection smoke env vars blanked - 2924 passed, 1 warning in 53.91s.
- Local PR review bundle with the current PR body - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Store + API threading | ~45 |
| Migration | ~10 |
| Tests | ~35 |
| Plan doc | ~85 |
| **Total** | **~175** |

Under the 400 LOC soft cap.
