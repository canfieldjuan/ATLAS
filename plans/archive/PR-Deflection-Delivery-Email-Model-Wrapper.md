# PR-Deflection-Delivery-Email-Model-Wrapper

## Why this slice exists

#1588 split the paid deflection report into separate surfaces: hosted result
page as the operating dashboard, PDF as a curated/shareable report, evidence
export as the complete archive, and email as a short delivery wrapper with key
numbers and links/attachments.

The structured `deflection.v1` report model is now persisted and consumed by
the PDF renderer and hosted paid result page, but the paid delivery email still
renders a generic "your report is ready" body. That leaves the email as the last
customer-facing surface that does not consume the model.

Root cause: the delivery email body predates the structured report model, so it
has no model projection seam and can only say "open the report." This fixes the
root for the email surface by rendering its summary from the persisted
`email_summary` model section, while preserving a legacy fallback for old
artifacts that do not carry `report_model`.

CI follow-up root cause: the dedicated delivery workflow collects
`tests/test_deflection_report_delivery_task.py`, whose task-registration
assertions import the autonomous scheduler. `apscheduler` requires `tzlocal`,
but `requirements.txt` only named APScheduler and left that scheduler runtime
dependency transitive. This PR makes `tzlocal` explicit so the delivery CI lane
does not depend on resolver/transitive behavior to collect scheduler-backed
task tests.

Diff-size note: this is over the 400 LOC soft cap because the renderer change
and its model/future-schema/malformed-data/legacy-fallback tests are one safety
unit, plus the one-line explicit scheduler dependency needed to make the
dedicated delivery CI lane collect. Splitting the tests into a follow-up would
leave the new customer-facing email path under-proven in the PR that introduces
it.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-shape
Slice phase: Vertical slice

1. Decode the paid report artifact once in the delivery path and derive a short
   email summary from the stored `deflection.v1` model when available.
2. Keep existing claim/send/finalize semantics, result URL generation,
   idempotency key, PDF attachment behavior, and link-only fallback unchanged.
3. Render HTML and text bodies with key numbers from the model plus the result
   link and attachment copy.
4. Add focused delivery tests covering model-backed email content, legacy
   fallback, schema-drift fallback, no raw Markdown/evidence leakage, and
   existing idempotency/attachment behavior.
5. Make the scheduler's `tzlocal` runtime dependency explicit because the
   dedicated delivery CI lane collects scheduler-backed task-registration tests.

### Files touched

- `atlas_brain/content_ops_deflection_delivery.py`
- `plans/PR-Deflection-Delivery-Email-Model-Wrapper.md`
- `requirements.txt`
- `tests/test_atlas_content_ops_deflection_delivery.py`

### Review Contract

Acceptance criteria:

- Model-backed artifacts render a concise email wrapper using only
  `deflection.v1` `email_summary` data, including repeat-ticket count,
  generated question count, support-cost sizing, drafted answer count, gap
  count, and ticket-source count.
- Legacy or schema-drift artifacts still send the prior generic delivery email
  instead of failing or rendering misleading zeroes.
- Email bodies do not expose full Markdown, raw evidence quotes, source IDs, or
  `resolution_evidence` internals.
- Existing PDF attachment/link-only fallback and deterministic send
  idempotency behavior remain unchanged.
- The dedicated delivery workflow can collect the scheduler-backed delivery
  task tests without relying on a hidden transitive dependency.

Affected surfaces:

- Paid FAQ deflection report delivery email body in
  `atlas_brain/content_ops_deflection_delivery.py`.
- Existing delivery worker tests in
  `tests/test_atlas_content_ops_deflection_delivery.py`.

Risk areas:

- Accidentally changing the delivery claim/finalize transaction semantics.
- Emitting misleading defaults when the model is missing, malformed, or a
  future schema.
- Pulling uncapped evidence into the email body instead of keeping email short.
- Hiding a scheduler runtime dependency behind APScheduler's transitive install
  edge.

Triggered reviewer rules:

- R1 requirements match, R2 test evidence, R8 idempotency/retry behavior,
  R12 CI/test enrollment, R13 class fix, R14 codebase verification.

## Mechanism

The delivery worker already fetches `r.artifact` for PDF rendering. This slice
decodes that artifact once, passes the decoded mapping to both the PDF
attachment helper and the email body builder, and uses the extracted stored
model projector to fail closed on malformed, missing, or future-schema models.

A small email-summary helper selects model sections whose `surfaces` include
`email_summary`; the current registry exposes `support_tax` for that surface.
If a valid section is present, the helper formats key numbers into concise HTML
and text bodies. If no valid summary is available, the existing generic body is
used unchanged.

The CI repair adds `tzlocal>=3.0,<6.0` next to `apscheduler>=3.10,<4.0` in
`requirements.txt`. That matches APScheduler's declared runtime dependency and
keeps the autonomous scheduler import collectable in the delivery workflow.

## Intentional

- This PR does not add a new section registry surface. It consumes the existing
  `email_summary` marker already on `support_tax`, keeping the slice to the
  delivery renderer instead of changing extracted model metadata.
- This PR does not change delivery CLI env parsing or config. No new setting is
  needed for a deterministic renderer choice.
- This PR does not make the email a mini-report. The hosted page, PDF, and
  export own detailed reading; email stays a delivery wrapper.
- This PR does not change the delivery workflow file. The failure is addressed
  at the runtime dependency boundary used by both CI and production imports.

## Deferred

- Direct per-section email-summary renderers can be added if future sections
  earn an `email_summary` surface. This slice only needs the existing
  `support_tax` summary to close the current surface gap.
- Optional PDF bookmarks/navigation remain parked under #1588 and are not
  required for email.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_atlas_content_ops_deflection_delivery.py -q`
  - 18 passed.
- `python scripts/sync_pr_plan.py plans/PR-Deflection-Delivery-Email-Model-Wrapper.md --check`
  - Passed.
- `rg -n "test_atlas_content_ops_deflection_delivery" scripts/run_extracted_pipeline_checks.sh .github/workflows -g '*.sh' -g '*.yml'`
  - Confirmed existing test enrollment in the extracted check script and
    dedicated delivery workflow.
- `python -m compileall -q <changed Python files>`
  - Passed.
- `python -m pytest tests/test_alerts.py tests/test_atlas_content_ops_deflection_delivery.py tests/test_content_ops_deflection_incidents.py tests/test_deflection_pdf_renderer.py tests/test_deflection_report_delivery_task.py tests/test_send_content_ops_deflection_report_deliveries.py -q`
  - 119 passed, 1 warning.
- Push-wrapper local review via the project wrapper.
  - Passed before the CI dependency fix; pending rerun before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/content_ops_deflection_delivery.py` | 231 |
| `plans/PR-Deflection-Delivery-Email-Model-Wrapper.md` | 161 |
| `requirements.txt` | 1 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 165 |
| **Total** | **558** |
