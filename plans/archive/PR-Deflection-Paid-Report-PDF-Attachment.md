# PR-Deflection-Paid-Report-PDF-Attachment

## Why this slice exists

Issue #1406 asks for deflection customers to receive report PDFs by email,
not only links. The end-to-end delivery surface has two owners: the free
Snapshot email is portfolio-owned, while the paid full-report delivery worker
lives in ATLAS. This PR takes the ATLAS-side half first so paid buyers get a
shareable/offline PDF attached to the existing queued delivery email without
changing checkout, unlock, or portfolio Snapshot behavior.

The diff is over the 400 LOC soft cap because the slice needs the renderer,
shared send-port attachment support, delivery-worker wiring, archive
housekeeping, and behavior tests together; splitting the send-port and worker
changes would leave no shippable attachment path to verify.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Production hardening

1. Attach a PDF copy of the paid full deflection report from the existing
   `content_ops_deflection_report_deliveries` queue when persisted artifact
   Markdown is present.
2. Extend the shared campaign `SendRequest`/Resend adapter to carry provider
   attachments and fail closed for SES attachments, which this flow does not
   use.
3. Keep delivery best-effort: if PDF rendering fails, send the paid report
   link without attachment copy and still mark the email delivered when the ESP
   send succeeds.
4. Archive the merged #1416 plan doc as required teardown housekeeping.

### Review Contract

- Acceptance criteria:
  - [ ] Paid deflection delivery emails include one base64 PDF attachment when
        the report artifact has Markdown.
  - [ ] Email body copy claims an attached PDF only when the attachment was
        successfully rendered.
  - [ ] PDF render failure does not fail an otherwise sendable paid delivery.
  - [ ] Resend receives the attachment payload through the shared send port;
        SES fails closed instead of silently dropping unsupported attachments.
  - [ ] No Snapshot/free-email behavior, checkout authorization, Stripe, or
        portfolio code changes are included.
- Affected surfaces: jobs, email/third-party sender port, paid report export,
  plan archive housekeeping.
- Risk areas: paid-content privacy, email deliverability, provider contract
  compatibility, retry/idempotency, CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R8, R10, R12.

### Files touched

- `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml`
- `atlas_brain/content_ops_deflection_delivery.py`
- `atlas_brain/deflection_pdf_renderer.py`
- `extracted_content_pipeline/campaign_ports.py`
- `extracted_content_pipeline/campaign_sender.py`
- `extracted_content_pipeline/services/campaign_sender.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Paid-Report-PDF-Attachment.md`
- `plans/archive/PR-Deflection-Real-Data-Proof.md`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_deflection_pdf_renderer.py`
- `tests/test_extracted_campaign_sender.py`
- `tests/test_extracted_vendor_briefing_seams.py`

## Mechanism

The delivery worker already claims rows from
`content_ops_deflection_report_deliveries` joined to
`content_ops_deflection_reports`. This PR adds `r.artifact` to that projection,
renders the artifact's persisted full-report Markdown with a small FPDF-based
renderer, base64-encodes the resulting bytes, and passes the attachment through
`SendRequest.attachments` to the Resend sender payload. The renderer lives at
`atlas_brain/deflection_pdf_renderer.py` and is imported lazily inside the
attachment helper so importing the delivery worker does not execute
`atlas_brain.services.__init__` or require torch/fpdf in dry-run and
torch-less extracted-check environments.

The email body is rendered from `has_attachment`, so attachment-specific copy
only appears after PDF bytes are available. Rendering exceptions are logged and
return an empty attachment tuple; the worker then sends the same secure result
link and keeps the existing delivered/failed state transition behavior tied to
the ESP send result.

## Intentional

- The renderer converts the existing paid Markdown artifact into a readable
  PDF instead of re-running report generation or reconstructing the report from
  nested FAQ items. That keeps the attachment aligned with the unlocked report
  customers already see.
- PDF rendering is best-effort. A paid delivery email with the secure result
  link is better than failing the entire paid delivery because FPDF could not
  render an attachment.
- SES attachment support fails closed with `NotImplementedError`; this slice's
  production path uses Resend, and silently dropping attachments would be less
  safe than an explicit unsupported-provider error.
- No new dependency is added; this reuses the existing `fpdf` stack.
- The PDF renderer stays out of `atlas_brain.services`; that package eagerly
  imports torch-backed service modules and is too heavy for the paid delivery
  worker/CLI import path.

## Deferred

- Portfolio-owned Snapshot email PDF attachment remains the next #1406 slice;
  this branch does not touch `atlas-portfolio` or free Snapshot delivery.
- Deflection PDF visual polish beyond readable Markdown export is deferred
  until the attachment path is proven in production.

Parked hardening: none.

## Verification

- `pytest -q tests/test_atlas_content_ops_deflection_delivery.py tests/test_send_content_ops_deflection_report_deliveries.py tests/test_extracted_campaign_sender.py tests/test_deflection_pdf_renderer.py tests/test_extracted_vendor_briefing_seams.py::test_campaign_sender_adapter_converts_legacy_kwargs_to_send_request` - 32 passed.
- `python - <<'PY' ... import atlas_brain.content_ops_deflection_delivery ... assert 'atlas_brain.services' not in sys.modules` - passed.
- `scripts/validate_extracted_content_pipeline.sh` via bash - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `scripts/check_ascii_python.sh` via bash - passed.
- `extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline` via bash - refreshed 46 files; no extra drift beyond intended files.
- `scripts/run_extracted_pipeline_checks.sh` via bash - 3538 passed, 10 skipped, 1 torch/pynvml warning.
- `python scripts/sync_pr_plan.py plans/PR-Deflection-Paid-Report-PDF-Attachment.md --check` - passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main` - passed.
- `python -m ruff check ...` - not run; `ruff` is not installed in this environment.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml` | 5 |
| `atlas_brain/content_ops_deflection_delivery.py` | 81 |
| `atlas_brain/deflection_pdf_renderer.py` | 188 |
| `extracted_content_pipeline/campaign_ports.py` | 1 |
| `extracted_content_pipeline/campaign_sender.py` | 4 |
| `extracted_content_pipeline/services/campaign_sender.py` | 2 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Paid-Report-PDF-Attachment.md` | 143 |
| `plans/archive/PR-Deflection-Real-Data-Proof.md` | 0 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 81 |
| `tests/test_deflection_pdf_renderer.py` | 30 |
| `tests/test_extracted_campaign_sender.py` | 35 |
| `tests/test_extracted_vendor_briefing_seams.py` | 2 |
| **Total** | **575** |
