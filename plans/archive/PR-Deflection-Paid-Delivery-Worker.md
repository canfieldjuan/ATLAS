## Why this slice exists

The paid webhook now queues post-purchase delivery rows, but nothing consumes
that queue. Buyers can unlock the report in the browser, yet they do not get
the paid result link in their inbox after purchase.

This slice adds the smallest worker core for that handoff: read pending
paid-delivery rows, send a transactional report-link email through an injected
sender, and mark the queue row delivered or failed. It intentionally stops
short of a manual CLI, scheduling, and non-buyer nurture policy.

## Scope (this PR)

Ownership lane: ai-content-ops/faq-deflection-paid-unlock

Slice phase: Production hardening

1. Add a Content Ops deflection delivery service that drains pending delivery
   rows joined to paid report rows with `delivery_email`.
2. Render a transactional email containing only the hosted paid-result link,
   not the report artifact, answers, evidence, or markdown.
3. Update delivery rows to `delivered` with provider/message id on success or
   `failed` with a bounded error on send failure/missing email.
4. Add dry-run support at the service level so operators can wire a manual
   script in a follow-up without changing the queue contract.
5. Enroll the atlas-brain importing test in a dedicated workflow.
6. Honor `ATLAS_DISABLE_DOTENV=1` in the existing deflection hosted-smoke
   scripts so the required full extracted pipeline check can run without
   local live env leakage.

### Files touched

- `atlas_brain/content_ops_deflection_delivery.py`
- `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/smoke_content_ops_deflection_submit_handoff.py`
- `scripts/smoke_content_ops_deflection_portfolio_result_page.py`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_smoke_content_ops_deflection_submit_handoff.py`
- `tests/test_smoke_content_ops_deflection_portfolio_result_page.py`
- `plans/PR-Deflection-Paid-Delivery-Worker.md`

## Mechanism

The worker selects pending queue rows from
`content_ops_deflection_report_deliveries`, joins the corresponding
`content_ops_deflection_reports` row, and requires the report to be paid before
delivery:

```text
pending queue row -> paid report + delivery_email -> send link -> mark delivered
```

The email body includes a portfolio result URL built from a configured URL
template or base URL. It does not embed the paid artifact. The sender is a
small port so tests can use an in-memory fake and a follow-up script can wire
the existing Resend campaign sender adapter.

The worker lives at `atlas_brain/content_ops_deflection_delivery.py` instead
of under `atlas_brain/services/` because `atlas_brain.services.__init__`
eagerly imports the LLM/embedding stack. Keeping this module on a lightweight
path lets the extracted CI lane import the worker test without requiring
torch.

Because the focused test imports `atlas_brain.*`, this slice adds a dedicated
atlas workflow and also enrolls the test in `run_extracted_pipeline_checks.sh`
to satisfy the repo's extracted pipeline enrollment audit.

The full extracted pipeline check also runs hosted-smoke unit tests in this
checkout. Those scripts now honor `ATLAS_DISABLE_DOTENV=1`, matching the
existing local-env opt-out used elsewhere so local live deflection credentials
cannot change unit-test defaults.

## Intentional

- No CLI or scheduler is added; this PR locks the queue consumer contract first.
- The dotenv opt-out change is limited to hosted-smoke scripts that already
  read local live env files; it exists to make the required full runner
  deterministic in developer checkouts.
- No non-buyer abandon/cancel nurture email is added; that needs consent and
  opt-out policy first.
- The email contains a hosted result link only, so the paywall remains enforced
  by the existing artifact route and Stripe-paid state.
- Delivery errors are stored as bounded text, not raw provider payloads.

## Deferred

- Future slice: manual script wiring using the existing Resend sender adapter.
- Future slice: scheduler/cron wiring for automatic delivery polling.
- Future slice: provider webhook ingestion for delivery/open/click events.
- Future slice: abandoned checkout follow-up policy and unsubscribe handling.
- Parked hardening: none.

## Verification

- `python -m py_compile atlas_brain/content_ops_deflection_delivery.py tests/test_atlas_content_ops_deflection_delivery.py && python -m pytest tests/test_atlas_content_ops_deflection_delivery.py -q` -- 7 passed, 1 warning.
- `python -m pytest tests/test_atlas_content_ops_deflection_delivery.py tests/test_smoke_content_ops_deflection_submit_handoff.py tests/test_smoke_content_ops_deflection_portfolio_result_page.py -q` -- 41 passed, 1 warning.
- `bash scripts/run_extracted_pipeline_checks.sh` -- 2934 passed, 10 skipped, 1 warning.
- `bash scripts/local_pr_review.sh --allow-dirty --current-pr-body-file .git/pr-deflection-paid-delivery-worker-body.md` -- passed.
- `python -m py_compile atlas_brain/content_ops_deflection_delivery.py tests/test_atlas_content_ops_deflection_delivery.py && <torch-blocked import check> && python -m pytest tests/test_atlas_content_ops_deflection_delivery.py -q` -- torch-blocked import ok; 7 passed.
- `python -m pytest tests/test_atlas_content_ops_deflection_delivery.py tests/test_smoke_content_ops_deflection_submit_handoff.py tests/test_smoke_content_ops_deflection_portfolio_result_page.py -q` -- 41 passed.
- `python -m py_compile atlas_brain/content_ops_deflection_delivery.py tests/test_atlas_content_ops_deflection_delivery.py && <torch-blocked import + production default URL check> && python -m pytest tests/test_atlas_content_ops_deflection_delivery.py tests/test_smoke_content_ops_deflection_submit_handoff.py tests/test_smoke_content_ops_deflection_portfolio_result_page.py -q` -- torch-blocked import and production default URL ok; 41 passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Delivery service | ~250 |
| Workflow | ~35 |
| Extracted runner enrollment | ~1 |
| Hosted-smoke dotenv opt-out | ~20 |
| Tests | ~220 |
| Plan doc | ~85 |
| **Total** | **~611** |

This exceeds the 400 LOC soft cap because the slice needs the service,
dedicated CI enrollment, and focused failure-branch tests to be useful and
reviewable. The implementation stays on one narrow queue consumer contract.
