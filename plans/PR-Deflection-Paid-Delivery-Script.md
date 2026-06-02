## Why this slice exists

The paid-delivery worker can now consume pending FAQ deflection delivery rows,
but there is no operator entrypoint to run it against Postgres. The next useful
step is a manual script that wires the worker to the existing Resend sender
adapter without adding scheduler semantics yet.

This slice is over the 400 LOC soft cap because the runnable script needs
fail-closed preflight branches, mocked transport tests, dedicated workflow
enrollment, and extracted runner enrollment in the same PR to be safe to use.

## Scope (this PR)

Ownership lane: ai-content-ops/faq-deflection-paid-unlock

Slice phase: Production hardening

1. Add a manual script for sending pending paid FAQ deflection report delivery
   emails from Postgres.
2. Default the script to dry-run mode and require explicit `--send` for live
   Resend sends.
3. Read database, sender, and result-link config from args/env with fail-closed
   validation.
4. Reuse the existing delivery worker and Resend campaign sender adapter.
5. Add focused CLI/preflight tests and enroll them in the dedicated atlas
   delivery workflow plus extracted runner.

### Files touched

- `scripts/send_content_ops_deflection_report_deliveries.py`
- `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_send_content_ops_deflection_report_deliveries.py`
- `plans/PR-Deflection-Paid-Delivery-Script.md`

## Mechanism

The script validates operator inputs, creates an asyncpg pool, constructs a
`DeflectionReportDeliveryConfig`, and calls
`send_pending_deflection_report_deliveries(...)`.

Dry-run is the default. Live sends require `--send`, a configured Resend API
key, a From address, and a result URL destination. The destination may be a
`result_base_url` using the production default path or an explicit
`result_url_template`.

## Intentional

- No cron/scheduler is added; this is a manual operator entrypoint.
- No abandoned checkout or non-buyer nurture flow is added.
- Live sending is opt-in via `--send`; missing Resend config fails before any
  DB connection or send attempt.
- The script reuses the existing Resend adapter instead of adding a new email
  provider integration.

## Deferred

- Future slice: scheduler/cron wiring and row-claiming for concurrent pollers.
- Future slice: provider webhook ingestion for delivered/open/click events.
- Future slice: abandoned checkout follow-up policy and unsubscribe handling.
- Parked hardening: none.

## Verification

- `python -m py_compile scripts/send_content_ops_deflection_report_deliveries.py tests/test_send_content_ops_deflection_report_deliveries.py && python -m pytest tests/test_atlas_content_ops_deflection_delivery.py tests/test_send_content_ops_deflection_report_deliveries.py -q` -- 13 passed.
- `scripts/run_extracted_pipeline_checks.sh` -- run with bash; 2950 passed, 10 skipped.
- Local PR review bundle with the prepared PR body -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Manual script | ~250 |
| Tests | ~185 |
| Workflow/enrollment | ~10 |
| Plan doc | ~80 |
| **Total** | **~525** |

Over the 400 LOC soft cap; justified above because the script, tests, and CI
enrollment are the smallest reviewable operational entrypoint.
