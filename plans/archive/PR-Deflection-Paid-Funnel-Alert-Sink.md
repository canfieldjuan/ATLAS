# PR-Deflection-Paid-Funnel-Alert-Sink

## Why this slice exists

#1386 still has one operational gap after #1445: paid-funnel failures now emit
stable `DEFLECTION_PAID_FUNNEL_INCIDENT` log records, but those records are not
routed into any notification surface. An operator still has to discover them by
reading logs.

This slice connects those records to the existing centralized alert manager so
configured callbacks can surface paid-but-locked, delivery-suppressed, and
revocation-miss incidents without changing money or delivery state transitions.

## Scope (this PR)

Ownership lane: go-live-deflection-cleanup
Slice phase: Production hardening

1. Add a paid-funnel incident alert event shape that carries only the bounded,
   explicit incident fields already produced by #1445.
2. Register a default centralized alert rule for paid-funnel incidents with no
   cooldown, so every high-risk paid failure can notify when alerts are enabled.
3. Route the existing #1445 incident call sites through a best-effort async alert
   dispatch while preserving the existing structured log record even when alert
   delivery is disabled or fails.
4. Add focused tests for alert dispatch, alert-failure fallback, and the existing
   billing/delivery incident branches.

### Review Contract

- Acceptance criteria:
  - [ ] Existing `DEFLECTION_PAID_FUNNEL_INCIDENT` logs still emit for payment,
        revocation, and delivery failures.
  - [ ] Enabled alerts receive a `deflection_paid_funnel_incident` event with
        bounded metadata.
  - [ ] Alert failures/timeouts are best-effort and bounded so slow sinks do not
        hold the Stripe webhook or delivery worker indefinitely.
  - [ ] The alert message names incident type, severity, account, and request
        without raw tickets, evidence, buyer email, or provider error bodies.
  - [ ] Tests cover successful routing, failures, and timeouts.
- Affected surfaces: incident helper, alert event/rule formatting, Stripe and
  delivery incident calls, focused tests, delivery/Stripe CI enrollment.
- Risk areas: PII leakage, alert spam/cooldown behavior, money-path retry
  semantics, and alert dependency failures blocking checkout/delivery.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R8.

### Files touched

- `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml`
- `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml`
- `atlas_brain/alerts/__init__.py`
- `atlas_brain/alerts/events.py`
- `atlas_brain/alerts/manager.py`
- `atlas_brain/alerts/rules.py`
- `atlas_brain/api/billing.py`
- `atlas_brain/content_ops_deflection_delivery.py`
- `atlas_brain/content_ops_deflection_incidents.py`
- `plans/PR-Deflection-Paid-Funnel-Alert-Sink.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_alerts.py`
- `tests/test_content_ops_deflection_incidents.py`

## Mechanism

`emit_deflection_paid_funnel_incident` keeps the bounded JSON log path and
returns the payload it logged. A new async wrapper logs first, then creates a
paid-funnel alert event from that payload and calls
`AlertManager.process_event` with a short timeout. The wrapper catches failures
and timeouts so alert delivery cannot hold a payment or delivery branch on a
slow external sink.

The alert manager gets a default `deflection_paid_funnel_incident` rule. The
rule matches the new event type with zero cooldown and formats a short operator
message from the bounded incident fields. Existing alert callbacks decide
whether that becomes log-only, ntfy, webhook, or persisted alert output.

## Intentional

- No new alert transport or secret is added; this slice uses the existing
  centralized alert manager and configured callbacks.
- Alerts are routed after the structured log record is emitted, so the log
  remains the fallback sink if `settings.alerts.enabled` is false.
- Dispatch is best-effort and bounded. Alert-manager, persistence, callback, or
  ntfy/webhook failures are operator-observable logs, not long-running
  checkout/delivery blockers.

## Deferred

- #1386 follow-up: dispute-closed/manual-restore incident handling after a
  restore path exists.
- #1386 follow-up: scrub or omit provider error metadata before any future
  transport routes full incident metadata off-box.

Parked hardening: none.

## Verification

- Incident helper tests: `4 passed in 0.18s`.
- Alert tests: `73 passed in 0.23s`.
- Stripe-paid focused tests: `39 passed, 1 warning in 2.96s`.
- Delivery focused tests: `12 passed in 0.43s`.
- Delivery workflow command set: `105 passed, 1 warning in 2.78s`.
- Stripe-paid workflow command set: `177 passed, 1 warning in 3.10s`.
- Extracted pipeline runner: `3648 passed, 10 skipped, 1 warning in 55.17s`.
- Python compile check for touched runtime modules: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml` | 5 |
| `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml` | 6 |
| `atlas_brain/alerts/__init__.py` | 2 |
| `atlas_brain/alerts/events.py` | 43 |
| `atlas_brain/alerts/manager.py` | 13 |
| `atlas_brain/alerts/rules.py` | 3 |
| `atlas_brain/api/billing.py` | 8 |
| `atlas_brain/content_ops_deflection_delivery.py` | 14 |
| `atlas_brain/content_ops_deflection_incidents.py` | 38 |
| `plans/PR-Deflection-Paid-Funnel-Alert-Sink.md` | 124 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_alerts.py` | 25 |
| `tests/test_content_ops_deflection_incidents.py` | 113 |
| **Total** | **395** |
