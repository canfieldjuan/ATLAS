# PR-Deflection-Paid-Funnel-Incidents

## Why this slice exists

#1386 still has an observability gap after the refund/dispute mechanics landed:
the paid funnel can fail closed correctly, but the operator signal is spread
across ad hoc log strings. The issue explicitly calls out operational
blindness for paid-but-not-unlocked, submit-fail, delivery-fail, and reversal
paths.

This slice gives the ATLAS side a stable incident record shape for the
highest-risk paid-funnel failures already handled in code. It does not create a
new notification transport; it makes the failure records machine-readable and
test-locked so a later alert-sink slice can route them without re-discovering
every failure branch.

This exceeds the 400 LOC soft cap because the slice deliberately proves each
incident branch with focused negative fixtures, and it also carries the #1436
plan archive required by merged-PR teardown.

## Scope (this PR)

Ownership lane: go-live-deflection-cleanup
Slice phase: Production hardening

1. Add a small `content_ops_deflection_incidents` helper that emits bounded,
   JSON-shaped `DEFLECTION_PAID_FUNNEL_INCIDENT` log records with stable
   `incident_type`, `severity`, `account_id`, `request_id`, `event_type`, and
   Stripe object/session context.
2. Emit incidents for ATLAS paid-funnel failure points that can otherwise leave
   a buyer paid-but-locked, relocked-without-operator-context, or paid-report
   delivery suppressed/failed.
3. Add focused tests that prove the incident detector fires for the payment
   and delivery branches this slice wires.
4. Archive the now-merged #1436 plan doc as teardown housekeeping in the same
   branch.

### Review Contract

- Acceptance criteria:
  - [ ] A missing report after a paid Checkout event emits a
        `paid_report_missing_after_payment` incident before returning the
        existing retryable 409.
  - [ ] Amount/currency mismatch on a deflection Checkout emits a
        `paid_report_checkout_terms_mismatch` incident while preserving the
        current no-unlock behavior.
  - [ ] A revocation event that maps to deflection metadata but misses the
        report emits a `paid_report_revocation_missed_report` incident.
  - [ ] Delivery worker failure branches emit incidents for missing delivery
        email, no-longer-sendable rows, and send exceptions without leaking
        buyer email or artifact text.
  - [ ] Incident records are bounded, structured JSON after a stable marker,
        and tests parse that payload rather than matching only free-form text.
- Affected surfaces: Stripe webhook billing logs, paid report delivery worker
  logs, focused webhook/delivery tests, plan archive housekeeping.
- Risk areas: PII leakage, noisy non-deflection events, changing Stripe retry
  semantics, changing delivery queue state transitions.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R8, R10.

### Files touched

- `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml`
- `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml`
- `atlas_brain/api/billing.py`
- `atlas_brain/content_ops_deflection_delivery.py`
- `atlas_brain/content_ops_deflection_incidents.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Paid-Funnel-Incidents.md`
- `plans/archive/PR-Deflection-Refund-Dispute-Revocation.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_content_ops_deflection_incidents.py`

## Mechanism

The incident helper accepts a narrow set of keyword fields, removes empty
values, bounds every string field, JSON-encodes the payload with sorted keys,
and logs through the caller's existing logger at error/warning/info severity.
The stable prefix is the sink contract:

```text
DEFLECTION_PAID_FUNNEL_INCIDENT {"incident_type": "...", ...}
```

Billing and delivery keep their existing control flow. The slice adds incident
emits immediately before the existing failure return/raise/continue points, so
Stripe retry behavior, paid-gate state, and delivery row transitions do not
change.

## Intentional

- No ntfy/webhook/email alert transport is added here. The repo has broad
  alert infrastructure, but no dedicated paid-funnel alert sink contract; this
  slice creates the stable incident record the transport can consume later.
- Non-deflection refunds/disputes remain info-only and do not emit paid-funnel
  incidents. Expected Stripe noise should not page the operator.
- The helper accepts only explicit fields and caps string lengths instead of
  serializing arbitrary Stripe/session objects, to avoid leaking PII or raw
  ticket artifacts.

## Deferred

- #1386 follow-up: route `DEFLECTION_PAID_FUNNEL_INCIDENT` records into the
  production alert sink once the sink contract is selected.
- #1386 follow-up: add dispute-closed/manual-restore incident handling after
  the restore path exists.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q`
  - Result: `39 passed, 1 warning in 3.00s`.
- `python -m pytest tests/test_atlas_content_ops_deflection_delivery.py -q`
  - Result: `12 passed in 0.21s`.
- `python -m pytest tests/test_content_ops_deflection_incidents.py -q`
  - Result: `1 passed in 0.04s`.
- `python -m py_compile atlas_brain/api/billing.py atlas_brain/content_ops_deflection_delivery.py atlas_brain/content_ops_deflection_incidents.py`
  - Result: passed.
- `git diff --check`
  - Result: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml` | 5 |
| `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml` | 4 |
| `atlas_brain/api/billing.py` | 35 |
| `atlas_brain/content_ops_deflection_delivery.py` | 49 |
| `atlas_brain/content_ops_deflection_incidents.py` | 55 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Paid-Funnel-Incidents.md` | 140 |
| `plans/archive/PR-Deflection-Refund-Dispute-Revocation.md` | 0 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 125 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 47 |
| `tests/test_content_ops_deflection_incidents.py` | 50 |
| **Total** | **514** |
