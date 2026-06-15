# Deflection Portfolio Paid Result Live Proof

Date: 2026-06-15

Issue: #1440

## Result

Pass for the production paid result-page blocker.

The prior full-funnel proof (#1572) proved hosted submit, paid unlock, webhook
replay idempotency, and paid report-email delivery, but production portfolio
still rendered `SNAPSHOT TEMPORARILY UNAVAILABLE` for the paid request. After
atlas-portfolio #309 landed, this rerun proves the buyer-facing result page now
renders the unlocked paid report under real hosted conditions.

This is still CFPB stress/scale proof. It is not Zendesk product-quality
calibration.

## Input

The raw CSV is not committed. It was regenerated locally from:

```text
tmp/faq_scale_stress_20260523/cfpb_50000_source_rows.jsonl
```

Generated upload CSV:

| Field | Value |
|---|---:|
| CSV records written | 41,608 |
| CSV bytes | 52,426,203 |
| SHA-256 | `a00517fec9c029ff9fb1fa4fb84cd2b6885941688673af8c1dc2c5d38b7ac77b` |

The first local regeneration included extra descriptive columns, fit only
34,996 records under the 50 MiB cap, and missed the calibrated repeat-ticket
floor. That run was discarded. The committed proof uses the leaner support-ticket
CSV shape above.

## Hosted Submit

Command shape:

```bash
python scripts/smoke_content_ops_deflection_submit_handoff.py \
  --token "<current-service-token>" \
  --csv-file tmp/deflection_portfolio_paid_result_live_proof_20260615/cfpb_real_upload_lean_under_50mib.csv \
  --company-name "CFPB Public Archive" \
  --support-platform other \
  --volume-gate-profile full-volume-cfpb \
  --timeout 420 \
  --output-result tmp/deflection_portfolio_paid_result_live_proof_20260615/submit-result.json \
  --json
```

Observed result:

| Check | Value |
|---|---:|
| Submit status | 200 |
| Snapshot status | 200 |
| Unpaid artifact status | 403 |
| Uploaded bytes | 52,426,203 |
| Source rows | 41,608 |
| Submitted rows | 41,608 |
| Generated questions | 1,720 |
| Repeat-ticket count | 26,989 |
| Top-question count | 5 |
| Elapsed seconds | 110.19 |

The calibrated `full-volume-cfpb` profile passed.

## Paid Unlock

Command shape:

```bash
python scripts/smoke_content_ops_deflection_stripe_paid_unlock.py \
  --token "<current-service-token>" \
  --webhook-secret "<deployed-matching-webhook-secret>" \
  --request-id "<redacted-paid-request-id>" \
  --timeout 240 \
  --replay-webhook \
  --output-result tmp/deflection_portfolio_paid_result_live_proof_20260615/paid-unlock.json \
  --json
```

Observed result:

| Check | Value |
|---|---:|
| Artifact before webhook | 403 |
| Stripe webhook status | 200 |
| Replay webhook status | 200 |
| Replay payload status | `already_processed` |
| Artifact after webhook | 200 |

## Portfolio Result Page

The smoke harness now has an explicit `--artifact-state unlocked` mode. Locked
mode still validates the free snapshot and unlock CTA; unlocked mode validates
the paid report page, artifact status 200, paid report markers, and absence of
the unavailable/locked states.

Command shape:

```bash
python scripts/smoke_content_ops_deflection_portfolio_result_page.py \
  --token "<current-service-token>" \
  --request-id "<redacted-paid-request-id>" \
  --result-url "https://juancanfield.com/systems/support-ticket-deflection/results/<redacted-paid-request-id>" \
  --artifact-state unlocked \
  --timeout 180 \
  --output-result tmp/deflection_portfolio_paid_result_live_proof_20260615/portfolio-result-page.json \
  --json
```

Observed result:

| Check | Value |
|---|---:|
| Portfolio page status | 200 |
| ATLAS snapshot status | 200 |
| ATLAS paid artifact status | 200 |
| Paid report markers present | true |
| Unavailable state present | false |
| Locked snapshot state present | false |
| Unlock CTA present | false |
| Account id exposed | false |

This closes the #1572 portfolio-page blocker: production no longer renders the
unavailable state for an unlocked paid request.

## Committed Evidence

Sanitized summary:

```text
docs/extraction/validation/fixtures/deflection_portfolio_paid_result_live_proof_20260615/summary.json
```

The summary intentionally excludes bearer tokens, webhook secrets, account id,
operator email, raw CFPB rows, live request id, result URL, paid report Markdown,
email bodies, PDFs, and portfolio HTML.

## Remaining #1440 Scope

Paid report delivery was already proven in #1572. The remaining ambiguity is
snapshot email/PDF: #1572 found no snapshot-email sender surface in Atlas. The
next decision is product/acceptance scope, not another submit or paid-result
rerun: either identify/build a real snapshot delivery surface, or update #1440
acceptance language so it does not ask for a nonexistent snapshot email/PDF.

## Follow-Up

The hosted proof scripts still default to `ATLAS_B2B_JWT` / `ATLAS_TOKEN`, while
the current working credential is stored as `ATLAS_B2B_SERVICE_TOKEN`. This run
used an explicit `--token` override. A future cleanup should add that alias to
the smoke scripts so live-proof commands do not start with a misleading 401.
