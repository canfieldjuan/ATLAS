# Deflection Full-Funnel Paid Delivery Proof

Date: 2026-06-15

Issue: #1440

## Result

Partial full-funnel pass.

The hosted revenue path is now proven for a fresh full-volume CFPB stress run:

1. Hosted multipart intake accepted the near-50 MiB CSV.
2. The unpaid artifact was locked before payment.
3. A signed Stripe `checkout.session.completed` webhook unlocked the artifact.
4. Webhook replay was idempotent.
5. The hosted deflection delivery task sent one report email with no failures.

The buyer-facing portfolio result page is still not proven. The canonical
production URL returns HTTP 200, and the ATLAS snapshot/artifact endpoints both
return 200 for the same paid request, but the portfolio page renders
`SNAPSHOT TEMPORARILY UNAVAILABLE` instead of the result markers. That remains
the next live blocker for #1440.

Follow-up on 2026-06-15: atlas-portfolio #309 fixed the paid-artifact fetch
timeout, and `deflection_portfolio_paid_result_live_proof_2026-06-15.md` proves
production now renders the unlocked paid report page for a fresh paid
full-volume request. The portfolio-page blocker recorded here is closed by that
follow-up proof.

Snapshot email/PDF delivery is also not independently observed in this proof.
During this slice, no snapshot-email sender surface was found in the Atlas repo;
the proven email send is the paid full-report delivery.

## Input

The raw CSV is not committed. It was regenerated locally from:

```text
tmp/faq_scale_stress_20260523/cfpb_50000_source_rows.jsonl
```

Generated upload CSV:

| Field | Value |
|---|---:|
| CSV records written | 42,646 |
| CSV bytes | 52,426,845 |
| SHA-256 | `e812f4eb6c41f2d81cbb852f9e9898aec08703e02aacdd17d77b21ef9aa69bde` |

Source role: CFPB remains stress/scale evidence. It is not Zendesk
product-quality calibration.

## Hosted Submit

Command shape:

```bash
python scripts/smoke_content_ops_deflection_submit_handoff.py \
  --csv-file tmp/deflection_full_funnel_paid_delivery_proof_20260615/cfpb_real_upload_under_50mib.csv \
  --company-name "CFPB Public Archive" \
  --contact-email "<contact-email>" \
  --support-platform other \
  --volume-gate-profile full-volume-cfpb \
  --timeout 360 \
  --output-result tmp/deflection_full_funnel_paid_delivery_proof_20260615/submit-result.json \
  --json
```

Observed result:

| Check | Value |
|---|---:|
| Submit status | 200 |
| Snapshot status | 200 |
| Unpaid artifact status | 403 |
| Uploaded bytes | 52,426,845 |
| Source rows | 42,646 |
| Submitted rows | 42,646 |
| Generated questions | 1,757 |
| Repeat-ticket count | 29,106 |
| Top-question count | 5 |
| Elapsed seconds | 60.80 |

Request id:

```text
<redacted-paid-request-id>
```

The calibrated `full-volume-cfpb` profile passed.

## Paid Unlock

The first rerun against the old root `.env` webhook secret still failed with
`Invalid signature`. The deployed host matched the 70-character webhook secret
present in the portfolio env/old backend backup, not the 148-character root
`.env` value. No secret value is committed.

Command shape:

```bash
python scripts/smoke_content_ops_deflection_stripe_paid_unlock.py \
  --request-id "<redacted-paid-request-id>" \
  --timeout 180 \
  --replay-webhook \
  --output-result tmp/deflection_full_funnel_paid_delivery_proof_20260615/paid-unlock-fresh-request.json \
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

## Report Delivery

The hosted autonomous delivery task was enabled and running:

```text
content_ops_deflection_report_delivery
```

Manual trigger:

```bash
POST /api/v1/autonomous/content_ops_deflection_report_delivery/run
```

Observed execution:

| Check | Value |
|---|---:|
| Execution id | `5af4e50a-7044-4445-bc38-72ed3e9231d5` |
| Task status | `completed` |
| Scanned | 1 |
| Sent | 1 |
| Failed | 0 |
| Dry run | 0 |
| Duration | 16,134 ms |

This proves the paid report-email delivery path for the fresh full-volume
request.

## Portfolio Result Page

Command shape:

```bash
python scripts/smoke_content_ops_deflection_portfolio_result_page.py \
  --result-url "<redacted-paid-result-url>" \
  --request-id "<redacted-paid-request-id>" \
  --timeout 120 \
  --output-result tmp/deflection_full_funnel_paid_delivery_proof_20260615/portfolio-result-page-fresh.json \
  --json
```

Observed result:

| Check | Value |
|---|---:|
| Portfolio page status | 200 |
| ATLAS snapshot status | 200 |
| ATLAS paid artifact status | 200 |
| Page rendered unavailable state | true |
| Required result markers present | false |

Page text began:

```text
Support Ticket Deflection Report: Cut Repeat Support Tickets SNAPSHOT TEMPORARILY UNAVAILABLE
```

This is the remaining product blocker: production portfolio can reach a public
route but still cannot render the paid result from the ATLAS data that exists.

## Committed Evidence

Sanitized summary:

```text
docs/extraction/validation/fixtures/deflection_full_funnel_paid_delivery_proof_20260615/summary.json
```

The summary intentionally excludes bearer tokens, webhook secrets, raw CFPB
rows, paid report Markdown, PDFs, email bodies, and portfolio HTML.

## Server-Side Invalidation Proof

After review identified the redacted live request id as a leaked capability,
the request was invalidated server-side through the deployed Stripe revocation
path. The summary binds this check to the leaked value by SHA-256 only:

```text
1ebb65093faf3823d2b1413d2b34db28f786908885406acc48846f224ac1011d
```

Observed revocation proof:

| Check | Value |
|---|---:|
| Revocation webhook status | 200 |
| Artifact status after invalidation | 403 |
| Artifact locked after invalidation | true |
| Checked at | 2026-06-15T13:55:05Z |

No request id, result URL, token, or report body is committed.

## Next Action

1. Fix portfolio production snapshot/artifact loading for the canonical paid
   result URL, then rerun the result-page smoke against
   a new paid request.
2. Decide whether snapshot email/PDF is a real product surface. If yes, add or
   identify the sender path and prove it under full-volume conditions; if no,
   update #1440 acceptance language so it does not keep asking for a nonexistent
   delivery surface.
3. Align local operator env naming so future paid smokes use the deployed
   Stripe webhook secret source, not the stale root `.env` value.
