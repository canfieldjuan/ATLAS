# Deflection Full-Volume Live Proof

Date: 2026-06-14

Issue: #1440

## Result

Partial live proof, not a full funnel pass.

The hosted ATLAS submit path accepted and processed a regenerated near-50 MB
CFPB CSV. The run proved real multipart upload, hosted generation, snapshot
fetch, and locked unpaid artifact behavior at full volume. The full buyer
funnel is still blocked by three live findings:

1. The configured repeat-ticket gate was too high for this regenerated sample:
   expected at least 30,000, got 27,384.
2. The public portfolio result route returned 404 for the generated request.
3. The deployed Stripe webhook rejected the local signing secret with
   `Invalid signature`, so paid unlock and delivery were not proven.

## Input

The raw CSV is not committed. It was regenerated locally from:

```text
/home/juan-canfield/Desktop/Atlas/tmp/faq_scale_stress_20260523/cfpb_50000_source_rows.jsonl
```

Generated upload CSV:

| Field | Value |
|---|---:|
| CSV records written | 40,383 |
| CSV bytes | 52,428,276 |
| SHA-256 | `43130a9a43c2bd821a16c2025694a14a45fc2a914f79cfe5278c88736b749193` |

The line count is not meaningful for this CSV because CFPB narratives include
embedded newlines. The record count above is from the CSV writer.

## Hosted Submit

Command shape:

```bash
python scripts/smoke_content_ops_deflection_submit_handoff.py \
  --csv-file tmp/deflection_full_volume_live_proof_20260614/cfpb_real_upload_under_50mb.csv \
  --company-name "CFPB Public Archive" \
  --contact-email canfieldjuan24@gmail.com \
  --support-platform other \
  --min-uploaded-bytes 50000000 \
  --min-source-row-count 30000 \
  --min-submitted-row-count 30000 \
  --min-generated-questions 30 \
  --min-repeat-ticket-count 30000 \
  --min-top-question-count 5 \
  --timeout 360 \
  --output-result tmp/deflection_full_volume_live_proof_20260614/submit-result.json \
  --json
```

Observed scalar result:

| Check | Value |
|---|---:|
| Submit status | 200 |
| Snapshot status | 200 |
| Unpaid artifact status | 403 |
| Uploaded bytes | 52,428,276 |
| Source rows | 40,383 |
| Submitted rows | 40,383 |
| Truncated rows | 0 |
| Generated questions | 1,659 |
| Repeat-ticket count | 27,384 |
| Top-question count | 5 |
| Elapsed seconds | 64.47 |

Request id:

```text
content-ops-45c06a6950ec4677a214368d6e4dc44f
```

The submit transport and snapshot/artifact probes passed. The command exited
nonzero only because `--min-repeat-ticket-count 30000` was stricter than the
actual hosted result.

## Portfolio Result Page

Command shape:

```bash
python scripts/smoke_content_ops_deflection_portfolio_result_page.py \
  --result-url https://juancanfield.com/services/faq-deflection/results/content-ops-45c06a6950ec4677a214368d6e4dc44f \
  --request-id content-ops-45c06a6950ec4677a214368d6e4dc44f \
  --timeout 120 \
  --output-result tmp/deflection_full_volume_live_proof_20260614/result-page.json \
  --json
```

Observed result:

| Check | Value |
|---|---:|
| Portfolio page status | 404 |
| ATLAS snapshot status | 200 |
| ATLAS unpaid artifact status | 403 |

The ATLAS request exists and remains locked before payment, but the public
portfolio route did not render the result page.

## Paid Unlock

Command shape:

```bash
python scripts/smoke_content_ops_deflection_stripe_paid_unlock.py \
  --request-id content-ops-45c06a6950ec4677a214368d6e4dc44f \
  --timeout 180 \
  --replay-webhook \
  --output-result tmp/deflection_full_volume_live_proof_20260614/paid-unlock.json \
  --json
```

Observed result:

| Check | Value |
|---|---:|
| Artifact before webhook | 403 |
| Stripe webhook status | 400 |
| Error detail probe | `Invalid signature` |
| Artifact after webhook | not run |

The local `ATLAS_SAAS_STRIPE_WEBHOOK_SECRET` does not match the deployed
`atlas-brain.tailc7bd29.ts.net` Stripe webhook secret, or the deployed host is
using a different secret source. Paid unlock and email delivery remain unproven.

## Committed Evidence

The committed fixture summary is:

```text
docs/extraction/validation/fixtures/deflection_full_volume_live_proof_20260614/summary.json
```

The summary intentionally excludes bearer tokens, webhook secrets, raw CFPB
rows, paid report Markdown, PDFs, and email bodies.

## Next Action

Fix or configure the live surfaces in this order:

1. Decide whether the repeat-volume gate should be below the observed 27,384
   repeat-ticket count for regenerated CFPB uploads, or keep 30,000 and require
   a larger/different sample.
2. Restore the public portfolio result route for
   `/services/faq-deflection/results/{request_id}`.
3. Align the local/operator Stripe webhook secret with the deployed
   `atlas-brain` secret, then rerun paid unlock and delivery.
