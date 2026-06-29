# Content Ops Deflection Launch Preflight Runbook

Use this runbook before public paid Resolution Audit traffic. The launch proof
is not complete until both buyer-facing email surfaces are observed with their
PDF attachments and the emailed hosted result URL opens the expected report.

Track the finished proof on #1921, then link the final artifact back to #1440
and #1386.

## Inputs

Set these locally before the proof:

```bash
export PORTFOLIO_BASE_URL="https://juancanfield.com"
export ATLAS_API_BASE_URL="https://atlas.example.com"
export ATLAS_ADMIN_TOKEN="<operator-token>"
export DATABASE_URL="<production-postgres-dsn>"
export LAUNCH_BUYER_EMAIL="<real-opted-in-buyer-email>"
export LAUNCH_CSV_FILE="<realistic-or-full-volume-support-export.csv>"
```

Use an opted-in buyer address for the live proof. Do not use `@example.com`,
`@test`, seed data, or an operator-only inbox as proof of customer delivery.

## Deployed Config Check

Verify portfolio can submit to ATLAS and send the Snapshot email:

```bash
ATLAS_API_BASE_URL
ATLAS_B2B_SERVICE_TOKEN
GAP_REPORT_NOTIFICATION_RESEND_API_KEY
GAP_REPORT_NOTIFICATION_FROM_EMAIL
```

Verify ATLAS can send the paid report email:

```bash
ATLAS_DEFLECTION_DELIVERY_ENABLED=true
ATLAS_DEFLECTION_DELIVERY_DRY_RUN=false
ATLAS_DEFLECTION_DELIVERY_FROM_EMAIL
ATLAS_DEFLECTION_DELIVERY_RESEND_API_KEY
ATLAS_DEFLECTION_DELIVERY_RESULT_BASE_URL="https://juancanfield.com"
```

`ATLAS_DEFLECTION_DELIVERY_RESULT_URL_TEMPLATE` may replace
`ATLAS_DEFLECTION_DELIVERY_RESULT_BASE_URL`, but it must contain
`{request_id}` and resolve to
`/systems/support-ticket-deflection/results/{request_id}`.

Confirm the paid delivery task is enabled and scheduled:

```bash
curl -fsS "$ATLAS_API_BASE_URL/api/v1/autonomous/?include_disabled=true" \
  -H "Authorization: Bearer $ATLAS_ADMIN_TOKEN" \
  | jq '.tasks[] | select(.name == "content_ops_deflection_report_delivery") | {id, enabled, next_run_at, metadata}'
```

Stop if `enabled` is not true, if `next_run_at` is empty, or if the deployed
delivery config still reports dry-run for the live-send proof.

## Database Gate

Verify the paid-report delivery migrations are present:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -c "
SELECT name
FROM schema_migrations
WHERE name IN (
  '331_content_ops_deflection_report_delivery_email',
  '332_content_ops_deflection_report_deliveries'
)
ORDER BY name;
"
```

Both rows must be present before paid delivery proof. Migration 331 stores the
buyer delivery email on the report; migration 332 creates the paid delivery
queue.

## Snapshot Email And PDF Proof

Submit the CSV through the deployed portfolio intake, not by directly writing
rows. The route must create an ATLAS report request, fetch the free Snapshot,
and call the customer Snapshot sender with the fetched Snapshot.

Proceed only when the received Snapshot email proves all of the following:

- Subject matches the Snapshot intake offer.
- Body says the free Snapshot is ready.
- Body includes the hosted result URL:
  `/systems/support-ticket-deflection/results/<request-id>`.
- A Snapshot PDF is attached.
- The PDF opens and contains summary counts, top repeat questions, customer
  wording, the free answer teaser, and locked rank/count placeholders.
- The PDF excludes source IDs, evidence quotes, raw ticket bodies, paid report markdown, and locked answer bodies.

Stop if the portfolio logs `deflection.record.snapshot_pdf_attachment_skipped`.
An email without the Snapshot PDF is not launch proof.

## Paid Unlock Gate

Complete checkout for the same `request_id`, then confirm the webhook unlocked
the report and queued paid delivery:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -c "
SELECT
  r.account_id,
  r.request_id,
  r.paid,
  r.delivery_email,
  d.delivery_status,
  d.payment_reference
FROM content_ops_deflection_reports r
LEFT JOIN content_ops_deflection_report_deliveries d
  ON d.account_id = r.account_id
 AND d.request_id = r.request_id
WHERE r.request_id = '<request-id>';
"
```

Proceed only if `paid` is true, `delivery_email` is the opted-in buyer email,
and `delivery_status` is `pending`.

## Paid Report Email And PDF Proof

Rehearse the delivery drain first:

```bash
python scripts/send_content_ops_deflection_report_deliveries.py \
  --database-url "$DATABASE_URL" \
  --from-email "$ATLAS_DEFLECTION_DELIVERY_FROM_EMAIL" \
  --result-base-url "$PORTFOLIO_BASE_URL" \
  --limit 1 \
  --json
```

The dry-run JSON must show at least one scanned row, `dry_run` at least 1,
`sent` 0, and `failed` 0. If the dry run scans zero rows, the email path was
not rehearsed.

Only after the dry-run payload is reviewed, send live:

```bash
python scripts/send_content_ops_deflection_report_deliveries.py \
  --database-url "$DATABASE_URL" \
  --from-email "$ATLAS_DEFLECTION_DELIVERY_FROM_EMAIL" \
  --result-base-url "$PORTFOLIO_BASE_URL" \
  --resend-api-key "$ATLAS_DEFLECTION_DELIVERY_RESEND_API_KEY" \
  --limit 1 \
  --send \
  --json
```

Proceed only if the live JSON has `sent` 1 and `failed` 0, and the buyer inbox
receives the paid report email.

The paid email must include key numbers, next actions, ready-to-publish rows
when available, the hosted result URL, and a paid report PDF attachment. A
link-only paid email is not launch proof.

## Paid PDF Shape Check

The paid PDF is the curated/shareable report, not the full evidence archive.
Confirm the attachment:

- Opens as a PDF.
- Includes a Table of contents.
- Includes the actionable report sections from the report model.
- Is not a 600+ page raw evidence dump.
- Keeps ranked question tables capped at 25 rows for PDF readability.
- Keeps question detail blocks capped at 10 questions for PDF readability.
- Points complete source IDs and evidence quotes to the complete evidence export
  JSON on the hosted paid result page.

The complete evidence export belongs on the hosted result page/export surface,
not in the email attachment.

## Hosted URL, Cleanup, And Tracker Closeout

Open the exact URL from each email:

- Before checkout, the Snapshot URL must render the locked/free Snapshot state.
- After checkout, the paid URL must render the unlocked paid report.
- The page must not fall back to demo data for the buyer request.

Before launch, also confirm:

- portfolio cleanup cron is installed and authenticated with `CRON_SECRET`;
- Privacy, Security, Terms, refund, and support-contact links are present or
  explicitly accepted as hand-held beta gaps;
- the final proof artifact is linked on #1921, #1440, and #1386.

If any gate fails, stop launch, record the failed gate on #1921, and rerun this
runbook after the fix lands.
