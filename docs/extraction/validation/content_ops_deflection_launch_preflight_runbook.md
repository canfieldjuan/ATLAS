# Content Ops Deflection Launch Preflight Runbook

Use this runbook before public paid Resolution Audit traffic. The launch proof
is not complete until both buyer-facing email surfaces are observed with their
PDF attachments and the emailed hosted result URL opens the expected report.

Track the finished proof on #1921, then link the sanitized proof scorecard back
to #1440 and #1386. Never attach raw live emails, exact result URLs, PDFs, or
artifact JSON to GitHub issues.

## Inputs

Set these locally before the proof:

```bash
export PORTFOLIO_BASE_URL="https://juancanfield.com"
export ATLAS_API_BASE_URL="https://atlas.example.com"
export ATLAS_ADMIN_TOKEN="<operator-token>"
export DATABASE_URL="<production-postgres-dsn>"
export LAUNCH_BUYER_EMAIL="<real-opted-in-buyer-email>"
export LAUNCH_CSV_FILE="<realistic-or-full-volume-support-export.csv>"
export ATLAS_DEFLECTION_DELIVERY_FROM_EMAIL="<paid-report-from-email>"
export ATLAS_DEFLECTION_DELIVERY_RESEND_API_KEY="<paid-report-resend-key>"
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

The Snapshot sender is implemented in the deployed `atlas-portfolio` Next.js
app (`web/src/lib/gap-report-intake.ts`, `sendSnapshotEmail`). The hosted result
URL must be served by that same `atlas-portfolio` deployment at
`/systems/support-ticket-deflection/results/{request_id}`. Do not point this
proof at the legacy in-repo `portfolio-ui` SPA route
`/services/faq-deflection/results/{request_id}`.

Verify ATLAS has the paid report email config present, but keep live scheduling
off until after queue isolation and PDF render validation:

```bash
ATLAS_DEFLECTION_DELIVERY_ENABLED=false
ATLAS_DEFLECTION_DELIVERY_DRY_RUN=true
ATLAS_DEFLECTION_DELIVERY_FROM_EMAIL
ATLAS_DEFLECTION_DELIVERY_RESEND_API_KEY
ATLAS_DEFLECTION_DELIVERY_RESULT_BASE_URL="https://juancanfield.com"
```

`ATLAS_DEFLECTION_DELIVERY_RESULT_URL_TEMPLATE` may replace
`ATLAS_DEFLECTION_DELIVERY_RESULT_BASE_URL`, but it must contain
`{request_id}` and resolve to
`/systems/support-ticket-deflection/results/{request_id}`.

Confirm the paid delivery task will not auto-send before the rehearsal. Before
checkout, either the task must be disabled or deployed delivery must still be in
dry-run:

```bash
curl -fsS "$ATLAS_API_BASE_URL/api/v1/autonomous/?include_disabled=true" \
  -H "Authorization: Bearer $ATLAS_ADMIN_TOKEN" \
  | jq '.tasks[] | select(.name == "content_ops_deflection_report_delivery") | {id, enabled, next_run_at, metadata}'
```

Stop if the scheduler is enabled and live-send config is active before the
queue-only rehearsal. That can claim the pending delivery row as soon as Stripe
queues it and bypass the dry-run proof.

## Database Gate

Verify the paid-report delivery migrations are present:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -c "
SELECT name
FROM schema_migrations
WHERE name IN (
  '331_content_ops_deflection_report_delivery_email',
  '332_content_ops_deflection_report_deliveries',
  '342_content_ops_deflection_checkout_authorization'
)
ORDER BY name;
"
```

All three rows must be present before paid delivery proof. Migration 331 stores
the buyer delivery email on the report; migration 332 creates the paid delivery
queue; migration 342 adds the checkout authorization columns that the real
Stripe unlock path reads and writes.

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

Complete a real Stripe Checkout for the same `request_id`, then confirm the
webhook unlocked the report and queued paid delivery. Do not replay a synthetic webhook for launch proof; replay proves delivery wiring but can skip the real Checkout price authorization path.

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
and `delivery_status` is `pending`. Record the returned `account_id` and
`request_id` for the scoped manual delivery proof:

```bash
export LAUNCH_ACCOUNT_ID="<account-id-from-query>"
export LAUNCH_REQUEST_ID="<request-id>"
```

## Paid Report Email And PDF Proof

The manual drain CLI accepts an account/request scope for launch proof. Before
running it, prove the target row is claimable:

```bash
psql "$DATABASE_URL" \
  -v ON_ERROR_STOP=1 \
  -v account_id="$LAUNCH_ACCOUNT_ID" \
  -v request_id="$LAUNCH_REQUEST_ID" \
  -c "
WITH target AS (
  SELECT account_id, request_id, created_at, updated_at, delivery_status
  FROM content_ops_deflection_report_deliveries
  WHERE account_id = :'account_id'
    AND request_id = :'request_id'
)
SELECT
  count(*) FILTER (
    WHERE delivery_status = 'pending'
  ) AS target_pending_rows,
  count(*) FILTER (
    WHERE delivery_status = 'pending'
       OR (
         delivery_status = 'sending'
         AND updated_at < NOW() - INTERVAL '15 minutes'
       )
  ) AS target_claimable_rows,
  min(created_at) FILTER (
    WHERE delivery_status = 'pending'
       OR (
         delivery_status = 'sending'
         AND updated_at < NOW() - INTERVAL '15 minutes'
       )
  ) AS oldest_claimable_at
FROM target;
"
```

Proceed only if `target_claimable_rows` is 1 and `target_pending_rows` is 1.
If the target row is already `sending`, stop unless this is an intentional
stale reclaim rehearsal; the first launch proof send should start from
`pending`.

Run the scoped delivery drain in queue-only dry-run mode:

```bash
python scripts/send_content_ops_deflection_report_deliveries.py \
  --database-url "$DATABASE_URL" \
  --from-email "$ATLAS_DEFLECTION_DELIVERY_FROM_EMAIL" \
  --result-base-url "$PORTFOLIO_BASE_URL" \
  --account-id "$LAUNCH_ACCOUNT_ID" \
  --request-id "$LAUNCH_REQUEST_ID" \
  --limit 1 \
  --json
```

The dry-run JSON must show at least one scanned row, `dry_run` at least 1,
`sent` 0, and `failed` 0. This proves queue selection and paid/email gating
only. It does not render the PDF or build the email body.

Before live send, render the paid PDF from the target artifact with the real
renderer:

```bash
export PREFLIGHT_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/deflection-launch-preflight.XXXXXX")"
trap 'rm -rf "$PREFLIGHT_TMP_DIR"' EXIT

psql "$DATABASE_URL" \
  -v ON_ERROR_STOP=1 \
  -v account_id="$LAUNCH_ACCOUNT_ID" \
  -v request_id="$LAUNCH_REQUEST_ID" \
  -v buyer_email="$LAUNCH_BUYER_EMAIL" \
  -At -c "
SELECT artifact::text
FROM content_ops_deflection_reports
WHERE account_id = :'account_id'
  AND request_id = :'request_id'
  AND paid IS TRUE
  AND COALESCE(delivery_email, '') = :'buyer_email';
" > "$PREFLIGHT_TMP_DIR/paid-artifact.json"

python - <<'PY'
import json
import os
from pathlib import Path
from atlas_brain.deflection_pdf_renderer import render_deflection_full_report_pdf

tmp_dir = Path(os.environ["PREFLIGHT_TMP_DIR"])
artifact_path = tmp_dir / "paid-artifact.json"
pdf_path = tmp_dir / "paid-report.pdf"
artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
pdf_bytes = render_deflection_full_report_pdf(artifact)
assert pdf_bytes.startswith(b"%PDF-")
assert len(pdf_bytes) > 1000
pdf_path.write_bytes(pdf_bytes)
print(f"rendered {pdf_path} ({len(pdf_bytes)} bytes)")
PY
```

Stop if rendering fails or the artifact query returns no row. The first exercise
of paid PDF rendering must not be the live buyer send. The files in
`PREFLIGHT_TMP_DIR` contain paid buyer report data; do not commit, upload, or
link them. The shell `trap` removes them on exit; if the shell is interrupted,
run `rm -rf "$PREFLIGHT_TMP_DIR"` before closeout.

Only after the scoped dry-run and local PDF render validation pass, rerun the
target claimability SQL above immediately before live send. Proceed only if it
still shows `target_claimable_rows` 1 and `target_pending_rows` 1. Then run one
manual live send for the scoped proof row:

```bash
ATLAS_DEFLECTION_DELIVERY_ENABLED=true
ATLAS_DEFLECTION_DELIVERY_DRY_RUN=false

python scripts/send_content_ops_deflection_report_deliveries.py \
  --database-url "$DATABASE_URL" \
  --from-email "$ATLAS_DEFLECTION_DELIVERY_FROM_EMAIL" \
  --result-base-url "$PORTFOLIO_BASE_URL" \
  --resend-api-key "$ATLAS_DEFLECTION_DELIVERY_RESEND_API_KEY" \
  --account-id "$LAUNCH_ACCOUNT_ID" \
  --request-id "$LAUNCH_REQUEST_ID" \
  --limit 1 \
  --send \
  --json
```

Proceed only if the live JSON has `sent` 1 and `failed` 0, and the buyer inbox
receives the paid report email.

Before launch, deploy or restart ATLAS with the hosted scheduler configured for
live paid delivery, not only the one-off manual CLI:

```bash
ATLAS_DEFLECTION_DELIVERY_ENABLED=true
ATLAS_DEFLECTION_DELIVERY_DRY_RUN=false
```

First, from the deployed ATLAS runtime after deploy/restart, verify the
scheduler-owned URL config builds the buyer result URL. This must use the same
environment as the autonomous scheduler, not local CLI flags:

```bash
python - <<'PY'
from urllib.parse import urlparse

from atlas_brain.config import settings
from atlas_brain.content_ops_deflection_delivery import (
    DeflectionReportDeliveryConfig,
    deflection_report_result_url,
)

cfg = settings.deflection_delivery
url = deflection_report_result_url(
    request_id="preflight-url-check",
    config=DeflectionReportDeliveryConfig(
        from_email=str(cfg.from_email or "").strip(),
        result_base_url=str(cfg.result_base_url or "").strip(),
        result_url_template=str(cfg.result_url_template or "").strip(),
        subject=str(cfg.subject or "").strip(),
    ),
)
parsed = urlparse(url)
expected_prefix = (
    "/systems/support-ticket-deflection/results/preflight-url-check"
)
if parsed.netloc != urlparse("https://juancanfield.com").netloc:
    raise SystemExit(f"deployed result URL host mismatch: {url}")
if parsed.scheme != "https":
    raise SystemExit(f"deployed result URL scheme mismatch: {url}")
if parsed.path != expected_prefix:
    raise SystemExit(f"deployed result URL path mismatch: {url}")
print(url)
PY
```

Then, from the operator shell, verify the deployed scheduler is enabled and
scheduled:

```bash
curl -fsS "$ATLAS_API_BASE_URL/api/v1/autonomous/status/summary" \
  -H "Authorization: Bearer $ATLAS_ADMIN_TOKEN" \
  | jq -e 'select(.running == true and .scheduled_count > 0)'

curl -fsS "$ATLAS_API_BASE_URL/api/v1/autonomous/?include_disabled=true" \
  -H "Authorization: Bearer $ATLAS_ADMIN_TOKEN" \
  | jq '.tasks[] | select(.name == "content_ops_deflection_report_delivery") | {id, enabled, next_run_at, metadata}'
```

Proceed only if the scheduler summary reports `running` true with at least one
scheduled task, the delivery task row is `enabled` with a `next_run_at`, and the
deployed runtime URL probe prints a
`https://juancanfield.com/systems/support-ticket-deflection/results/preflight-url-check`
URL. Run the URL probe inside the deployed ATLAS runtime after deploy/restart;
do not satisfy it with the local CLI `--result-base-url`. Metadata alone does
not prove the scheduler loop is running, the deployed URL config, or the live
dry-run setting.

Before probing the hosted scheduler, prove there are no claimable paid delivery
rows left; the manual proof row should now be delivered, and the earlier
isolation gate proved no other row was claimable:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -c "
WITH claimable AS (
  SELECT account_id, request_id
  FROM content_ops_deflection_report_deliveries
  WHERE delivery_status = 'pending'
     OR (
       delivery_status = 'sending'
       AND updated_at < NOW() - INTERVAL '15 minutes'
     )
)
SELECT count(*) AS claimable_rows FROM claimable;
"
```

Proceed only if `claimable_rows` is 0. If any row is claimable, stop and drain
or quarantine it before running the autonomous task probe.

Now run the deployed task without metadata overrides and inspect the completed
execution result. Do not pass `{"dry_run": false}` in the request body; that
would prove only an override, not the hosted setting.

```bash
export TASK_ID="$(
  curl -fsS "$ATLAS_API_BASE_URL/api/v1/autonomous/?include_disabled=true" \
    -H "Authorization: Bearer $ATLAS_ADMIN_TOKEN" \
    | jq -er '.tasks[]
      | select(.name == "content_ops_deflection_report_delivery")
      | select(.enabled == true and .next_run_at != null)
      | .id'
)"

export RUN_ID="$(
  curl -fsS -X POST \
    "$ATLAS_API_BASE_URL/api/v1/autonomous/content_ops_deflection_report_delivery/run" \
    -H "Authorization: Bearer $ATLAS_ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{}' \
    | jq -er '.execution_id'
)"

python - <<'PY'
import ast
import json
import os
import time
import urllib.request

base_url = os.environ["ATLAS_API_BASE_URL"].rstrip("/")
task_id = os.environ["TASK_ID"]
run_id = os.environ["RUN_ID"]
token = os.environ["ATLAS_ADMIN_TOKEN"]

for _ in range(12):
    req = urllib.request.Request(
        f"{base_url}/api/v1/autonomous/{task_id}/executions?limit=5",
        headers={"Authorization": f"Bearer {token}"},
    )
    payload = json.loads(urllib.request.urlopen(req, timeout=30).read())
    execution = next(
        (item for item in payload.get("executions", []) if item.get("id") == run_id),
        None,
    )
    if execution and execution.get("status") != "running":
        break
    time.sleep(5)
else:
    raise SystemExit("scheduler proof execution did not finish")

if execution is None:
    raise SystemExit("scheduler proof execution missing")
if execution.get("status") != "completed":
    raise SystemExit(f"scheduler proof execution failed: {execution!r}")

# HeadlessRunner persists builtin dict results with str(result), not JSON.
result = ast.literal_eval(str(execution.get("result_text") or "{}"))
expected = {
    "dry_run_enabled": False,
    "scanned": 0,
    "sent": 0,
    "failed": 0,
}
for key, value in expected.items():
    if result.get(key) != value:
        raise SystemExit(f"scheduler proof mismatch for {key}: {result!r}")
print(json.dumps({"execution_id": run_id, "result": result}, sort_keys=True))
PY
```

Proceed only if the execution result proves `dry_run_enabled` is false with zero
claimable work scanned/sent/failed. Stop if the task is disabled, has no
`next_run_at`, the scheduler summary is not running, the execution is
missing/failed/running, `result_text` cannot be parsed as the builtin task
result repr, `dry_run_enabled` is true, or any row is scanned. The manual one-off email is not enough to launch public paid delivery.

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
- the local paid artifact/PDF temp directory has been removed:
  `rm -rf "${PREFLIGHT_TMP_DIR:-}"`;
- a sanitized proof scorecard, not raw live artifacts, is linked on #1921, #1440, and #1386.

Build the scorecard from manually sanitized proof notes only. Exclude raw
emails, exact hosted result URLs, raw PDFs, paid artifact JSON, source IDs,
ticket evidence, buyer email addresses, request IDs, checkout session IDs,
Resend message IDs, and Stripe event IDs. Raw live bundles stay uncommitted.
Before linking anything, run the redaction gate:

```bash
python scripts/check_deflection_full_report_proof_bundle.py \
  tmp/deflection-launch-preflight-scorecard \
  --output tmp/deflection-launch-preflight-scorecard/redaction-check.json \
  --pretty
```

The redaction gate must fail on request IDs, exact result URLs, buyer emails,
checkout session IDs, payment intent IDs, Resend provider message IDs, Stripe
event IDs, raw evidence, source ID lists, private notes, local paths, unreadable
artifacts, and PDFs. Link only the sanitized scorecard plus the passing
redaction-check output. Stop if the redaction gate fails.

If any gate fails, stop launch, record the failed gate on #1921, and rerun this
runbook after the fix lands.
