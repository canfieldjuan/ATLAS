# Content Ops Deflection Delta Go-Live Runbook

Use this runbook when monthly Report Delta delivery is ready to move from
checked code to a live buyer-facing cron. Do not run the live-send section until
the dry-run rehearsal has a clean execution payload.

## Inputs

Set these locally before running the checks:

```bash
export ATLAS_API_BASE_URL="https://atlas.example.com"
export ATLAS_ADMIN_TOKEN="<operator-token>"
export DATABASE_URL="<production-postgres-dsn>"
```

The buyer prerequisite is a real paid account whose current paid deflection
report has a delivery email and at least one earlier paid baseline report.
Without that current-plus-baseline pair there is no customer-safe delta to send.
The delivery address must be a real paying customer address, not a seed,
example, test, or operator inbox.

## Migration Check

The delivery queue table comes from migration 341. Verify it is recorded before
enabling the monthly task:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -c "
SELECT name
FROM schema_migrations
WHERE name = '341_content_ops_deflection_delta_deliveries';
"
```

If the query returns no row, deploy/restart the brain with normal Atlas
migration execution first, then rerun this check.

## Paid Pair Check

Confirm there is a deliverable current paid report with an earlier paid
baseline. The query deliberately surfaces the delivery addresses because a
count alone can hide test/operator data:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -c "
WITH paid_reports AS (
  SELECT
    account_id,
    request_id,
    delivery_email,
    created_at,
    row_number() OVER (
      PARTITION BY account_id
      ORDER BY created_at DESC, request_id DESC
    ) AS report_rank,
    count(*) OVER (PARTITION BY account_id) AS paid_report_count
  FROM content_ops_deflection_reports
  WHERE paid IS TRUE
),
current_reports AS (
  SELECT *
  FROM paid_reports
  WHERE report_rank = 1
    AND COALESCE(delivery_email, '') <> ''
)
SELECT
  current_reports.account_id,
  current_reports.request_id AS current_request_id,
  current_reports.delivery_email AS current_delivery_email,
  current_reports.paid_report_count,
  count(baseline.request_id) AS baseline_report_count,
  array_agg(DISTINCT COALESCE(NULLIF(baseline.delivery_email, ''), '<missing>'))
    AS baseline_delivery_email_sample,
  bool_or(
    lower(COALESCE(current_reports.delivery_email, '')) LIKE '%@example.com'
    OR lower(COALESCE(current_reports.delivery_email, '')) LIKE '%@test%'
    OR lower(COALESCE(baseline.delivery_email, '')) LIKE '%@example.com'
    OR lower(COALESCE(baseline.delivery_email, '')) LIKE '%@test%'
  ) AS has_reserved_test_email
FROM current_reports
JOIN paid_reports AS baseline
  ON baseline.account_id = current_reports.account_id
 AND baseline.created_at < current_reports.created_at
GROUP BY
  current_reports.account_id,
  current_reports.request_id,
  current_reports.delivery_email,
  current_reports.paid_report_count
HAVING count(baseline.request_id) >= 1
ORDER BY current_reports.paid_report_count DESC
LIMIT 10;
"
```

If no account appears, stop. If `has_reserved_test_email` is true, or if the
surfaced addresses are `@example.com`, `@test`, seeds, your own inbox, or any
other operator/test address, stop. The go-live check passes only when every
delivery address you will send to is confirmed as a real paying customer. Copy
the opted-in row's `account_id` and `current_request_id`; the manual rehearsal
and live-send commands must use those exact values as `target_account_id` and
`current_request_id`.

## Dry-Run Activation

Start with generation enabled and keep global delivery settings otherwise
unchanged. `ATLAS_DEFLECTION_DELIVERY_DRY_RUN` is shared with the already-live
paid report delivery drain, so do not flip that global flag for the delta
rehearsal. Use the per-run `delivery_dry_run` override below instead.

```bash
ATLAS_DEFLECTION_DELTA_ENABLED=true
ATLAS_DEFLECTION_DELTA_CRON_EXPRESSION="0 8 1 * *"
ATLAS_DEFLECTION_DELIVERY_FROM_EMAIL="reports@example.com"
```

Restart the brain, then verify the task row is enabled and scheduled:

```bash
curl -fsS "$ATLAS_API_BASE_URL/api/v1/autonomous/?include_disabled=true" \
  -H "Authorization: Bearer $ATLAS_ADMIN_TOKEN" \
  | jq '.tasks[] | select(.name == "content_ops_deflection_delta_automation") | {id, enabled, cron_expression, next_run_at, metadata}'
```

Run one manual rehearsal. The `delivery_dry_run` override must stay true for the
first run even if environment defaults have drifted:

```bash
curl -fsS -X POST "$ATLAS_API_BASE_URL/api/v1/autonomous/content_ops_deflection_delta_automation/run" \
  -H "Authorization: Bearer $ATLAS_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"delivery_dry_run": true, "target_account_id": "<account-id>", "current_request_id": "<current-request-id>"}'
```

Poll the execution by UUID and review the final payload:

```bash
DELTA_TASK_ID="$(
  curl -fsS "$ATLAS_API_BASE_URL/api/v1/autonomous/?include_disabled=true" \
    -H "Authorization: Bearer $ATLAS_ADMIN_TOKEN" \
    | jq -r '.tasks[] | select(.name == "content_ops_deflection_delta_automation") | .id'
)"

curl -fsS "$ATLAS_API_BASE_URL/api/v1/autonomous/$DELTA_TASK_ID/executions?limit=1" \
  -H "Authorization: Bearer $ATLAS_ADMIN_TOKEN" \
  | jq '.executions[0] | {status, result, error}'
```

Proceed only if the execution payload echoes the expected `target_account_id`
and `current_request_id`, has no `delivery_missing_config`,
`delivery_dry_run_enabled` is true, `reports_scanned` is 1,
`delta_deliveries_enqueued` is at least 1, `delivery_dry_run` is 1, and
`delivery_failed` is 0. If the dry-run processed zero delivery rows, stop; the
email rendering/delivery path has not been rehearsed.

## Live Activation

After the dry-run payload is clean, flip live email delivery and restart the
brain. A manual live run for one opted-in buyer must include the same
`target_account_id` and `current_request_id` used in the dry run. Omitting
`target_account_id` scans paid accounts globally and drains pending delivery
rows globally; omitting `current_request_id` scopes to the account but may send
more than one pending delta for that account. Reserve the unscoped path for the
scheduled monthly cron after the entitlement/opt-in list is ready.

```bash
ATLAS_DEFLECTION_DELTA_ENABLED=true
ATLAS_DEFLECTION_DELIVERY_DRY_RUN=false
ATLAS_DEFLECTION_DELIVERY_FROM_EMAIL="reports@example.com"
ATLAS_DEFLECTION_DELIVERY_RESEND_API_KEY="<resend-api-key>"
```

After restart, send only to the opted-in account:

```bash
curl -fsS -X POST "$ATLAS_API_BASE_URL/api/v1/autonomous/content_ops_deflection_delta_automation/run" \
  -H "Authorization: Bearer $ATLAS_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"delivery_dry_run": false, "target_account_id": "<account-id>", "current_request_id": "<current-request-id>"}'
```

Poll the execution as above. Proceed only if the payload echoes the expected
`target_account_id` and `current_request_id`, `reports_scanned` is 1,
`delivery_dry_run_enabled` is false, `delivery_sent` is 1, and
`delivery_failed` is 0. Otherwise disable the task and use the rollback steps
below before retrying.

## Rollback

Disable delta generation first, then restart the brain:

```bash
ATLAS_DEFLECTION_DELTA_ENABLED=false
```

If an execution reports terminal delivery failure, keep the task disabled until
the failed rows have been inspected in `content_ops_deflection_delta_deliveries`.
Do not change `ATLAS_DEFLECTION_DELIVERY_DRY_RUN` during rollback unless you
intend to pause the separate paid report delivery drain too.
