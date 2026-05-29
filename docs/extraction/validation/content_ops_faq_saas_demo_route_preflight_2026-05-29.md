# Content Ops FAQ SaaS Demo Route Preflight - 2026-05-29

## Why This Ran

After the FAQ startup migration drift fix landed, the next validation target is
the seeded B2B SaaS FAQ route E2E:

1. Generate the checked SaaS FAQ demo.
2. Save and approve it in Postgres.
3. Verify the search projection.
4. Hit the deployed FAQ search and detail routes under demo traffic.

The runbook requires live hosted inputs before it can safely write data or send
route traffic, so this slice ran the existing preflight-only mode first.

## Command

```bash
mkdir -p tmp/faq_saas_demo_route_preflight_20260529
python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py \
  --database-url "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}" \
  --base-url "$ATLAS_API_BASE_URL" \
  --token "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}" \
  --account-id "${ATLAS_FAQ_SEARCH_ACCOUNT_ID:-$ATLAS_ACCOUNT_ID}" \
  --preflight-only \
  --json \
  --output-result tmp/faq_saas_demo_route_preflight_20260529/result.json
python -m json.tool tmp/faq_saas_demo_route_preflight_20260529/result.json
```

## Result

Exit code: `2`

This is the expected preflight-blocked exit code when required inputs are
missing. No Postgres writes were attempted, no hosted route requests were sent,
and cleanup was skipped because no seeded FAQ was created.

## Required Input Status

| Input | Env names checked | Present |
|---|---|---:|
| Database URL | `EXTRACTED_DATABASE_URL`, `DATABASE_URL`, or `atlas_brain.storage.config.db_settings.dsn` from explicit `ATLAS_DB_HOST`/`ATLAS_DB_SOCKET_PATH` settings | no |
| Deployed API base URL | `ATLAS_API_BASE_URL` | no |
| Bearer token | `ATLAS_B2B_JWT`, `ATLAS_TOKEN` | no |
| Account id | `ATLAS_FAQ_SEARCH_ACCOUNT_ID`, `ATLAS_ACCOUNT_ID` | no |

## Result Artifact Summary

```json
{
  "ok": false,
  "phase": "preflight",
  "preflight_errors": [
    "Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL",
    "ATLAS_API_BASE_URL or --base-url is required",
    "ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required",
    "ATLAS_FAQ_SEARCH_ACCOUNT_ID, ATLAS_ACCOUNT_ID, or --account-id is required"
  ],
  "required_inputs": {
    "database_url": {"present": false},
    "base_url": {"present": false},
    "token": {"present": false},
    "account_id": {"present": false}
  },
  "seed": {"ok": false, "skipped": true, "not_run_reason": "preflight_failed"},
  "route": {"ok": false, "skipped": true, "not_run_reason": "preflight_failed"},
  "cleanup": {"ok": true, "skipped": true, "not_run_reason": "preflight_failed"}
}
```

## Next Run

Set the required inputs, then run:

If `EXTRACTED_DATABASE_URL`/`DATABASE_URL` are not set, the scripts now use
`atlas_brain.storage.config.db_settings.dsn` only when an explicit
`ATLAS_DB_HOST` or `ATLAS_DB_SOCKET_PATH` target setting is present. This keeps
Atlas' localhost defaults from satisfying hosted proof preflight by themselves.

```bash
python scripts/run_extracted_content_pipeline_migrations.py
python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py \
  --database-url "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}" \
  --base-url "$ATLAS_API_BASE_URL" \
  --token "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}" \
  --account-id "${ATLAS_FAQ_SEARCH_ACCOUNT_ID:-$ATLAS_ACCOUNT_ID}" \
  --route-requests 40 \
  --concurrency 8 \
  --max-error-rate 0 \
  --max-case-error-rate 0 \
  --max-p95-ms 1500 \
  --max-single-request-ms 3000 \
  --max-case-p95-ms 1500 \
  --max-case-single-request-ms 3000 \
  --max-detail-ms 2500 \
  --artifact-dir /tmp/faq-saas-demo-route-e2e-artifacts \
  --output-result /tmp/faq-saas-demo-route-e2e-result.json
```

Keep the generated result artifact from that run; it is the proof needed before
calling the hosted SaaS demo search path ready.

## Follow-Up: DB Settings Fallback

After adding the same `db_settings.dsn` fallback used by the other Content Ops
Postgres smokes, the preflight was rerun without `EXTRACTED_DATABASE_URL` or
`DATABASE_URL`:

```bash
python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py \
  --preflight-only \
  --json \
  --output-result tmp/faq_saas_demo_route_preflight_db_settings_20260529/result3.json
```

Exit code remained `2`, but the database input is now present via explicit
Atlas DB settings. The remaining blockers are the hosted API base URL, bearer
token, and account id.

```json
{
  "preflight_errors": [
    "ATLAS_API_BASE_URL or --base-url is required",
    "ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required",
    "ATLAS_FAQ_SEARCH_ACCOUNT_ID, ATLAS_ACCOUNT_ID, or --account-id is required"
  ],
  "required_inputs": {
    "database_url": {"present": true},
    "base_url": {"present": false},
    "token": {"present": false},
    "account_id": {"present": false}
  }
}
```
