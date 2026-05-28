# Content Ops FAQ SaaS Demo Local Route E2E - 2026-05-28

## Summary

The checked synthetic B2B SaaS FAQ demo was seeded into local Atlas Postgres,
queried through the local Atlas HTTP route, hydrated through the FAQ detail
route, and cleaned up successfully.

This proves the local route flow now works end to end:

1. Generate the SaaS demo FAQ from the checked source CSV.
2. Save and approve the FAQ draft in Postgres.
3. Write the FAQ search projection.
4. Query `GET /api/v1/content-ops/faq-deflection-search`.
5. Hydrate `GET /api/v1/content-ops/faq-deflection-search/{faq_id}`.
6. Delete the seeded FAQ row.

This is not the deployed-host proof. The deployed run still needs the real
deployed Atlas API URL, bearer token, and matching account ID.

## Inputs

- Worktree: `/home/juan-canfield/Desktop/Atlas-faq-after-986`
- API host: `http://127.0.0.1:8000`
- Database: local Atlas Postgres from gitignored `.env.local`
- Auth: local B2B growth smoke account and JWT from gitignored `.env.local`
- Source corpus: `extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv`
- Source rows: `36`

No secret values were printed or committed.

## Setup

The extracted FAQ search migration runner was executed with the local DSN passed
through the process environment. It applied the FAQ search projection migration:

- `327_ticket_faq_search_documents.sql`: applied
- Other extracted content migrations: skipped as already present

The local Atlas API was started with voice disabled for route validation:

```bash
ATLAS_VOICE_ENABLED=false ATLAS_VOICE_AUTO_START_ASR=false \
  python -m uvicorn atlas_brain.main:app --host 127.0.0.1 --port 8000
```

Voice is not part of the FAQ route path. Disabling it avoided local ASR/CUDA
startup noise while keeping the Atlas API, auth, database pool, and Content Ops
route wiring intact.

## Command

```bash
python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py \
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
  --output-result /tmp/faq-saas-demo-route-e2e-result.json \
  --json
```

## Result

- Overall: `ok=true`
- Phase: `complete`
- Elapsed: `0.5104803719732445` seconds
- Seed: `ok=true`
- Route: `ok=true`
- Cleanup: `ok=true`
- Errors: `0`

Seed result:

- FAQ source rows: `36`
- Generated FAQ items: `7`
- Projected search documents: `7`
- Corpus: `synthetic-b2b-saas-demo`
- Target mode: `support_account`
- Status: `approved`
- Verification search matched the seeded FAQ: `true`
- Route case file cases: `1`

Route result:

- Requests: `40`
- Concurrency: `8`
- Aggregate error rate: `0.0`
- Aggregate p50 latency: `14.566523 ms`
- Aggregate p95 latency: `189.579806 ms`
- Aggregate max latency: `192.388902 ms`
- Detail checks: `40`
- Detail failures: `0`
- Detail p50 latency: `7.162183 ms`
- Detail p95 latency: `8.509531 ms`
- Detail max latency: `13.614998 ms`

Cleanup result:

- Deleted FAQ rows: `1`
- Delete status: `DELETE 1`

## Budget Checks

All configured budgets passed:

- `error_rate <= 0.0`: passed
- `p95_ms <= 1500`: passed
- `max_ms <= 3000`: passed
- `detail_max_ms <= 2500`: passed
- `case_error_rate <= 0.0`: passed
- `case_p95_ms <= 1500`: passed
- `case_max_ms <= 3000`: passed

## Observed Startup Issues

These did not block the FAQ route flow, but they were visible during local
server startup and are parked in `HARDENING.md`:

- Atlas startup migration check warned on
  `309_campaign_sequences_unique_active_recipient.sql` because
  `b2b_campaigns.updated_at` is missing in the local DB schema.
- Starting Atlas without disabling voice attempted to auto-start ASR on CUDA and
  failed because no CUDA GPU was available. The FAQ route proof used
  `ATLAS_VOICE_ENABLED=false ATLAS_VOICE_AUTO_START_ASR=false`.

Other optional local-service warnings also appeared (`models` router unavailable,
default LLM not configured, Twilio credentials missing, `mcp` missing), but they
did not affect the FAQ route, auth, database, seed, search, detail, or cleanup
path in this validation.
