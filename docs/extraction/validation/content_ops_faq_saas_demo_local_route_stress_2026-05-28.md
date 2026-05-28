# Content Ops FAQ SaaS Demo Local Route Stress - 2026-05-28

## Summary

The checked synthetic B2B SaaS FAQ demo passed a larger local HTTP route stress
run against Atlas. The smoke seeded the FAQ, queried search and detail through
the authenticated local API, and cleaned up the seeded FAQ row.

This run increases the prior local proof from 40 requests at concurrency 8 to
200 requests at concurrency 16.

## Inputs

- Worktree: `/home/juan-canfield/Desktop/Atlas-faq-after-986`
- API host: `http://127.0.0.1:8000`
- Database: local Atlas Postgres from gitignored `.env.local`
- Auth: local B2B growth smoke account and JWT from gitignored `.env.local`
- Source corpus: `extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv`
- Source rows: `36`

No secret values were printed or committed.

## Server

The local Atlas API was started with voice disabled because voice and ASR are
unrelated to the FAQ route path:

```bash
ATLAS_VOICE_ENABLED=false ATLAS_VOICE_AUTO_START_ASR=false \
  python -m uvicorn atlas_brain.main:app --host 127.0.0.1 --port 8000
```

The same pre-existing startup warnings from the local proof were visible:

- `b2b_campaigns.updated_at` missing during the host migration check.
- Optional local-service warnings for models, default LLM, Twilio, and MCP.

Those warnings did not affect the FAQ seed, search, detail, or cleanup path.

## Command

```bash
python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py \
  --base-url http://127.0.0.1:8000 \
  --route-requests 200 \
  --concurrency 16 \
  --max-error-rate 0 \
  --max-case-error-rate 0 \
  --max-p95-ms 750 \
  --max-single-request-ms 3000 \
  --max-case-p95-ms 750 \
  --max-case-single-request-ms 3000 \
  --max-detail-ms 1000 \
  --artifact-dir /tmp/faq-saas-demo-route-stress-artifacts \
  --output-result /tmp/faq-saas-demo-route-stress-result.json \
  --json
```

## Result

- Overall: `ok=true`
- Phase: `complete`
- Elapsed: `1.0170168680197094` seconds
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

- Requests: `200`
- Concurrency: `16`
- Aggregate error rate: `0.0`
- Aggregate p50 latency: `31.196987 ms`
- Aggregate p95 latency: `377.852722 ms`
- Aggregate max latency: `400.439793 ms`
- Detail checks: `200`
- Detail failures: `0`
- Detail p50 latency: `15.469312 ms`
- Detail p95 latency: `17.47523 ms`
- Detail max latency: `28.451518 ms`

Cleanup result:

- Deleted FAQ rows: `1`
- Delete status: `DELETE 1`

## Budget Checks

All configured budgets passed:

- `error_rate <= 0.0`: passed
- `p95_ms <= 750`: passed
- `max_ms <= 3000`: passed
- `detail_max_ms <= 1000`: passed
- `case_error_rate <= 0.0`: passed
- `case_p95_ms <= 750`: passed
- `case_max_ms <= 3000`: passed

## Interpretation

The read path held under the higher local request and concurrency level. Search
latency rose from the 40-request proof, but remained well inside the explicit
budget. Detail hydration stayed lower-latency than search and had no failures.

This still does not prove deployed-host behavior. The next production-relevant
step is the same smoke against the deployed Atlas API and deployed database once
the host URL, bearer token, and matching account ID are available.
