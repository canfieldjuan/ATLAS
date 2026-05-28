# Content Ops FAQ SaaS Demo Hosted Route Proof - 2026-05-28

## Summary

The deployed-host FAQ route proof did not pass. The configured inputs were
present, and the database seed/projection/cleanup path worked, but the HTTP
route phase hit a local API base URL with no server listening. All route
requests failed with connection refused, so this run does not prove deployed
Atlas API behavior.

No secret values are recorded here.

## Scope

This validation attempted the checked synthetic B2B SaaS FAQ demo hosted-route
flow:

1. Generate the SaaS demo FAQ from the checked source CSV.
2. Save and approve the FAQ draft in the configured Postgres database.
3. Write the FAQ search projection.
4. Query the configured API base URL at
   `/api/v1/content-ops/faq-deflection-search`.
5. Hydrate FAQ detail from the returned search result.
6. Delete the seeded FAQ row.

Source corpus:

`extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv`

## Preflight

The preflight passed:

- Database URL: present
- API base URL: present
- Bearer token: present
- Account id: present

After the route failure, the API base URL was classified without printing it:

- Scheme: `http`
- Host kind: local
- Port: present

That means the configured route target was not a deployed Atlas API host for
this run.

## Result

Status: not accepted as a deployed-host proof.

| Phase | Result | Notes |
|---|---|---|
| Preflight | Passed | Required inputs were present. |
| Seed | Passed | 36 source rows generated 7 FAQ items and 7 search documents. |
| DB verification search | Passed | The configured database search matched the seeded FAQ. |
| Hosted route/detail | Failed | 40/40 route requests failed with `[Errno 111] Connection refused`; no detail rows were checked. |
| Cleanup | Passed | The seeded FAQ row was deleted with `DELETE 1`. |

Route budget failures:

- `error_rate exceeded 0.0`
- `detail_max_ms had no checked detail rows`
- `case_error_rate exceeded 0.0 for case 0`

## Interpretation

The FAQ database write path is healthy enough to seed, project, verify, and
clean up the SaaS demo FAQ under the configured account. The read path was not
tested against a deployed route because the configured API base URL pointed at a
local HTTP host with no listening server.

The next FAQ slice should rerun this same smoke with a real deployed
`ATLAS_API_BASE_URL` that matches the configured database and bearer token. If
we want this failure mode to be fail-fast, promote the parked hardening item for
rejecting local API URLs in hosted-proof mode.

## Artifacts

Local artifacts from this run:

| Artifact | Path |
|---|---|
| Preflight result | `/tmp/faq-saas-demo-hosted-route-proof-preflight.json` |
| One-command result | `/tmp/faq-saas-demo-hosted-route-proof-result.json` |
| Seed result | `/tmp/faq-saas-demo-hosted-route-proof-artifacts/seed-result.json` |
| Route result | `/tmp/faq-saas-demo-hosted-route-proof-artifacts/route-result.json` |
| Cleanup result | `/tmp/faq-saas-demo-hosted-route-proof-artifacts/cleanup-result.json` |
| Route cases | `/tmp/faq-saas-demo-hosted-route-proof-artifacts/route-cases.json` |

## Verification

- Command: python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py --database-url "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}" --base-url "$ATLAS_API_BASE_URL" --token "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}" --account-id "${ATLAS_FAQ_SEARCH_ACCOUNT_ID:-$ATLAS_ACCOUNT_ID}" --preflight-only --json --output-result /tmp/faq-saas-demo-hosted-route-proof-preflight.json
  - Passed with all required inputs present.
- Command: python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py --database-url "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}" --base-url "$ATLAS_API_BASE_URL" --token "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}" --account-id "${ATLAS_FAQ_SEARCH_ACCOUNT_ID:-$ATLAS_ACCOUNT_ID}" --route-requests 40 --concurrency 8 --max-error-rate 0 --max-case-error-rate 0 --max-p95-ms 1500 --max-single-request-ms 3000 --max-case-p95-ms 1500 --max-case-single-request-ms 3000 --max-detail-ms 2500 --artifact-dir /tmp/faq-saas-demo-hosted-route-proof-artifacts --output-result /tmp/faq-saas-demo-hosted-route-proof-result.json --json
  - Failed on route requests with connection refused.
  - Seed and cleanup both passed.
