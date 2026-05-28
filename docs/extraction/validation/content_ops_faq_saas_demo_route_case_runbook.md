# Content Ops FAQ SaaS Demo Route Case Runbook

Use this runbook to prove the checked synthetic B2B SaaS FAQ demo can be seeded
into the FAQ search tables and read back through the deployed hosted FAQ search
route.

This is the demo-specific validation path. For generic multi-corpus seeded route
coverage, use `content_ops_faq_seeded_route_e2e_runbook.md`.

## Required Inputs

- `EXTRACTED_DATABASE_URL` or `DATABASE_URL`: Postgres database used by the
  deployed Atlas API.
- `ATLAS_API_BASE_URL`: deployed Atlas API host, for example
  `https://atlas-api.example.com`.
- `ATLAS_B2B_JWT` or `ATLAS_TOKEN`: bearer token for the account under test.
- `ATLAS_FAQ_SEARCH_ACCOUNT_ID`: account ID matching the bearer token. The
  seeder writes the demo FAQ under this account, and the hosted route smoke uses
  the token to query it.

Run the FAQ search migrations before this smoke:

```bash
python scripts/run_extracted_content_pipeline_migrations.py
```

## Recommended One-Command Smoke

```bash
python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py \
  --database-url "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}" \
  --base-url "$ATLAS_API_BASE_URL" \
  --token "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}" \
  --account-id "$ATLAS_FAQ_SEARCH_ACCOUNT_ID" \
  --route-requests 40 \
  --concurrency 8 \
  --max-error-rate 0 \
  --max-case-error-rate 0 \
  --max-p95-ms 1500 \
  --max-single-request-ms 3000 \
  --max-case-p95-ms 1500 \
  --max-case-single-request-ms 3000 \
  --max-detail-ms 2500 \
  --output-result /tmp/faq-saas-demo-route-e2e-result.json
```

Use `--json` when stdout should be machine-readable. Without `--json`, stdout is
a compact status line; the full proof lives in `--output-result`.

The smoke writes a temporary route case file, validates the hosted search and
detail path against it, then deletes the seeded FAQ unless `--keep-data` is set.

## Manual Fallback: Seed The SaaS Demo And Write Route Cases

Use the manual steps when you need to inspect or preserve the emitted route case
file between seed and route validation.

```bash
python scripts/seed_content_ops_faq_saas_demo.py \
  --database-url "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}" \
  --account-id "$ATLAS_FAQ_SEARCH_ACCOUNT_ID" \
  --route-case-file-output /tmp/faq-saas-demo-route-cases.json \
  --output-result /tmp/faq-saas-demo-seed-result.json
```

Use `--json` when stdout should be machine-readable. Without `--json`, stdout is
a compact status line; the full seed proof lives in `--output-result`.

The route case file contains the generated FAQ's expected first-result identity
and expected hydrated detail fields:

- `expected_first_account_id`
- `expected_first_corpus_id`
- `expected_first_faq_id`
- `expected_detail_account_id`
- `expected_detail_target_id`
- `expected_detail_target_mode`
- `expected_detail_title`
- `expected_detail_status`

## Manual Fallback: Validate The Hosted Route Against The Seeded Case

```bash
python scripts/smoke_content_ops_faq_search_route_concurrency.py \
  --base-url "$ATLAS_API_BASE_URL" \
  --token "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}" \
  --case-file /tmp/faq-saas-demo-route-cases.json \
  --requests 40 \
  --concurrency 8 \
  --require-detail \
  --max-error-rate 0 \
  --max-case-error-rate 0 \
  --max-p95-ms 1500 \
  --max-single-request-ms 3000 \
  --max-case-p95-ms 1500 \
  --max-case-single-request-ms 3000 \
  --max-detail-ms 2500 \
  --output-result /tmp/faq-saas-demo-route-result.json
```

Use `--json` when stdout should be machine-readable. Without `--json`, stdout is
a compact status line with aggregate and worst-case route signals.

## What This Proves

- The checked synthetic B2B SaaS FAQ can be generated from the local source CSV.
- The FAQ draft can be saved and approved in the configured Postgres database.
- The approval path writes the FAQ search projection used by the hosted route.
- The hosted route returns the seeded FAQ as the first result for the demo query.
- The hosted detail path hydrates the same FAQ, account, target, title, and
  status emitted by the seeder.

## Result Checks

Open `/tmp/faq-saas-demo-route-e2e-result.json` and inspect:

- `ok`: seed, hosted route/detail, cleanup, and artifact-read status.
- `seed.result_artifact.faq_id`: generated FAQ id used for route and cleanup.
- `route.result_artifact.requests`: hosted route request count.
- `route.result_artifact.detail`: hydrated detail checks and failures.
- `cleanup`: cleanup status, or `not_run_reason` when `--keep-data` is set.

Open `/tmp/faq-saas-demo-seed-result.json` and inspect:

- `ok`: seed, approval, projection, verification search, and route-case write
  status.
- `faq_id`: generated FAQ id used for route and cleanup checks.
- `search.matched_seeded_faq`: whether the DB search repository found the seeded
  FAQ id before route validation.
- `route_case_file`: path and case count for the emitted route case artifact.

Open `/tmp/faq-saas-demo-route-result.json` and inspect:

- `ok`: route, detail, and budget status.
- `requests.total`: hosted route request count.
- `cases.summaries[]`: per-case route and detail health.
- `detail.failures`: hydrated detail mismatches or route/detail errors.
- `budgets.failures[]`: deterministic latency or error budget failures.

## Cleanup

The recommended one-command smoke cleans up the seeded FAQ by default. If you
use the manual fallback and the seeded FAQ should not remain in the host
database, delete it with the FAQ id from the seed result:

```bash
python scripts/seed_content_ops_faq_saas_demo.py \
  --database-url "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}" \
  --account-id "$ATLAS_FAQ_SEARCH_ACCOUNT_ID" \
  --cleanup-faq-id "<faq_id_from_seed_result>" \
  --output-result /tmp/faq-saas-demo-cleanup-result.json
```

Cleanup mode does not accept `--route-case-file-output` because no new seeded
FAQ id is produced.

## Interpreting Failures

- Seeder preflight failures exit with code `1` and do not connect to Postgres.
- Seeder runtime failures still write the result artifact when
  `--output-result` is provided.
- Route preflight failures exit with code `2` and do not issue hosted requests.
- Route contract, detail, or budget failures exit with code `1` and still write
  the result artifact.
- A passing aggregate route error rate does not prove each case is healthy. Keep
  `--max-case-error-rate 0` for demo validation.

## Deferred Thresholds

The sample latency values above are placeholders. Replace them with thresholds
from repeated hosted demo runs before treating them as production SLOs.
