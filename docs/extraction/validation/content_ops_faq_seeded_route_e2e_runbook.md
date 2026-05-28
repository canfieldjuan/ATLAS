# Content Ops FAQ Seeded Route E2E Runbook

Use this runbook to seed FAQ search rows into Postgres, exercise the hosted FAQ
search route, hydrate one generated FAQ detail, and clean up the seeded rows in
one command.

This is the preferred go-live probe when you need to prove the real search path,
not just route liveness. It composes the seeded DB smoke, hosted route
concurrency smoke, detail contract checker, and cleanup guard.

## Required Inputs

- `EXTRACTED_DATABASE_URL` or `DATABASE_URL`: Postgres database used by the
  deployed Atlas API.
- `ATLAS_API_BASE_URL`: deployed Atlas API host, for example
  `https://atlas-api.example.com`.
- `ATLAS_B2B_JWT` or `ATLAS_TOKEN`: bearer token for the account under test.
- `ATLAS_FAQ_SEARCH_ACCOUNT_ID`: account ID matching the bearer token. The
  runner does not decode the token; it seeds rows under the account you provide.

Run the FAQ search migrations before this smoke:

```bash
python scripts/run_extracted_content_pipeline_migrations.py
```

## Recommended Seeded E2E Smoke

```bash
python scripts/smoke_content_ops_faq_search_seeded_route_e2e.py \
  --database-url "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}" \
  --base-url "$ATLAS_API_BASE_URL" \
  --token "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}" \
  --account-id "$ATLAS_FAQ_SEARCH_ACCOUNT_ID" \
  --corpora-per-account 2 \
  --documents-per-corpus 3 \
  --seed-iterations 12 \
  --route-requests 40 \
  --concurrency 8 \
  --pool-size 2 \
  --max-error-rate 0 \
  --max-case-error-rate 0 \
  --max-p95-ms 1500 \
  --max-single-request-ms 3000 \
  --max-case-p95-ms 1500 \
  --max-case-single-request-ms 3000 \
  --output-result /tmp/faq-search-seeded-route-e2e-result.json
```

Use `--json` when stdout should be machine-readable. Without `--json`, stdout is
a compact status line; the full proof lives in `--output-result`.

## What The Runner Does

1. Seeds approved generated FAQ rows into Postgres for the supplied account.
2. Writes a route case file with expected hit and miss cases for each seeded
   corpus.
3. Runs the hosted FAQ search route concurrency smoke against the seeded cases.
4. Runs one detail contract check for the first seeded hit case.
5. Deletes the seeded FAQ rows by emitted FAQ IDs unless `--keep-data` is set.
6. Writes one compact JSON result with seed, route, detail, cleanup, and child
   artifact summaries.

## Result Checks

Open the result artifact and inspect:

- `ok`: overall pass/fail across seed, route, detail, cleanup, and artifact
  cleanup.
- `seed.result_artifact`: compact seed counts, setup status, latency, and
  cleanup-manifest proof from the DB smoke.
- `route.result_artifact.requests`: hosted route request count and concurrency
  used by the child route smoke.
- `route.result_artifact.cases`: number of route cases loaded from the seeded
  case file.
- `route.result_artifact.budgets`: aggregate and per-case route budget checks.
- `detail.result_artifact`: detail contract status, hydrated FAQ ID, and search
  plus detail timings.
- `cleanup.requested_faq_ids` and `cleanup.deleted_faq_ids`: cleanup row-count
  proof for seeded FAQ drafts.
- `artifact_cleanup`: whether temporary child artifacts were removed after the
  top-level summary embedded compact child results.

## Interpreting Failures

- Preflight failures exit with code `2` and do not seed or call the hosted
  route.
- Seed, route, detail, cleanup, or budget failures exit with code `1` and still
  write the top-level result artifact.
- A route aggregate error budget can pass while one query/corpus case fails. Use
  `--max-case-error-rate`, `--max-case-p95-ms`, and
  `--max-case-single-request-ms` to fail closed per case.
- Cleanup failure makes the run fail unless `--keep-data` is set. Use
  `--keep-data` only when you intentionally want to inspect seeded rows after
  the run.
- Detail checking is enabled by default in this seeded e2e runner. Use
  `--skip-detail-check` only for a search-only liveness probe.

## Deferred Thresholds

The sample latency values above are placeholders. Replace them with thresholds
from repeated hosted runs before treating them as production SLOs.
