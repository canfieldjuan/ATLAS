# Content Ops FAQ Route Concurrency Runbook

Use this runbook to exercise the hosted FAQ search route under mixed query and
corpus traffic. It assumes the route already has searchable FAQ rows.

## Required Inputs

- `ATLAS_API_BASE_URL`: deployed Atlas API host, for example
  `https://atlas-api.example.com`.
- `ATLAS_B2B_JWT` or `ATLAS_TOKEN`: bearer token for the target account.
- A JSON case file with representative hit cases for detail hydration checks.

Minimal detail case-file shape:

```json
[
  {
    "query": "reset password",
    "corpus_id": "demo-saas",
    "limit": 5,
    "require_results": true
  }
]
```

Use hit-only case files with `--require-detail`, because detail hydration needs
a `results[0].faq_id` value. Run known miss cases separately without
`--require-detail`.

## Recommended Smoke

```bash
python scripts/smoke_content_ops_faq_search_route_concurrency.py \
  --base-url "$ATLAS_API_BASE_URL" \
  --token "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}" \
  --case-file /tmp/faq-route-cases.json \
  --requests 40 \
  --concurrency 8 \
  --require-detail \
  --max-error-rate 0 \
  --max-case-error-rate 0 \
  --max-p95-ms 1500 \
  --max-single-request-ms 3000 \
  --max-detail-ms 1000 \
  --output-result /tmp/faq-route-concurrency-result.json
```

For miss/liveness coverage, use a separate case file containing
`"require_results": false` cases and omit the detail flags:

```bash
python scripts/smoke_content_ops_faq_search_route_concurrency.py \
  --base-url "$ATLAS_API_BASE_URL" \
  --token "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}" \
  --case-file /tmp/faq-route-miss-cases.json \
  --requests 20 \
  --concurrency 4 \
  --max-error-rate 0 \
  --max-case-error-rate 0 \
  --output-result /tmp/faq-route-miss-result.json
```

Use `--json` when stdout should be machine-readable. Without `--json`, the
default line includes aggregate errors and latency plus the worst per-case
signal:

- `worst_case_index`
- `worst_case_errors`
- `worst_case_p95_ms`
- `worst_case_max_ms`

## Result Checks

Open the result artifact and inspect:

- `ok`: overall pass/fail after preflight, contract, and budget checks.
- `errors.rate`: aggregate route error rate.
- `cases.summaries[]`: one compact row per loaded case, including request
  count, error rate, route latency, detail checks, and detail latency.
- `detail.checked`: number of requests that hydrated generated FAQ detail.
- `detail.failures`: detail hydration or detail contract failures.
- `budgets.checks[]`: aggregate and per-case budget checks.
- `budgets.failures[]`: deterministic failure strings for exceeded budgets.

`cases.items` is only a preview of the first 20 case definitions. Use
`cases.summaries` for complete case-level status.

## Interpreting Failures

- Preflight failures exit with code `2` and do not issue route requests.
- Contract or budget failures exit with code `1` and still write the result
  artifact.
- A passing aggregate `errors.rate` does not prove every case is healthy. Use
  `--max-case-error-rate` to fail closed on any bad query/corpus case.
- `--max-detail-ms` only applies with `--require-detail`; otherwise the smoke
  would claim a detail budget for work it did not perform.

## Deferred Thresholds

The sample latency values above are placeholders. Replace them with hosted
thresholds from real demo traffic before treating them as production SLOs.
