# PR-Content-Ops-FAQ-SaaS-Demo-Local-Route-Stress

## Why this slice exists

The SaaS FAQ demo route has a passing 40-request local e2e proof. The next
question is whether the same seeded search/detail path holds under a modestly
harder local read pattern before we move to deployed-host validation. This slice
keeps the work in the FAQ-search lane and records the larger run without
changing product code.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Robust testing.

1. Run the checked SaaS FAQ demo route e2e smoke locally at a higher request and
   concurrency level.
2. Record search/detail latency, error, seed, and cleanup signals.
3. Do not change runtime behavior.

### Files touched

- `docs/extraction/validation/content_ops_faq_saas_demo_local_route_stress_2026-05-28.md`
- `plans/PR-Content-Ops-FAQ-SaaS-Demo-Local-Route-Stress.md`

## Mechanism

This is a validation documentation slice. The run uses
`scripts/smoke_content_ops_faq_saas_demo_route_e2e.py` against local
`uvicorn atlas_brain.main:app`, with the same gitignored local smoke inputs as
the functional proof, but raises route requests and concurrency to exercise the
read path harder.

## Intentional

- No code changes: this slice measures the current route behavior.
- No deployed-host claim: this is local robust testing only.
- No secret values are committed.

## Deferred

- Deployed-host validation remains deferred until the real deployed API URL,
  bearer token, and matching account ID are available.
- Parked hardening: none unless the stress run exposes a new nonblocking issue.

## Verification

```bash
ATLAS_VOICE_ENABLED=false ATLAS_VOICE_AUTO_START_ASR=false python -m uvicorn atlas_brain.main:app --host 127.0.0.1 --port 8000
python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py --base-url http://127.0.0.1:8000 --route-requests 200 --concurrency 16 --max-error-rate 0 --max-case-error-rate 0 --max-p95-ms 750 --max-single-request-ms 3000 --max-case-p95-ms 750 --max-case-single-request-ms 3000 --max-detail-ms 1000 --artifact-dir /tmp/faq-saas-demo-route-stress-artifacts --output-result /tmp/faq-saas-demo-route-stress-result.json --json
```

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Validation proof doc | ~118 |
| Plan doc | ~60 |
| **Total** | **~178** |
