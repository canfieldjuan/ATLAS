# PR-Content-Ops-FAQ-SaaS-Demo-Local-Route-Proof

## Why this slice exists

The FAQ SaaS demo route now has local smoke inputs and a preflight path, but the
next useful validation is the real flow: seed the checked SaaS demo FAQ into
Postgres, query it through the Atlas HTTP route with bearer auth, hydrate detail,
and clean up the seeded FAQ. This closes the local proof gap before a deployed
host run.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Functional validation.

1. Record the local SaaS FAQ demo route e2e proof.
2. Park unrelated startup issues observed during the run.
3. Keep product/runtime behavior unchanged.

### Files touched

- `docs/extraction/validation/content_ops_faq_saas_demo_local_route_e2e_2026-05-28.md`
- `HARDENING.md`
- `plans/PR-Content-Ops-FAQ-SaaS-Demo-Local-Route-Proof.md`

## Mechanism

This is a validation documentation slice. The run used
`scripts/smoke_content_ops_faq_saas_demo_route_e2e.py` against a local
`uvicorn atlas_brain.main:app` server with a local B2B growth smoke account and
gitignored `.env.local` inputs. The result artifact is summarized in the
validation note without committing secret values.

## Intentional

- No code changes: the slice proves the current route wiring rather than
  changing it.
- The proof is local-hosted, not deployed-hosted. The deployed API URL and
  deployed bearer token are still operator-provided inputs.
- Local account IDs and JWT values are not committed.

## Deferred

- Deployed-host validation remains deferred until the real deployed
  `ATLAS_API_BASE_URL`, bearer token, and matching account ID are available.
- Parked hardening:
  - Atlas startup migration warning on `b2b_campaigns.updated_at`.
  - Voice/ASR auto-start attempts CUDA during non-voice route validation unless
    explicitly disabled.

## Verification

```bash
EXTRACTED_DATABASE_URL="$(python - <<'PY'
from pathlib import Path
for line in Path('.env.local').read_text().splitlines():
    if line.startswith('EXTRACTED_DATABASE_URL='):
        print(line.split('=', 1)[1])
        break
PY
)" python scripts/run_extracted_content_pipeline_migrations.py --json
ATLAS_VOICE_ENABLED=false ATLAS_VOICE_AUTO_START_ASR=false python -m uvicorn atlas_brain.main:app --host 127.0.0.1 --port 8000
python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py --route-requests 40 --concurrency 8 --max-error-rate 0 --max-case-error-rate 0 --max-p95-ms 1500 --max-single-request-ms 3000 --max-case-p95-ms 1500 --max-case-single-request-ms 3000 --max-detail-ms 2500 --artifact-dir /tmp/faq-saas-demo-route-e2e-artifacts --output-result /tmp/faq-saas-demo-route-e2e-result.json --json
```

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Validation proof doc | ~136 |
| Parked hardening entries | ~20 |
| Plan doc | ~74 |
| **Total** | **~230** |
