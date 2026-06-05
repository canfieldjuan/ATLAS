# PR-Content-Ops-FAQ-Search-Local-Multi-Corpus-Route-Stress

## Why this slice exists

The SaaS FAQ demo route now has local seed/search/detail proof and a larger
single-corpus route stress run. The next survivability question is whether the
same local route path stays correct when concurrent traffic rotates across
multiple ticket corpora for the authenticated account. This slice records that
route-level proof before deployed-host validation is available.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Robust testing.

1. Run the existing seeded FAQ route e2e smoke locally with multiple corpora for
   the smoke account.
2. Verify each mixed route case keeps search and detail results scoped to the
   expected corpus and generated FAQ.
3. Fix the seeded smoke's persisted FAQ draft shape so the detail route returns
   representative FAQ items instead of an empty `items` array.
4. Record route/detail latency, error, seed, and cleanup signals.

### Files touched

- `docs/extraction/validation/content_ops_faq_search_local_multi_corpus_route_stress_2026-05-28.md`
- `plans/PR-Content-Ops-FAQ-Search-Local-Multi-Corpus-Route-Stress.md`
- `scripts/smoke_content_ops_faq_search_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_concurrency.py`

## Mechanism

The run uses
`scripts/smoke_content_ops_faq_search_seeded_route_e2e.py` against local
`uvicorn atlas_brain.main:app`, with the gitignored local database, bearer
token, and account ID. The smoke seeds multiple corpora for that account, emits
route cases with expected corpus/FAQ/detail fields, hits the local route under
concurrency, performs a detail check, and cleans up the seeded FAQ rows.

The first run exposed that `scripts/smoke_content_ops_faq_search_concurrency.py`
seeded parent FAQ drafts with `items=[]`, so search passed but the detail
contract correctly failed. This slice updates that seeded draft to include
minimal generated-FAQ-shaped items derived from the same seeded case and corpus
data.

## Intentional

- One local bearer token means this is multi-corpus route stress for one tenant,
  not cross-account hosted-route stress.
- No deployed-host claim: this is local robust testing only.
- No secret values are committed.

## Deferred

- Cross-account route concurrency remains deferred until we have multiple real
  bearer tokens mapped to different accounts or a dedicated local auth fixture.
- Deployed-host validation remains deferred until the real deployed API URL,
  bearer token, and matching account ID are available.
- Parked hardening: none unless the run exposes a new nonblocking issue.

## Verification

```bash
ATLAS_VOICE_ENABLED=false ATLAS_VOICE_AUTO_START_ASR=false python -m uvicorn atlas_brain.main:app --host 127.0.0.1 --port 8000
python scripts/smoke_content_ops_faq_search_seeded_route_e2e.py --base-url http://127.0.0.1:8000 --corpora-per-account 6 --documents-per-corpus 6 --seed-iterations 72 --route-requests 360 --concurrency 24 --pool-size 4 --max-error-rate 0 --max-case-error-rate 0 --max-p95-ms 1000 --max-single-request-ms 4000 --max-case-p95-ms 1000 --max-case-single-request-ms 4000 --max-detail-ms 1000 --artifact-dir /tmp/faq-local-multi-corpus-route-stress-artifacts --output-result /tmp/faq-local-multi-corpus-route-stress-result.json --json
```

- First route run exposed the seeded-detail issue: route passed, cleanup
  deleted 6 rows, detail failed with `detail.items must include at least one
  item`.
- `python -m py_compile scripts/smoke_content_ops_faq_search_concurrency.py tests/test_smoke_content_ops_faq_search_concurrency.py` - passed.
- `pytest tests/test_smoke_content_ops_faq_search_concurrency.py -q` - passed,
  27 tests.
- Post-fix route run passed: `ok=true`, 360 route requests, concurrency 24,
  12 route cases, 0 route errors, route p95 `437.914288 ms`, detail checked
  with 0 errors, cleanup `DELETE 6`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Validation proof doc | ~155 |
| Plan doc | ~87 |
| Seeded smoke fix | ~54 |
| Tests | ~23 |
| **Total** | **~317** |
