# Content Ops FAQ Search Local Multi-Corpus Route Stress - 2026-05-28

## Summary

The seeded FAQ search route e2e passed a local multi-corpus route stress run
against Atlas after fixing the smoke seed data to include generated FAQ items
for the detail route.

This run seeded 6 corpora for the authenticated smoke account, generated 12
route cases across hit and miss queries, ran 360 local HTTP route requests at
concurrency 24, performed a search-to-detail contract check, and cleaned up the
seeded FAQ rows.

## Inputs

- Worktree: `/home/juan-canfield/Desktop/Atlas-faq-after-986`
- API host: `http://127.0.0.1:8000`
- Database: local Atlas Postgres from gitignored `.env.local`
- Auth: local B2B growth smoke JWT from gitignored `.env.local`
- Account: local smoke account from gitignored `.env.local`
- Seeded corpora: `6`
- Documents per corpus: `6`
- Route cases: `12`

No secret values were printed or committed.

## Server

The local Atlas API was started with voice disabled because voice and ASR are
unrelated to the FAQ route path:

```bash
ATLAS_VOICE_ENABLED=false ATLAS_VOICE_AUTO_START_ASR=false \
  python -m uvicorn atlas_brain.main:app --host 127.0.0.1 --port 8000
```

The same pre-existing startup warnings from earlier local proofs were visible:

- `b2b_campaigns.updated_at` missing during the host migration check.
- Optional local-service warnings for models, default LLM, Twilio, and MCP.

Those warnings did not affect the FAQ seed, search, detail, or cleanup path.

## Issue Found And Fixed

The first run failed the detail contract:

- Search route phase: passed.
- Cleanup phase: passed, `DELETE 6`.
- Detail phase: failed with `detail.items must include at least one item`.

Root cause: `scripts/smoke_content_ops_faq_search_concurrency.py` seeded
parent `ticket_faq_markdown` rows with `items=[]` while also emitting detail
route expectations. The route was correct to return the persisted empty items,
and the contract checker was correct to reject that shape.

Fix: the seeded smoke now persists generated-FAQ-shaped items derived from the
same seeded corpus data. The rerun below is the post-fix proof.

## Command

```bash
python scripts/smoke_content_ops_faq_search_seeded_route_e2e.py \
  --base-url http://127.0.0.1:8000 \
  --corpora-per-account 6 \
  --documents-per-corpus 6 \
  --seed-iterations 72 \
  --route-requests 360 \
  --concurrency 24 \
  --pool-size 4 \
  --max-error-rate 0 \
  --max-case-error-rate 0 \
  --max-p95-ms 1000 \
  --max-single-request-ms 4000 \
  --max-case-p95-ms 1000 \
  --max-case-single-request-ms 4000 \
  --max-detail-ms 1000 \
  --artifact-dir /tmp/faq-local-multi-corpus-route-stress-artifacts \
  --output-result /tmp/faq-local-multi-corpus-route-stress-result.json \
  --json
```

## Result

- Overall: `ok=true`
- Phase: `complete`
- Elapsed: `1.233586` seconds
- Seed: `ok=true`
- Route: `ok=true`
- Detail: `ok=true`
- Cleanup: `ok=true`

Seed result:

- Accounts: `1`
- Corpora per account: `6`
- Documents per corpus: `6`
- Search cases: `12`
- Seed iterations: `72`
- Seed concurrency: `24`
- Seed isolation failures: `0`
- Seed p95 latency: `6.884256 ms`
- Seed max latency: `8.532935 ms`

Route result:

- Requests: `360`
- Concurrency: `24`
- Route cases: `12`
- Aggregate error rate: `0.0`
- Aggregate p50 latency: `20.065656 ms`
- Aggregate p95 latency: `437.914288 ms`
- Aggregate max latency: `516.502223 ms`
- Case error rate: `0.0` for all 12 cases
- Worst case p95 latency: `495.702247 ms`
- Worst case max latency: `516.502223 ms`

Detail result:

- Detail checked: `true`
- Search elapsed: `14.069 ms`
- Detail elapsed: `1.935 ms`
- Total elapsed: `16.004 ms`
- Detail errors: `0`

Cleanup result:

- Requested FAQ rows: `6`
- Deleted FAQ rows: `6`
- Delete status: `DELETE 6`

## Budget Checks

All configured budgets passed:

- `error_rate <= 0.0`: passed
- `p95_ms <= 1000`: passed
- `max_ms <= 4000`: passed
- `case_error_rate <= 0.0`: passed for all 12 cases
- `case_p95_ms <= 1000`: passed for all 12 cases
- `case_max_ms <= 4000`: passed for all 12 cases
- `detail_max_ms <= 1000`: passed

## Interpretation

The local route path held under mixed-corpus concurrent reads. Each case stayed
inside its expected corpus/FAQ result, miss cases stayed empty, the route stayed
inside the latency budgets, and the detail route returned the full generated FAQ
shape after the seed-data fix.

This still does not prove cross-account hosted-route concurrency because this
local run used one bearer token mapped to one account. Cross-account route stress
needs multiple real tokens or a dedicated local auth fixture. It also does not
prove deployed-host behavior; that remains blocked on the deployed API URL,
bearer token, and matching account ID.
