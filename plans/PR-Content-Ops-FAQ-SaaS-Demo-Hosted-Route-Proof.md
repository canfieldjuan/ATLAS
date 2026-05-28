# Content Ops FAQ SaaS Demo Hosted Route Proof

## Why this slice exists

The FAQ lane has already proven the synthetic B2B SaaS FAQ demo locally:
generation, Postgres persistence, search projection, hosted-route-shaped search,
detail hydration, cleanup, and a 200-request / concurrency-16 local stress run
all passed. The remaining production-relevant gap named in the validation notes
is the same one-command smoke against the deployed Atlas API and deployed
database.

This slice uses the existing hosted smoke and gitignored deployed credentials to
seed the checked SaaS FAQ demo into the deployed database, query the deployed
FAQ search route, hydrate detail, clean up the seeded FAQ, and record the
observed result without committing secrets.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Robust testing

1. Run the hosted SaaS demo FAQ route preflight with existing gitignored env
   values.
2. Run the deployed one-command FAQ route/detail smoke if preflight passes.
3. Record route latency, detail latency, seed/projection counts, and cleanup
   status in a validation note.
4. Do not change FAQ generator, search, API, or route implementation unless the
   hosted proof exposes a blocker required for the smoke to function honestly.

### Files touched

- `docs/extraction/validation/content_ops_faq_saas_demo_hosted_route_proof_2026-05-28.md` - hosted FAQ route validation record.
- `HARDENING.md` - parked non-blocking smoke/runbook polish found during the hosted attempt.
- `plans/PR-Content-Ops-FAQ-SaaS-Demo-Hosted-Route-Proof.md` - this plan.

## Mechanism

The slice uses the existing one-command runner:

```bash
python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py \
  --database-url "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}" \
  --base-url "$ATLAS_API_BASE_URL" \
  --token "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}" \
  --account-id "${ATLAS_FAQ_SEARCH_ACCOUNT_ID:-$ATLAS_ACCOUNT_ID}" \
  --route-requests 40 \
  --concurrency 8 \
  --max-error-rate 0 \
  --max-case-error-rate 0 \
  --max-p95-ms 1500 \
  --max-single-request-ms 3000 \
  --max-case-p95-ms 1500 \
  --max-case-single-request-ms 3000 \
  --max-detail-ms 2500 \
  --artifact-dir /tmp/faq-saas-demo-hosted-route-proof-artifacts \
  --output-result /tmp/faq-saas-demo-hosted-route-proof-result.json \
  --json
```

The script seeds through Postgres, validates the deployed HTTP route with the
bearer token, hydrates detail from returned FAQ ids, and deletes the seeded FAQ
unless `--keep-data` is passed. No secret values are printed or committed.

## Intentional

- No implementation changes are planned. This is a hosted FAQ proof slice, not
  a generator/search refactor.
- The local route stress result remains separate; this slice records only the
  deployed API/database behavior.
- Existing root `HARDENING.md` FAQ startup-noise items remain parked unless
  they block the hosted smoke. The hosted proof does not start local voice/ASR.
- Blog/generated-content PRs currently open in the repository are explicitly
  out of scope.

## Deferred

- If the deployed proof passes, repeated hosted threshold calibration can follow
  as a later production SLO slice.
- If the deployed proof fails on route latency or deployment wiring, the next
  slice should fix that source blocker before increasing traffic.
- Parked hardening added by this slice:
  `FAQ hosted route proof preflight accepts local API URLs` and
  `FAQ route concurrency result top-level query can disagree with case-file
  query`.
- Parked hardening considered but left parked: `Atlas startup migration check
  warns on missing b2b_campaigns.updated_at` and `Voice ASR auto-start blocks
  non-voice route validation on CUDA-less hosts`; neither is exercised by this
  deployed hosted-route smoke.

## Verification

- Command: python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py ... --preflight-only --json --output-result /tmp/faq-saas-demo-hosted-route-proof-preflight.json
  - Passed; required DB URL, base URL, token, and account id were present.
- Command: python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py ... --route-requests 40 --concurrency 8 ... --json
  - Failed on route requests with `[Errno 111] Connection refused`.
  - Seed passed: 36 source rows, 7 FAQ items, verification search matched the
    seeded FAQ in the configured database.
  - Cleanup passed: deleted the seeded FAQ row.
  - The configured API base URL was classified as local HTTP with a port, so
    this did not prove deployed-host behavior.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/content-ops-faq-saas-demo-hosted-route-proof-pr-body.md
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Validation note | ~90 |
| Hardening entry | ~20 |
| Plan doc | ~105 |
| Total | ~215 |
