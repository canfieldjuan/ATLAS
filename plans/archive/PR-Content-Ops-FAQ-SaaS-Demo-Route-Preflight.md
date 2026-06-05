# PR-Content-Ops-FAQ-SaaS-Demo-Route-Preflight

## Why this slice exists

The next natural FAQ-lane validation is the seeded B2B SaaS FAQ route E2E:
generate the checked SaaS FAQ, save and approve it in Postgres, verify the
search projection, and hit the deployed `/content-ops/faq-deflection-search`
route under concurrent demo traffic.

That live smoke requires four operator-provided inputs: a database URL, a
deployed Atlas API base URL, a bearer token, and the account id matching that
token. This checkout does not currently expose those required env names, so
this slice records the preflight state before attempting a DB write or hosted
route test.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Functional validation

1. Run the existing `scripts/smoke_content_ops_faq_saas_demo_route_e2e.py`
   preflight-only mode with the current checkout environment.
2. Record the non-secret result artifact shape and missing-input status.
3. Keep the live seeded route smoke deferred until the required hosted inputs
   are available.

### Files touched

| File | Purpose |
|---|---|
| `docs/extraction/validation/content_ops_faq_saas_demo_route_preflight_2026-05-29.md` | Validation note with command, result, and live-smoke blocker. |
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Route-Preflight.md` | Slice contract. |

## Mechanism

This slice uses the already-built preflight path:

```bash
python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py \
  --database-url "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}" \
  --base-url "$ATLAS_API_BASE_URL" \
  --token "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}" \
  --account-id "${ATLAS_FAQ_SEARCH_ACCOUNT_ID:-$ATLAS_ACCOUNT_ID}" \
  --preflight-only \
  --json \
  --output-result tmp/faq_saas_demo_route_preflight_20260529/result.json
```

The script reports only present/missing booleans for required inputs; it does
not echo secret values. Exit code `2` is expected when preflight inputs are
missing.

## Intentional

- No product code changes. The route E2E runner and seeded demo flow already
  exist; this slice only records whether the live validation can run from this
  checkout today.
- No synthetic fallback token, local API host, or fake database URL is used.
  The hosted proof mode is intentionally strict because localhost cannot prove
  deployed demo traffic behavior.

## Deferred

- Live seeded SaaS FAQ route E2E is deferred until `EXTRACTED_DATABASE_URL` or
  `DATABASE_URL`, `ATLAS_API_BASE_URL`, `ATLAS_B2B_JWT` or `ATLAS_TOKEN`, and
  `ATLAS_FAQ_SEARCH_ACCOUNT_ID` or `ATLAS_ACCOUNT_ID` are configured.
- Parked hardening: none.

## Verification

To run before opening the PR:

```bash
python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py \
  --database-url "${EXTRACTED_DATABASE_URL:-$DATABASE_URL}" \
  --base-url "$ATLAS_API_BASE_URL" \
  --token "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}" \
  --account-id "${ATLAS_FAQ_SEARCH_ACCOUNT_ID:-$ATLAS_ACCOUNT_ID}" \
  --preflight-only \
  --json \
  --output-result tmp/faq_saas_demo_route_preflight_20260529/result.json
bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/content-ops-faq-saas-demo-route-preflight-pr-body.md
```

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 92 |
| Validation note | 97 |
| **Total** | **189** |
