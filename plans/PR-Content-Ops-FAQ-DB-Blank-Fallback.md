# PR-Content-Ops-FAQ-DB-Blank-Fallback

## Why this slice exists

PR-Content-Ops-FAQ-SaaS-Demo-DB-Settings-Fallback added a guarded Atlas DB
settings fallback for the SaaS FAQ demo seed and hosted route smoke. The
runbook examples still pass an explicit database URL argument built from
`EXTRACTED_DATABASE_URL`/`DATABASE_URL`.
When both URL env vars are empty, that explicit blank CLI value bypasses the
parser default and defeats the fallback the prior slice introduced.

This slice closes that integration gap before the next hosted proof attempt.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Functional validation

1. Normalize blank database URL CLI values in the SaaS demo seed and one-command
   route E2E scripts through the same guarded fallback used by parser defaults.
2. Keep fail-closed behavior when neither URL envs nor explicit Atlas DB target
   settings are present.
3. Update the SaaS demo route runbook/preflight examples so they rely on script
   defaults instead of passing an explicit empty database URL.

### Files touched

| File | Purpose |
|---|---|
| `scripts/seed_content_ops_faq_saas_demo.py` | Treat blank database URL CLI input as missing and re-run guarded default discovery. |
| `scripts/smoke_content_ops_faq_saas_demo_route_e2e.py` | Apply the same normalization before preflight validation. |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | Cover seed CLI blank database URL fallback and fail-closed behavior. |
| `tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py` | Cover route E2E blank database URL fallback and fail-closed behavior. |
| `docs/extraction/validation/content_ops_faq_saas_demo_route_case_runbook.md` | Remove explicit empty DB URL arguments from examples. |
| `docs/extraction/validation/content_ops_faq_saas_demo_route_preflight_2026-05-29.md` | Align recorded next-run examples with fallback behavior. |
| `plans/PR-Content-Ops-FAQ-DB-Blank-Fallback.md` | Slice contract. |

## Mechanism

Both scripts normalize parsed args before validation:

```python
def _normalize_args(args):
    if not str(args.database_url or "").strip():
        args.database_url = _default_database_url()
    return args
```

The fallback itself remains guarded by explicit `ATLAS_DB_HOST` or
`ATLAS_DB_SOCKET_PATH`, so blank CLI input cannot convert localhost defaults
into a hosted-proof database.

## Intentional

- This does not broaden what counts as a valid DB fallback; it only makes blank
  CLI input behave like an omitted database URL.
- This does not change deployed API URL, token, or account-id handling. Those
  remain required hosted proof inputs.
- This keeps the examples short by relying on the scripts' existing env/default
  discovery instead of adding shell conditionals.

## Deferred

- Live seeded SaaS FAQ route E2E remains deferred until deployed API base URL,
  bearer token, and account id are configured.
- Parked hardening: none.

## Verification

To run before opening the PR:

```bash
python -m py_compile scripts/seed_content_ops_faq_saas_demo.py scripts/smoke_content_ops_faq_saas_demo_route_e2e.py tests/test_content_ops_faq_saas_demo_corpus.py tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py
EXTRACTED_PIPELINE_STANDALONE=1 python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py -q
bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/content-ops-faq-db-blank-fallback-pr-body.md
```

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 87 |
| Script normalization | 16 |
| Tests | 71 |
| Docs | 6 |
| **Total** | **180** |
