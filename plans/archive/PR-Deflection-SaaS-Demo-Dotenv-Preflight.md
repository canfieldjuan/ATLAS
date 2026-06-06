## Why this slice exists

`HARDENING.md` has a parked FAQ-deflection local verification bug:
`test_script_preflight_uses_atlas_db_settings_fallback` removes required hosted
route env vars before spawning the SaaS demo smoke, but the subprocess reloads
repo-root `.env` and can silently repopulate them. In provisioned workspaces,
that makes the local extracted mirror fail even though CI's clean environment
passes.

The local database is also available at
`postgresql://atlas@localhost:5433/atlas`, so full local mirrors can run
DB-backed tests instead of skipping them.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection-backend

Slice phase: Production hardening

1. Add an explicit dotenv opt-out for the SaaS demo route smoke.
2. Use that opt-out in the subprocess preflight regression so popped env vars
   cannot be reloaded from repo-local dotenv files.
3. Keep normal smoke runs unchanged: `.env` and `.env.local` still load by
   default.
4. Drain the fixed `HARDENING.md` entry.

### Files touched

- `scripts/smoke_content_ops_faq_saas_demo_route_e2e.py`
- `tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py`
- `HARDENING.md`
- `plans/PR-Deflection-SaaS-Demo-Dotenv-Preflight.md`

## Mechanism

`_load_dotenv_files()` will return early when `ATLAS_DISABLE_DOTENV=1` is set.
The smoke continues to load repo `.env` and `.env.local` by default for real
operator runs. The failing preflight test sets only this opt-out in the
subprocess env, then keeps the existing strict assertion that missing hosted
inputs produce exit code 2 while Atlas DB settings still provide a database DSN.

For verification in a live-provisioned checkout, the full extracted mirror
should be run with a local database URL and blank hosted smoke envs:

```bash
EXTRACTED_DATABASE_URL=postgresql://atlas@localhost:5433/atlas \
ATLAS_API_BASE_URL= ATLAS_B2B_JWT= ATLAS_TOKEN= ATLAS_ACCOUNT_ID= \
ATLAS_FAQ_SEARCH_ACCOUNT_ID= ATLAS_DEFLECTION_SUBMIT_BLOB_URL= \
ATLAS_DEFLECTION_SUBMIT_CSV_FILE= ATLAS_DEFLECTION_COMPANY_NAME= \
ATLAS_DEFLECTION_CONTACT_EMAIL= ATLAS_DEFLECTION_SUPPORT_PLATFORM= \
ATLAS_DEFLECTION_PORTFOLIO_RESULT_URL= ATLAS_DEFLECTION_REQUEST_ID= \
bash scripts/run_extracted_pipeline_checks.sh
```

That exercises the local DB-backed tests rather than skipping them.

## Intentional

- This does not change hosted route validation, seeding, cleanup, or live smoke
  behavior.
- The opt-out is env-only rather than a CLI flag because the bug happens before
  normal argument defaults are safe to compute.
- This slice does not alter the seed script dotenv behavior; the failing
  preflight path is in the route smoke subprocess.

## Deferred

- Parked hardening: none; this drains the SaaS demo preflight dotenv item.

## Verification

- `pytest tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py -q`
  - 26 passed in 0.99s.
- `bash scripts/validate_extracted_content_pipeline.sh`
  - Passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - Passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt`
  - Passed.
- `bash scripts/check_ascii_python.sh`
  - Passed.
- `EXTRACTED_DATABASE_URL=postgresql://atlas@localhost:5433/atlas ATLAS_API_BASE_URL= ATLAS_B2B_JWT= ATLAS_TOKEN= ATLAS_ACCOUNT_ID= ATLAS_FAQ_SEARCH_ACCOUNT_ID= ATLAS_DEFLECTION_SUBMIT_BLOB_URL= ATLAS_DEFLECTION_SUBMIT_CSV_FILE= ATLAS_DEFLECTION_COMPANY_NAME= ATLAS_DEFLECTION_CONTACT_EMAIL= ATLAS_DEFLECTION_SUPPORT_PLATFORM= ATLAS_DEFLECTION_PORTFOLIO_RESULT_URL= ATLAS_DEFLECTION_REQUEST_ID= bash scripts/run_extracted_pipeline_checks.sh`
  - 2924 passed, 1 warning in 52.13s.
- `bash scripts/local_pr_review.sh --allow-dirty --current-pr-body-file .git/pr-deflection-saas-demo-dotenv-preflight-body.md`
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| SaaS demo smoke | ~5 |
| Tests | ~5 |
| `HARDENING.md` | ~20 removed |
| Plan doc | ~75 |
| **Total** | **~105** |

Under the 400 LOC soft cap.
