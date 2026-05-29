# PR-Content-Ops-FAQ-SaaS-Demo-DB-Settings-Fallback

## Why this slice exists

PR-Content-Ops-FAQ-SaaS-Demo-Route-Preflight proved the hosted SaaS FAQ route
E2E cannot run from this checkout because required inputs are missing. One of
those inputs is the database URL, but other Content Ops Postgres smokes already
derive it from `atlas_brain.storage.config.db_settings.dsn` when
`EXTRACTED_DATABASE_URL`/`DATABASE_URL` are not set.

The SaaS demo seed/E2E scripts are out of line with that established pattern,
so their preflight reports the database as missing even when explicit Atlas DB
target settings are available. This slice removes that avoidable blocker
without letting Atlas' localhost defaults satisfy hosted proof by themselves.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Functional validation

1. Add a guarded `db_settings.dsn` fallback used by related FAQ/Postgres smokes
   to the SaaS demo seeder and one-command E2E runner.
2. Keep explicit `--database-url`, `EXTRACTED_DATABASE_URL`, and `DATABASE_URL`
   precedence unchanged.
3. Add focused tests for env precedence and fallback behavior on both scripts.
4. Update the SaaS demo route validation notes/runbook to name the fallback.

### Files touched

| File | Purpose |
|---|---|
| `scripts/seed_content_ops_faq_saas_demo.py` | Derive the default database URL from explicit Atlas DB target settings when URL envs are absent. |
| `scripts/smoke_content_ops_faq_saas_demo_route_e2e.py` | Use the same guarded fallback for the one-command SaaS route E2E. |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | Cover seeder fallback and env precedence. |
| `tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py` | Cover E2E fallback and env precedence. |
| `docs/extraction/validation/content_ops_faq_saas_demo_route_case_runbook.md` | Document accepted DB input forms. |
| `docs/extraction/validation/content_ops_faq_saas_demo_route_preflight_2026-05-29.md` | Refine the prior missing-input note with the new fallback behavior. |
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-DB-Settings-Fallback.md` | Slice contract. |

## Mechanism

Both scripts get a local `_default_database_url()` helper:

```python
raw = _env("EXTRACTED_DATABASE_URL", "DATABASE_URL")
if raw:
    return raw
if not _env("ATLAS_DB_HOST", "ATLAS_DB_SOCKET_PATH"):
    return ""
try:
    spec = importlib.util.spec_from_file_location(
        "_atlas_storage_config_for_saas_demo",
        ROOT / "atlas_brain/storage/config.py",
    )
    ...
    spec.loader.exec_module(module)
except Exception:
    return ""
db_settings = getattr(module, "db_settings", None)
return str(getattr(db_settings, "dsn", "") or "").strip()
```

The parser default changes from `_env("EXTRACTED_DATABASE_URL", "DATABASE_URL")`
to `_default_database_url()`. Existing validation remains fail-closed if no URL
or settings-derived DSN exists. The Atlas settings fallback only activates when
an explicit `ATLAS_DB_HOST` or `ATLAS_DB_SOCKET_PATH` target setting is present,
so the default `localhost:5433/atlas` settings cannot accidentally satisfy a
hosted route proof. The config file is loaded directly so CI can read
`db_settings.dsn` without importing `atlas_brain.storage.__init__` and its
runtime DB dependencies.

## Intentional

- This does not set the deployed API base URL, bearer token, or account id.
  Those are hosted proof inputs and should remain operator-provided.
- This does not attempt a live DB connection. It only fixes default discovery;
  connection success is still validated by the existing seed/E2E runtime path.
- This treats only explicit Atlas DB target settings as fallback evidence. Other
  `ATLAS_DB_*` tuning fields do not prove the operator selected a database
  target for hosted proof.
- This deliberately loads `atlas_brain/storage/config.py` by file path instead
  of package import because the package initializer pulls in runtime database
  modules that are not required for preflight discovery.

## Deferred

- Live seeded SaaS FAQ route E2E remains deferred until the deployed API base
  URL, bearer token, and account id are configured.
- Parked hardening: none.

## Verification

To run before opening the PR:

```bash
python -m py_compile scripts/seed_content_ops_faq_saas_demo.py scripts/smoke_content_ops_faq_saas_demo_route_e2e.py tests/test_content_ops_faq_saas_demo_corpus.py tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py
EXTRACTED_PIPELINE_STANDALONE=1 python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py -q
bash scripts/run_extracted_pipeline_checks.sh
env -u EXTRACTED_DATABASE_URL -u DATABASE_URL -u ATLAS_API_BASE_URL -u ATLAS_B2B_JWT -u ATLAS_TOKEN -u ATLAS_FAQ_SEARCH_ACCOUNT_ID -u ATLAS_ACCOUNT_ID ATLAS_DB_HOST=db-settings-host ATLAS_DB_PORT=5432 ATLAS_DB_DATABASE=atlas_settings ATLAS_DB_USER=atlas_settings_user ATLAS_DB_PASSWORD=atlas_settings_password python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py --preflight-only --json --output-result tmp/faq_saas_demo_route_preflight_db_settings_20260529/result5.json
env -u EXTRACTED_DATABASE_URL -u DATABASE_URL -u ATLAS_API_BASE_URL -u ATLAS_B2B_JWT -u ATLAS_TOKEN -u ATLAS_FAQ_SEARCH_ACCOUNT_ID -u ATLAS_ACCOUNT_ID -u ATLAS_DB_HOST -u ATLAS_DB_SOCKET_PATH python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py --preflight-only --json --output-result tmp/faq_saas_demo_route_preflight_db_settings_20260529/result5_no_target.json
bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/content-ops-faq-saas-demo-db-settings-fallback-pr-body.md
```

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 113 |
| Script fallback | 59 |
| Tests | 132 |
| Docs | 43 |
| **Total** | **347** |
