# HARDENING.md

Park non-blocking hardening discoveries here when they are not required for the
current thin slice to function. Newest entries go first.

Do not use this file to defer issues that break the slice's real flow, AGENTS
contract, tests, CI, security, or data truthfulness. Those must be fixed inline
or the slice must stop.

When starting a slice, scan this file for entries touching the same ownership
lane or files. Fix only entries required for that slice to function; otherwise
leave them parked and mention the reason in the plan's Deferred section if they
were considered. Periodically drain stale entries or promote them into the debt
register under `docs/technical-debt/`.

## Entry Format

```md
## YYYY-MM-DD

### <short title>
- File/location:
- Description:
- Why it matters:
- Effort: S / M / L
- Category: correctness / polish / tech-debt / security
- Owner/session:
- Found during:
```

## Parked Items

## 2026-06-02

### SaaS demo preflight subprocess reloads live repo dotenv
- File/location: `tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py::test_script_preflight_uses_atlas_db_settings_fallback` / `scripts/smoke_content_ops_faq_saas_demo_route_e2e.py`
- Description: The test removes required env vars before spawning the smoke script, but the subprocess reloads the repo-root `.env`, so local checkouts with live `ATLAS_API_BASE_URL` / token / account envs can return success instead of missing-inputs.
- Why it matters: It makes `scripts/run_extracted_pipeline_checks.sh` fail locally in provisioned workspaces even when CI's clean environment should pass.
- Effort: S
- Category: tech-debt
- Owner/session: Codex FAQ deflection answer copy polish slice
- Found during: PR-Deflection-Answer-Copy-Polish
- Root cause (verified 2026-06-01, independent review): `_load_dotenv_files()` (called from `_build_parser`) runs `load_dotenv(ROOT / ".env")` then `load_dotenv(ROOT / ".env.local", override=True)`, repopulating the exact vars the test popped from the subprocess env. This workspace's `.env` carries 3 of them (`ATLAS_API_BASE_URL` / token / account), so preflight finds all required inputs present and exits 0 instead of 2. Reproduced locally as `assert 0 == 2`; passes in CI's clean checkout. `cwd=tmp_path` does not help because the script loads `ROOT/.env` by absolute path.
- Proposed fix: gate `_load_dotenv_files()` on an opt-out (`ATLAS_DISABLE_DOTENV=1`, or a `--no-dotenv` flag parsed before the load) and set it in `test_script_preflight_uses_atlas_db_settings_fallback`, so the popped vars cannot be reloaded from the on-disk dotenv. Real runs still load `.env` / `.env.local` unchanged.
- Acceptance criteria: (1) the test passes both in CI (clean env) and locally with a populated `ROOT/.env`; (2) with the opt-out set and required inputs absent, the script still exits 2 and emits the missing-input preflight errors (do not relax the assertion to accept 0, which would defeat the test); (3) without the opt-out, dotenv loading is byte-for-byte unchanged for real runs.
- Not-a-bug note: the 10 co-reported mirror skips are unrelated Postgres integration tests that skip on `EXTRACTED_DATABASE_URL or DATABASE_URL is required`; verified they pass when `EXTRACTED_DATABASE_URL=postgresql://atlas@localhost:5433/atlas` is set (those 6 files: 197 passed). No code change needed for the skips.

## 2026-05-29

### atlas-intel-ui npm audit vulnerabilities
- File/location: `atlas-intel-ui/package-lock.json`
- Description: `npm ci` reports 6 dependency audit findings (2 moderate, 4 high).
- Why it matters: Dependency vulnerabilities can become deploy-time security exposure, but resolving them may require package upgrades outside this UI rendering slice.
- Effort: M
- Category: security
- Owner/session: Codex FAQ deflection report UI slice
- Found during: PR-FAQ-Deflection-Report-UI-Readonly

> **Atlas blog / deep-dive content pipeline** (`content-ops/blog-*` ownership
> lanes): parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md),
> kept separate to avoid append-collisions with the concurrent
> content-ops-station sessions. Scan that file too when working those lanes.
