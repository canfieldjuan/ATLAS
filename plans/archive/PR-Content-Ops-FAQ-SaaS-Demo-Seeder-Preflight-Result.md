# PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Preflight-Result

## Why this slice exists

The SaaS FAQ demo seeder is now the operator handoff path for loading the
synthetic B2B SaaS FAQ into DB-backed search, but validation failures still exit
before writing the requested `--output-result` file. That recreates the same
operator visibility gap we just closed in the FAQ search smoke runners: a
failed setup leaves no machine-readable artifact for reviewers or deployment
scripts.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Production hardening

1. Write a compact preflight result payload when seeder argument validation
   fails.
2. Keep the existing exit code and human-readable validation message.
3. Add focused negative tests for seed-mode and cleanup-mode validation
   failures writing `--output-result`.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Preflight-Result.md` | Plan contract for this hardening slice. |
| `HARDENING.md` | Park runtime setup result-artifact follow-up. |
| `scripts/seed_content_ops_faq_saas_demo.py` | Emit a preflight JSON result before validation exits. |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | Cover fail-closed preflight result artifacts for seed and cleanup modes. |

## Mechanism

`main()` already owns argument parsing, output-result writing, and final exit
code selection. This slice keeps validation in `main()`, but when
`_validate_args(...)` returns errors it builds:

```python
{
    "phase": "preflight",
    "ok": False,
    "errors": [...],
    "mode": "seed" | "cleanup",
}
```

Then it writes the payload to `--output-result`, prints the same joined error
message through `SystemExit`, and returns the same failure behavior.

## Intentional

- No database, generator, search projection, cleanup SQL, or route behavior
  changes.
- Runtime pool/connection failures stay deferred; this slice only closes the
  validation path that is already locally reproducible without a database.
- The preflight result omits token/database URL values so the artifact stays
  safe to share.

## Deferred

- Runtime setup failures after validation can get their own result-artifact
  slice if operators hit missing `asyncpg` or connection errors.
- Parked hardening: `SaaS demo seeder runtime setup result artifact` in
  `HARDENING.md`.

## Verification

- `python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py -q` - 14 passed.
- `python -m py_compile scripts/seed_content_ops_faq_saas_demo.py tests/test_content_ops_faq_saas_demo_corpus.py` - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Preflight-Result.md` - passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` - passed with 122 matching tests enrolled.
- `git diff --check` - passed.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - passed with 2495 passed, 7 skipped, 1 warning.

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 87 |
| Hardening note | 11 |
| Script | 10 |
| Tests | 57 |
| **Total** | **165** |
