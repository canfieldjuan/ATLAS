# PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Runtime-Result

## Why this slice exists

PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Preflight-Result made validation failures
write `--output-result`, then parked the adjacent runtime gap: after validation,
missing `asyncpg`, pool creation failures, repository errors, or cleanup errors
can still abort the SaaS FAQ demo seeder before a machine-readable artifact is
written.

This production-hardening slice drains that `HARDENING.md` item so deployment
operators get a result artifact for runtime failures as well as preflight
failures.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Production hardening

1. Add a compact runtime-failure result payload for post-validation exceptions.
2. Write and print that payload before returning exit code `1`.
3. Redact the configured database URL from runtime error messages before they
   are written to output.
4. Add focused tests for seed-mode and cleanup-mode runtime failures.
5. Remove the drained `HARDENING.md` entry.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Runtime-Result.md` | Plan contract for this hardening slice. |
| `HARDENING.md` | Replace the drained runtime result-artifact item with the narrower pool-close preservation follow-up. |
| `scripts/seed_content_ops_faq_saas_demo.py` | Emit safe runtime-failure result artifacts after validation. |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | Cover seed and cleanup runtime-failure result artifacts. |

## Mechanism

`main()` already owns validation, output writing, and exit-code selection. After
validation passes, it now wraps `asyncio.run(_run(args))` in a narrow
`except Exception` block. The failure payload keeps the same shared `errors`
list convention:

```python
{
    "phase": "runtime",
    "ok": False,
    "mode": "seed" | "cleanup",
    "errors": ["RuntimeError: ..."],
    "error": {"type": "RuntimeError", "message": "..."},
}
```

The message is sanitized with the exact configured database URL replaced by
`[redacted-database-url]` before printing or writing.

## Intentional

- No database, generator, search projection, cleanup SQL, or route behavior
  changes.
- No broad lifecycle refactor; this slice only guarantees a result artifact for
  exceptions that currently escape `main()`.
- Exit code `1` remains the runtime failure signal, matching the existing
  seed/cleanup failure behavior.

## Deferred

- Pool-close lifecycle metadata that preserves a successful seed payload while
  separately reporting close failure is deferred until operators hit that more
  specific case. This slice captures the failure instead of leaving no artifact.
- Parked hardening: `SaaS demo seeder pool-close result preservation` in
  `HARDENING.md`.

## Verification

- `python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py -q` - 16 passed.
- `python -m py_compile scripts/seed_content_ops_faq_saas_demo.py tests/test_content_ops_faq_saas_demo_corpus.py` - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Runtime-Result.md` - passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` - passed with 122 matching tests enrolled.
- `git diff --check` - passed.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - passed with 2497 passed, 7 skipped, 1 warning.

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 94 |
| Hardening note update | 8 |
| Script | 38 |
| Tests | 86 |
| **Total** | **226** |
