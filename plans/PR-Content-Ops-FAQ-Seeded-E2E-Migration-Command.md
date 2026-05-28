# PR-Content-Ops-FAQ-Seeded-E2E-Migration-Command

## Why this slice exists

`PR-Content-Ops-FAQ-Search-E2E-Runbook` added a seeded hosted FAQ search e2e
runbook, but its migration prerequisite points operators at
`extracted_content_pipeline/storage/migration_runner.py --apply`. That file is
the library implementation, not the host CLI, and it does not define an
`--apply` flag. Following the runbook can therefore fail before the actual e2e
smoke starts.

This slice fixes the operator command at the source and pins it with a doc test.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Replace the seeded e2e runbook migration command with the real host-facing
   migration CLI.
2. Add a focused doc test that verifies the migration command uses
   `scripts/run_extracted_content_pipeline_migrations.py` and does not include
   the invalid library-module command.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Seeded-E2E-Migration-Command.md` | Plan contract for this operator-command correction. |
| `docs/extraction/validation/content_ops_faq_seeded_route_e2e_runbook.md` | Fix the migration prerequisite command. |
| `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` | Pin the migration command in the seeded e2e runbook. |

## Mechanism

The runbook's migration prerequisite now matches the host install runbook and
the existing migration CLI:

```bash
python scripts/run_extracted_content_pipeline_migrations.py
```

The new test reads the seeded e2e runbook and asserts that command is present,
that `scripts/run_extracted_content_pipeline_migrations.py` exists, and that the
invalid `extracted_content_pipeline/storage/migration_runner.py --apply` string
is absent.

## Intentional

- No migration runner behavior changes; this is a documentation contract fix.
- No dry-run command is added here. The seeded e2e runbook is a concise go-live
  validation runbook, while the host install runbook already documents migration
  preview and custom migration-table options.

## Deferred

- Parked hardening: none. `HARDENING.md` was scanned and has no active FAQ
  search entries touching this docs surface.
- No broader migration-doc cleanup outside the seeded e2e runbook.

## Verification

- `python -m pytest tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q` - 59 passed.
- `python -m py_compile tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` - passed.
- `python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-Seeded-E2E-Migration-Command.md && python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Seeded-E2E-Migration-Command.md` - passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` - passed; 122 matching tests enrolled.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed; 0 Atlas runtime import findings.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - 2573 passed, 7 skipped.
- `git diff --check` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 80 |
| Seeded e2e runbook | 2 |
| Tests | 9 |
| **Total** | **91** |
