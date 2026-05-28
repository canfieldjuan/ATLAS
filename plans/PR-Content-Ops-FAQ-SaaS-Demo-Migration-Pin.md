# PR-Content-Ops-FAQ-SaaS-Demo-Migration-Pin

## Why this slice exists

The SaaS demo route-case runbook now documents the one-command hosted route e2e
smoke, and parser-pins the seed, route, and wrapper commands. Review has twice
noted the remaining unpinned command: the migration preamble. This lane already
had one migration-command drift bug, so leaving the preamble unpinned keeps a
known operator footgun open.

This slice adds the smallest parser-backed guard for the migration command in
the SaaS demo runbook.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Functional validation

1. Import the migration CLI in the SaaS demo runbook test file.
2. Parse the documented migration command with the real migration parser.
3. Assert the runbook does not mention the old invalid migration-runner command.
4. Keep runbook prose and runtime behavior unchanged.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Migration-Pin.md` | Plan contract for this parser-pin slice. |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | Add migration preamble parser coverage for the SaaS demo runbook. |

## Mechanism

The test extracts:

```bash
python scripts/run_extracted_content_pipeline_migrations.py
```

from `content_ops_faq_saas_demo_route_case_runbook.md`, parses it with
`scripts/run_extracted_content_pipeline_migrations.py`'s `_parse_args`, and
asserts the obsolete `extracted_content_pipeline/storage/migration_runner.py
--apply` command is absent.

## Intentional

- No docs edit. The command is already correct; this slice pins it.
- No live migration run. The parser guard proves CLI shape without requiring a
  database.
- No broad runbook command helper refactor. The existing test extraction helper
  is enough for this narrow check.

## Deferred

- Parked hardening: none. `HARDENING.md` was scanned; no active FAQ-search item
  is required for this parser-pin slice.

## Verification

- `python -m py_compile tests/test_content_ops_faq_saas_demo_corpus.py` - passed.
- `python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py -q` - 25 passed.
- `python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Migration-Pin.md` - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Migration-Pin.md` - passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` - passed; 123 matching tests enrolled.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed; 0 Atlas runtime import findings.
- `bash scripts/check_ascii_python.sh` - passed.
- `git diff --check` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 76 |
| Test | 23 |
| **Total** | **99** |
