# PR-Content-Ops-FAQ-SaaS-Demo-Migration-Env-Isolation

## Why this slice exists

PR #1065 added a parser pin for the SaaS FAQ demo runbook migration command.
Post-merge review found that the test can fail in developer environments that
already set `EXTRACTED_DATABASE_URL` or `DATABASE_URL`: the migration CLI
defaults from those env vars, while the test is trying to prove the documented
bare command carries no explicit DSN.

## Scope (this PR)

Ownership lane: ai-content-ops/faq-validation

Slice phase: Functional validation.

1. Isolate the migration parser pin from ambient database URL env vars.
2. Keep the runbook and migration CLI behavior unchanged.

### Files touched

- `tests/test_content_ops_faq_saas_demo_corpus.py`
- `plans/PR-Content-Ops-FAQ-SaaS-Demo-Migration-Env-Isolation.md`

## Mechanism

The migration-command parser test now accepts `monkeypatch` and clears
`EXTRACTED_DATABASE_URL` and `DATABASE_URL` immediately before parsing the
documented runbook command. That keeps the assertion focused on the command
text rather than on the runner's shell.

## Intentional

- No runbook changes: the documented bare migration command is still valid
  because the CLI intentionally supports env-default DSNs.
- No CLI changes: this is a test-isolation defect, not a production parser
  defect.

## Deferred

Parked hardening: none.

## Verification

```bash
python -m py_compile tests/test_content_ops_faq_saas_demo_corpus.py
python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py -q
DATABASE_URL=postgres://example EXTRACTED_DATABASE_URL=postgres://example2 python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py::test_saas_demo_route_case_runbook_migration_command_matches_parser -q
```

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Test env isolation | ~6 |
| Plan doc | ~53 |
| **Total** | **~59** |
