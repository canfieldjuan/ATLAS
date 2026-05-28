# PR-Content-Ops-FAQ-SaaS-Demo-E2E-Runbook

## Why this slice exists

PR-Content-Ops-FAQ-SaaS-Demo-Route-E2E-Smoke added a one-command SaaS demo
hosted route smoke that composes seeding, route/detail validation, and cleanup.
The SaaS demo route-case runbook still presents the older manual seed and route
commands first, so operators can miss the safer cleanup-aware wrapper.

This slice updates the runbook to make the one-command smoke the recommended
path and pins that documented command against the actual wrapper parser.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Functional validation

1. Add the one-command SaaS demo hosted route e2e smoke to the SaaS demo
   validation runbook.
2. Make the existing manual seed/route commands a fallback for inspecting
   intermediate artifacts.
3. Add a focused parser-backed test for the documented wrapper command.
4. Keep runtime scripts, API, repository, and CI enrollment unchanged.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-E2E-Runbook.md` | Plan contract for this runbook update. |
| `docs/extraction/validation/content_ops_faq_saas_demo_route_case_runbook.md` | Document the cleanup-aware one-command SaaS demo route e2e smoke. |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | Parser-pin the documented one-command smoke invocation. |

## Mechanism

The runbook gains a new recommended command:

```bash
python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py ...
```

The previous seed/route commands remain in the file as a manual fallback when an
operator needs to inspect the route-case artifact between steps. The test imports
the wrapper script, extracts the fenced command from the runbook, parses it with
`_build_parser()`, and asserts `_validate_args(parsed) == []`.

## Intentional

- No wrapper behavior changes. The previous slice already tested the subprocess
  composition and fail-closed branches.
- No host install runbook change. It already links to this SaaS demo validation
  runbook.
- No live hosted run. Required hosted inputs are not present in this checkout.

## Deferred

- Parked hardening: none. `HARDENING.md` was scanned; no active FAQ-search item
  is required for this documentation slice.
- Future robust-testing slice: execute the documented wrapper against a deployed
  host and save the result artifact once the required hosted inputs are
  available.

## Verification

- `python -m py_compile tests/test_content_ops_faq_saas_demo_corpus.py` - passed.
- `python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py -q` - 24 passed.
- `python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-SaaS-Demo-E2E-Runbook.md` - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-SaaS-Demo-E2E-Runbook.md` - passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` - passed; 123 matching tests enrolled.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed; 0 Atlas runtime import findings.
- `bash scripts/check_ascii_python.sh` - passed.
- `git diff --check` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 82 |
| Runbook doc | 46 |
| Test | 28 |
| **Total** | **156** |
