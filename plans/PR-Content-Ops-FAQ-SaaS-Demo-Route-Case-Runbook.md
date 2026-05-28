# PR-Content-Ops-FAQ-SaaS-Demo-Route-Case-Runbook

## Why this slice exists

PR-Content-Ops-FAQ-SaaS-Demo-Route-Case-Output closed the code handoff between
the SaaS FAQ demo seeder and the hosted FAQ search route smoke by emitting a
route case file with search and detail expectations. The operator path is still
implicit: a host operator has to infer which two commands prove the seeded SaaS
demo through the deployed route.

This slice documents that thinnest validation path and pins the commands against
the actual parsers so future flag drift does not break the demo handoff quietly.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Functional validation

1. Add a SaaS demo route-case validation runbook that shows the two-command
   flow: seed the checked SaaS FAQ into Postgres while writing a route case
   file, then run the hosted route concurrency smoke against that case file.
2. State what the route case proves: first-result identity and hydrated detail
   fields for the generated FAQ.
3. Add focused tests proving both runbook commands parse with the real CLIs and
   share the same route case path.
4. Keep runtime seeder, route smoke, and API code unchanged.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Route-Case-Runbook.md` | Plan contract for this runbook validation slice. |
| `docs/extraction/validation/content_ops_faq_saas_demo_route_case_runbook.md` | Operator runbook for seeded SaaS demo hosted route validation. |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | Parser-backed tests for the runbook commands and shared route-case handoff path. |

## Mechanism

The runbook uses the seeder's existing `--route-case-file-output` flag:

```bash
python scripts/seed_content_ops_faq_saas_demo.py \
  --route-case-file-output /tmp/faq-saas-demo-route-cases.json
```

The second command passes the same file to
`scripts/smoke_content_ops_faq_search_route_concurrency.py` with
`--require-detail` and fail-closed aggregate/per-case budgets. The smoke already
loads the emitted `expected_first_*` and `expected_detail_*` fields; this slice
only documents the operational composition.

The test imports both scripts and parses the documented commands through their
real argparse parsers. It also asserts the route smoke's `--case-file` value
matches the seeder's `--route-case-file-output` value, so a future doc edit
cannot silently split the handoff.

## Intentional

- No new one-command wrapper. The seeded e2e runner already covers the generic
  DB-backed hosted route path; this runbook is specifically for the checked
  SaaS demo artifact and the emitted route case file.
- No live DB or hosted route invocation in unit tests. This slice validates the
  documented operator contract; live execution remains host/environment work.
- No cleanup helper changes. The seeder already returns the FAQ id and cleanup
  mode can remove it when an operator intentionally keeps seeded data.

## Deferred

- Parked hardening: none. `HARDENING.md` was scanned; no active FAQ-search item
  is required for this runbook slice.
- Future robust-testing slices can use this runbook as the operator entry point
  for repeated hosted demo runs with real latency thresholds.

## Verification

- `python -m py_compile tests/test_content_ops_faq_saas_demo_corpus.py` - passed.
- `python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py -q` - 23 passed.
- `python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Route-Case-Runbook.md` - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Route-Case-Runbook.md` - passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` - passed; 122 matching tests enrolled.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed; 0 Atlas runtime import findings.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - 2590 passed, 7 skipped.
- `git diff --check` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 94 |
| Runbook doc | 133 |
| Tests | 66 |
| **Total** | **293** |
