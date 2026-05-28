# PR-Content-Ops-FAQ-Search-E2E-Runbook

## Why this slice exists

The seeded hosted FAQ search e2e runner now composes DB seed, hosted route
concurrency, detail hydration, cleanup, child-result summaries, and per-case
route budgets. Operators can run the real flow from the CLI, but the validation
docs only cover the route concurrency child smoke. That leaves the safest
one-command go-live probe under-documented and makes it easier to run the route
smoke without seeding/cleanup proof.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Add an operator runbook for the seeded hosted FAQ search e2e runner.
2. Document the required inputs, recommended command, budget flags, detail
   behavior, cleanup behavior, and result fields.
3. Add a doc contract test that parses the documented command flags against the
   real seeded e2e parser.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Search-E2E-Runbook.md` | Plan contract for this operator-doc slice. |
| `docs/extraction/validation/content_ops_faq_seeded_route_e2e_runbook.md` | Operator runbook for the seeded hosted e2e command. |
| `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` | Verify the runbook command stays parseable by the actual CLI. |

## Mechanism

The new runbook shows a single recommended command for
`scripts/smoke_content_ops_faq_search_seeded_route_e2e.py` using environment
variables for database URL, deployed API host, token, and account ID. The
command includes aggregate route budgets and the per-case route budgets added in
the previous slice.

The test extracts the first fenced bash command containing the seeded e2e script,
normalizes shell continuations and environment substitutions into concrete
placeholder values, then parses the resulting arguments with the real
`_build_parser()`. It asserts the expected budget, detail, cleanup, and artifact
flags survive the documentation example.

## Intentional

- No runner behavior changes; this is documentation plus a parser-backed doc
  contract.
- No hosted live invocation is added because credentials and deployed hosts are
  operator/runtime concerns.
- No production SLO values are introduced. The runbook labels sample latency
  values as placeholders that should be replaced with repeated hosted baselines.

## Deferred

- Parked hardening: none. `HARDENING.md` was scanned and has no active FAQ
  search entries touching this runner or docs.
- A host-install-runbook cross-link can follow if operators want the seeded
  route e2e command repeated outside the validation docs.

## Verification

- `pytest` on `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` - 57 passed.
- `py_compile` on `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` - passed.
- `scripts/audit_plan_doc.py` on `plans/PR-Content-Ops-FAQ-Search-E2E-Runbook.md` - passed.
- `scripts/audit_plan_code_consistency.py` on `plans/PR-Content-Ops-FAQ-Search-E2E-Runbook.md` - passed.
- `scripts/audit_extracted_pipeline_ci_enrollment.py` - 122 matching tests enrolled.
- `git diff --check` - passed.
- `scripts/validate_extracted_content_pipeline.sh` - passed.
- `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` on `extracted_content_pipeline` - passed.
- `scripts/audit_extracted_standalone.py` with fail-on-debt - passed.
- `scripts/check_ascii_python.sh` - passed.
- `scripts/run_extracted_pipeline_checks.sh` - 2571 passed, 7 skipped, 1 warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 82 |
| Runbook | 102 |
| Tests | 38 |
| **Total** | **222** |
