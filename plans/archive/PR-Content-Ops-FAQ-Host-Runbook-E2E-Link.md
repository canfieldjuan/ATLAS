# PR-Content-Ops-FAQ-Host-Runbook-E2E-Link

## Why this slice exists

`PR-Content-Ops-FAQ-Search-E2E-Runbook` added the seeded hosted FAQ search e2e
validation runbook, but the host install runbook still only points operators at
offline FAQ generation and the persisted lifecycle smoke. A host operator
following the install docs can miss the preferred go-live probe for the hosted
FAQ search route.

This slice closes the deferred discoverability gap from the seeded e2e runbook
slice without changing runtime behavior.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Add a host-install-runbook pointer from the FAQ lifecycle section to the
   seeded hosted FAQ search e2e validation runbook.
2. Add a focused doc test that keeps the cross-link and referenced validation
   runbook path present.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Host-Runbook-E2E-Link.md` | Plan contract for this docs-discoverability slice. |
| `extracted_content_pipeline/docs/host_install_runbook.md` | Link host operators to the seeded hosted FAQ search e2e validation runbook. |
| `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` | Verify the host runbook links to the seeded e2e runbook and that the target exists. |

## Mechanism

The host install runbook gains one short paragraph after the persisted FAQ
lifecycle smoke command. It points hosted operators to
`docs/extraction/validation/content_ops_faq_seeded_route_e2e_runbook.md` when
they need to seed search rows, hit the deployed route, hydrate detail, and clean
up in one go-live probe.

The test reads `extracted_content_pipeline/docs/host_install_runbook.md`,
asserts that exact validation-doc path is present, and asserts the target file
exists. This is a lightweight doc contract rather than another parser test
because the target runbook already pins its command against the real CLI parser.

## Intentional

- No CLI, route, seed, database, or generated-asset behavior changes.
- The full seeded e2e command is not duplicated in the host runbook; duplication
  would recreate command drift. The validation runbook remains the canonical
  command source.

## Deferred

- Parked hardening: none. `HARDENING.md` was scanned and has no active FAQ
  search entries touching this docs surface.
- No broader host install runbook restructure; this only closes seeded e2e
  discoverability.

## Verification

- `pytest` on `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` - 58 passed.
- `py_compile` on `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` - passed.
- `scripts/audit_plan_doc.py` on `plans/PR-Content-Ops-FAQ-Host-Runbook-E2E-Link.md` - passed.
- `scripts/audit_plan_code_consistency.py` on `plans/PR-Content-Ops-FAQ-Host-Runbook-E2E-Link.md` - passed.
- `scripts/audit_extracted_pipeline_ci_enrollment.py` - 122 matching tests enrolled.
- `git diff --check` - passed.
- `scripts/validate_extracted_content_pipeline.sh` - passed.
- `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` on `extracted_content_pipeline` - passed.
- `scripts/audit_extracted_standalone.py` with fail-on-debt - passed.
- `scripts/check_ascii_python.sh` - passed.
- `scripts/run_extracted_pipeline_checks.sh` - 2572 passed, 7 skipped, 1 warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 80 |
| Host runbook | 5 |
| Tests | 9 |
| **Total** | **94** |
