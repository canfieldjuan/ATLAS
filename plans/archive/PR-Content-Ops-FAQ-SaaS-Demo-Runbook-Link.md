# PR-Content-Ops-FAQ-SaaS-Demo-Runbook-Link

## Why this slice exists

PR-Content-Ops-FAQ-SaaS-Demo-Route-Case-Runbook added the demo-specific
seed-to-hosted-route validation runbook, but host operators still only see the
generic seeded route e2e runbook from the extracted package install guide. That
keeps the checked B2B SaaS demo path discoverable only if someone already knows
the exact docs file name.

This slice adds the thinnest discoverability link from the existing host
operator runbook to the SaaS demo validation runbook.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Functional validation

1. Link the SaaS demo route-case validation runbook from
   `extracted_content_pipeline/docs/host_install_runbook.md` near the existing
   hosted FAQ search go-live validation note.
2. Add a focused doc-link test proving the host runbook references the SaaS demo
   runbook and the target file exists.
3. Keep runtime scripts, API code, and validation commands unchanged.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Runbook-Link.md` | Plan contract for this discoverability slice. |
| `extracted_content_pipeline/docs/host_install_runbook.md` | Link host operators to the SaaS demo route-case validation runbook. |
| `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` | Verify the host runbook link points to the checked SaaS demo validation runbook. |

## Mechanism

The host install runbook already has a hosted FAQ search validation paragraph.
This slice extends that paragraph with the demo-specific path:

```text
docs/extraction/validation/content_ops_faq_saas_demo_route_case_runbook.md
```

The test reads `host_install_runbook.md`, asserts the relative path is present,
and asserts the linked runbook exists on disk. That is the same parser-light
pattern already used for the generic seeded route e2e runbook link.

## Intentional

- No changes to the SaaS demo runbook commands. PR-Content-Ops-FAQ-SaaS-Demo-
  Route-Case-Runbook already parser-pinned them.
- No broad docs cleanup. This slice only makes the existing validation artifact
  reachable from the host operator surface.
- No live validation run. This is a discoverability slice, not a hosted
  environment execution slice.

## Deferred

- Parked hardening: none. `HARDENING.md` was scanned; no active FAQ-search item
  is required for this docs-link slice.
- Future robust-testing slices can run the linked SaaS demo validation against a
  deployed host and record the result artifact.

## Verification

- `python -m py_compile tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` - passed.
- `python -m pytest tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q` - 62 passed.
- `python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Runbook-Link.md` - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Runbook-Link.md` - passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` - passed; 122 matching tests enrolled.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed; 0 Atlas runtime import findings.
- `bash scripts/check_ascii_python.sh` - passed.
- `git diff --check` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 83 |
| Host runbook | 3 |
| Test | 9 |
| **Total** | **95** |
