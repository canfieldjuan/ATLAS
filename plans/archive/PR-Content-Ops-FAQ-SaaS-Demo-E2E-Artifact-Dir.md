# PR-Content-Ops-FAQ-SaaS-Demo-E2E-Artifact-Dir

## Why this slice exists

The SaaS demo runbook now recommends the one-command hosted route e2e smoke and
pins all documented commands. The recommended command still omits
`--artifact-dir`, so operators get the top-level result artifact but the child
seed, route, cleanup, and route-case files are written under a temporary
directory that is removed after the process exits.

This slice makes the recommended command preserve child artifacts by default.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Functional validation

1. Add `--artifact-dir /tmp/faq-saas-demo-route-e2e-artifacts` to the
   recommended SaaS demo e2e smoke command.
2. Update the runbook result-check language to tell operators where child
   artifacts live.
3. Extend the parser-backed runbook test to pin the documented artifact
   directory.
4. Keep runtime wrapper behavior unchanged.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-E2E-Artifact-Dir.md` | Plan contract for this runbook artifact-preservation slice. |
| `docs/extraction/validation/content_ops_faq_saas_demo_route_case_runbook.md` | Preserve child artifacts in the recommended one-command smoke. |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | Parser-pin the documented artifact directory. |

## Mechanism

The wrapper already supports `--artifact-dir`. This slice only adds it to the
documented recommended invocation and asserts the parser sees:

```python
Path("/tmp/faq-saas-demo-route-e2e-artifacts")
```

No script code changes are required.

## Intentional

- No runtime changes. The wrapper already writes child artifacts to
  `--artifact-dir` when provided.
- No manual fallback command changes. The fallback commands already write their
  own explicit `/tmp/faq-saas-demo-*` artifacts.
- No live hosted run. Required hosted inputs are not present in this checkout.

## Deferred

- Parked hardening: none. `HARDENING.md` was scanned; no active FAQ-search item
  is required for this runbook slice.
- Future robust-testing slice: run the documented command against a deployed
  host and commit a validation report once hosted inputs are available.

## Verification

- `python -m py_compile tests/test_content_ops_faq_saas_demo_corpus.py` - passed.
- `python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py -q` - 25 passed.
- `python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-SaaS-Demo-E2E-Artifact-Dir.md` - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-SaaS-Demo-E2E-Artifact-Dir.md` - passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` - passed; 123 matching tests enrolled.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed; 0 Atlas runtime import findings.
- `bash scripts/check_ascii_python.sh` - passed.
- `git diff --check` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 80 |
| Runbook doc | 4 |
| Test | 1 |
| **Total** | **85** |
