# PR-Content-Ops-FAQ-Seeded-E2E-Detail-Budget

## Why this slice exists

`PR-Content-Ops-FAQ-Search-E2E-Case-Budgets` threaded aggregate and per-case
search-route latency budgets into the seeded hosted FAQ search e2e wrapper, but
explicitly deferred the detail-specific budget. The detail contract checker now
already supports `--max-detail-ms`, so the seeded e2e command should expose that
budget instead of leaving operators to run a second command when they want to
prove search-to-detail survivability.

This is a small robust-testing slice: it closes the existing wrapper gap without
changing the hosted route, the DB seed shape, or the detail contract checker.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Robust testing

1. Add `--max-detail-ms` to the seeded hosted FAQ search e2e runner.
2. Validate the budget is finite and positive, and fail closed if it is provided
   while detail checking is disabled.
3. Forward the budget to the existing detail contract checker.
4. Preserve the configured detail budget in the compact detail result artifact.
5. Document the budget in the seeded e2e runbook command and result checklist.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Seeded-E2E-Detail-Budget.md` | Plan contract for this robust-testing slice. |
| `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py` | Expose, validate, forward, and summarize the detail latency budget. |
| `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` | Cover parser, validation, forwarding, result-summary, and runbook command behavior. |
| `docs/extraction/validation/content_ops_faq_seeded_route_e2e_runbook.md` | Add the recommended detail latency budget to the hosted seeded e2e command. |

## Mechanism

The seeded e2e parser gains:

```bash
--max-detail-ms <milliseconds>
```

When present, `_validate_args` enforces a finite positive value and rejects the
combination with `--skip-detail-check` so a configured detail budget cannot be
silently ignored. `_detail_command(...)` passes the value through to
`check_content_ops_faq_search_route_contract.py`, which already enforces detail
latency after hydrating the selected FAQ.

The top-level e2e result still stores a compact child artifact rather than the
full detail checker payload, but the compact detail summary now includes
`max_detail_ms` so operators can prove which threshold was applied.

## Intentional

- No new detail timing implementation. The contract checker already measures
  `detail_elapsed_ms`; this wrapper only wires the existing budget into the
  seeded end-to-end path.
- No `--max-total-ms` wrapper flag in this slice. The deferred gap was
  detail-specific latency; route aggregate and per-case budgets are already
  wired separately.
- The runbook uses a placeholder threshold, consistent with the existing route
  latency placeholders, until repeated hosted runs establish production SLOs.

## Deferred

- Parked hardening: none. `HARDENING.md` was scanned and has no active FAQ
  search entries touching this wrapper surface.
- Production SLO selection remains deferred until repeated hosted runs produce
  stable search/detail latency baselines.

## Verification

- `python -m py_compile scripts/smoke_content_ops_faq_search_seeded_route_e2e.py tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` - passed.
- `python -m pytest tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q` - 61 passed.
- `python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-Seeded-E2E-Detail-Budget.md && python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Seeded-E2E-Detail-Budget.md` - passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` - passed; 122 matching tests enrolled.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed; 0 Atlas runtime import findings.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - 2575 passed, 7 skipped.
- `git diff --check` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 93 |
| Seeded e2e runner | 8 |
| Tests | 55 |
| Runbook | 5 |
| **Total** | **161** |
