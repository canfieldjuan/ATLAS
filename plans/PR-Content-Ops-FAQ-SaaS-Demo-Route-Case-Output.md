# PR-Content-Ops-FAQ-SaaS-Demo-Route-Case-Output

## Why this slice exists

The SaaS FAQ demo seeder can generate the checked B2B SaaS FAQ, save it, approve
it, and verify DB-backed search. The hosted route concurrency smoke can validate
deployed search/detail behavior from a case file. The handoff between those two
steps is still manual: operators must copy the seeded FAQ id, corpus id, query,
status, and detail expectations into a route case by hand.

This slice adds the thinnest end-to-end handoff between the existing seeder and
the existing hosted route smoke. It does not call the hosted API itself; it emits
the route-case artifact the existing smoke consumes and wires that smoke to
enforce the emitted detail expectations when `--require-detail` is enabled.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Vertical slice

1. Add an optional `--route-case-file-output` flag to the SaaS FAQ demo seeder.
2. Reject route-case output in cleanup mode, where no seeded FAQ id is produced.
3. Write a deterministic route case containing the seeded query, corpus/status,
   expected first FAQ/account/corpus ids, and expected detail fields.
4. Teach the hosted route concurrency smoke to load and enforce the emitted
   detail expectation fields during detail hydration.
5. Include route-case output metadata in the seeder result payload.
6. Add focused tests for validation, payload shape, successful case-file
   emission.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Route-Case-Output.md` | Plan contract for this route-case handoff slice. |
| `scripts/seed_content_ops_faq_saas_demo.py` | Emit optional hosted-route case files from the seeded SaaS FAQ id. |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | Cover validation and route-case artifact generation. |
| `scripts/smoke_content_ops_faq_search_route_concurrency.py` | Consume expected detail fields from route case files. |
| `tests/test_smoke_content_ops_faq_search_route_concurrency.py` | Prove expected detail fields fail closed when the hydrated detail drifts. |

## Mechanism

The seeder parser gains:

```bash
--route-case-file-output /tmp/saas-demo-route-cases.json
```

When seeding succeeds, `seed_saas_demo_faq(...)` builds one route case from the
actual saved FAQ id and the configured corpus/query/status. The case uses the
same fields consumed by `smoke_content_ops_faq_search_route_concurrency.py`:
`expected_first_account_id`, `expected_first_corpus_id`,
`expected_first_faq_id`, and the detail expectation fields. The file is written
as sorted, indented JSON for deterministic diffs and operator inspection.

If the file write fails, the seeder still returns the seed/search payload with
the FAQ id, but marks the overall result failed and records the route-case error.

The route concurrency smoke now loads the emitted `expected_detail_*` keys into
each case and maps them to the detail envelope fields already supported by
`check_content_ops_faq_search_route_contract._validate_detail(...)`. With
`--require-detail`, a hydrated detail response whose account, target, title, or
status no longer matches the seeded FAQ fails the route smoke.

## Intentional

- No hosted route call is added to the seeder. The existing route concurrency
  smoke owns HTTP requests, auth, case-level budgets, and optional detail
  hydration.
- No cleanup behavior changes. Cleanup mode deletes an explicit FAQ id and does
  not have a new seeded route case to emit.
- The route case contains one known hit case only; miss-case coverage remains in
  the generic seeded e2e runner.

## Deferred

- Parked hardening: none. `HARDENING.md` was scanned; no active FAQ-search item
  is required for this handoff.
- A future runbook slice can document the two-command SaaS demo route validation
  flow after this artifact exists.

## Verification

- `python -m py_compile scripts/seed_content_ops_faq_saas_demo.py scripts/smoke_content_ops_faq_search_route_concurrency.py tests/test_content_ops_faq_saas_demo_corpus.py tests/test_smoke_content_ops_faq_search_route_concurrency.py` - passed.
- `python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py tests/test_smoke_content_ops_faq_search_route_concurrency.py -q` - 92 passed.
- `python scripts/audit_plan_doc.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Route-Case-Output.md && python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Route-Case-Output.md` - passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` - passed; 122 matching tests enrolled.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed; 0 Atlas runtime import findings.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - 2580 passed, 7 skipped.
- `git diff --check` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 103 |
| Seeder script | 54 |
| Route concurrency smoke | 43 |
| Tests | 167 |
| **Total** | **367** |
