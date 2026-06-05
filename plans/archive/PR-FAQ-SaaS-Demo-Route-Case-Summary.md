# PR-FAQ-SaaS-Demo-Route-Case-Summary

## Why this slice exists

The SaaS demo one-command FAQ route smoke writes a child seed artifact that
proves the route-case file was produced, but the top-level e2e summary drops
that proof during compaction. Operators can see the route case path under
`artifacts.route_case_file`, but not whether the seed step successfully wrote
the case file and how many cases it emitted without opening the child seed JSON.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Functional validation.

1. Preserve the seed child artifact's `route_case_file` proof in the compact
   one-command SaaS demo e2e result.
2. Pin the wrapper test so the top-level result exposes route-case write status,
   path, and case count.
3. Update the SaaS demo runbook result checks to point operators at the compact
   route-case proof.

### Files touched

- `plans/PR-FAQ-SaaS-Demo-Route-Case-Summary.md`
- `scripts/smoke_content_ops_faq_saas_demo_route_e2e.py`
- `tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py`
- `docs/extraction/validation/content_ops_faq_saas_demo_route_case_runbook.md`

## Mechanism

`_compact_artifact(...)` already copies bounded child-result sections such as
`search`, `requests`, `detail`, and `budgets`. This slice adds a small
`route_case_file` mapping to that same compaction path:

```python
summary["route_case_file"] = {
    "ok": route_case_file.get("ok"),
    "path": route_case_file.get("path"),
    "cases": route_case_file.get("cases"),
    "error": route_case_file.get("error"),
}
```

The wrapper continues to preserve the child artifact path separately under
`artifacts.route_case_file`; this addition exposes the seed step's own write
status and case count.

## Intentional

- No route-case file format changes.
- No seeder behavior changes; the seeder already emits this proof.
- No broad child-artifact dump is added; only the stable, small
  `route_case_file` mapping is surfaced.
- No live hosted route behavior changes.

## Deferred

- Parked hardening: none.
- Broader live-host threshold tuning remains deferred until repeated hosted
  route runs produce real latency baselines.

## Verification

- `python -m py_compile scripts/smoke_content_ops_faq_saas_demo_route_e2e.py tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py`
  - passed.
- `python -m pytest tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py -q`
  - 6 passed.
- `git diff --check`
  - passed.
- `python scripts/audit_plan_doc.py plans/PR-FAQ-SaaS-Demo-Route-Case-Summary.md`
  - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-FAQ-SaaS-Demo-Route-Case-Summary.md`
  - passed.
- `bash scripts/check_ascii_python.sh`
  - passed.
- `ATLAS_CURRENT_PR_BODY_FILE=/home/juan-canfield/Desktop/atlas-pr-bodies/faq-saas-demo-route-case-summary.md bash scripts/local_pr_review.sh`
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 88 |
| E2E wrapper | 8 |
| Tests | 11 |
| Runbook | 2 |
| **Total** | **109** |
