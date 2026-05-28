# PR-FAQ-SaaS-Demo-Cleanup-Summary

## Why this slice exists

The SaaS demo one-command FAQ route smoke now preserves child artifacts, but the
top-level cleanup summary only carries generic identifiers. The cleanup child
artifact already reports the account, deleted FAQ rowcount, and raw delete
status; hiding those fields from the compact e2e result makes a hosted demo run
harder to audit after cleanup.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Functional validation.

1. Preserve cleanup-specific proof fields in the compact child artifact shown
   in the one-command SaaS demo e2e result.
2. Pin the wrapper test so the result JSON exposes the cleanup account, deleted
   FAQ rowcount, and raw delete status.
3. Update the runbook result checks to tell operators where to inspect that
   cleanup proof.

### Files touched

- `plans/PR-FAQ-SaaS-Demo-Cleanup-Summary.md`
- `scripts/smoke_content_ops_faq_saas_demo_route_e2e.py`
- `tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py`
- `docs/extraction/validation/content_ops_faq_saas_demo_route_case_runbook.md`

## Mechanism

`_compact_artifact(...)` already copies a small allowlist of stable child
result keys into the top-level e2e summary. This slice extends that allowlist
with cleanup proof fields that are already emitted by
`seed_content_ops_faq_saas_demo.py` cleanup mode:

```python
for key in (..., "account_id", "deleted_faq_ids", "delete_status"):
    if key in payload:
        summary[key] = payload[key]
```

The wrapper still avoids duplicating large child artifacts; it only exposes the
small fields needed to prove cleanup ran against the expected account and row.

## Intentional

- No seeder cleanup behavior changes. Rowcount parsing and fail-closed cleanup
  already live in the seeder path.
- No live hosted route behavior changes. This is result visibility for the
  existing validation wrapper.
- No broad compact-artifact dump is added; the e2e summary stays bounded and
  deterministic.

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
- `python scripts/audit_plan_doc.py plans/PR-FAQ-SaaS-Demo-Cleanup-Summary.md`
  - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-FAQ-SaaS-Demo-Cleanup-Summary.md`
  - passed.
- `bash scripts/check_ascii_python.sh`
  - passed.
- `ATLAS_CURRENT_PR_BODY_FILE=/home/juan-canfield/Desktop/atlas-pr-bodies/faq-saas-demo-cleanup-summary.md bash scripts/local_pr_review.sh`
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 86 |
| E2E wrapper | 12 |
| Tests | 14 |
| Runbook | 3 |
| **Total** | **115** |
