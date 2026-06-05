# PR-FAQ-SaaS-Demo-Artifact-Status-Gate

## Why this slice exists

The SaaS demo one-command FAQ route smoke runbook says top-level `ok` includes
artifact-read status, but the wrapper currently derives `ok` only from child
process return status plus seed FAQ-id extraction. A child command can exit 0
while its JSON result artifact is missing, malformed, or reports `ok: false`;
that would leave the top-level e2e result looking successful even though the
proof artifact is not trustworthy.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Functional validation.

1. Make expected child result artifacts load-bearing in the SaaS demo e2e
   wrapper.
2. Add compact top-level diagnostics when a child result artifact is missing,
   malformed, or reports `ok: false`.
3. Add a focused negative fixture proving a route command that exits 0 but
   omits its result artifact fails the wrapper.

### Files touched

- `plans/PR-FAQ-SaaS-Demo-Artifact-Status-Gate.md`
- `scripts/smoke_content_ops_faq_saas_demo_route_e2e.py`
- `tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py`

## Mechanism

`_compact_artifact(...)` already normalizes artifact read failures into:

```python
{"available": False, "ok": False, "path": "...", "errors": [...]}
```

This slice adds a small `_artifact_status_errors(...)` helper that checks the
compacted seed, route, and cleanup artifacts that were expected to exist. The
top-level `summary["ok"]` includes `not artifact_errors`, and the diagnostics
are appended to `summary["errors"]`.

Skipped artifacts remain skipped: route is not checked when seed failed, and
cleanup is not checked when `--keep-data` or missing seed FAQ id intentionally
skips cleanup.

## Intentional

- No child command behavior changes.
- No artifact schema expansion beyond top-level diagnostic strings.
- No retry or recovery is added; this wrapper is a validation gate, so missing
  proof should fail closed.
- No live hosted route behavior changes.

## Deferred

- Parked hardening: none.
- Broader live-host threshold tuning remains deferred until repeated hosted
  route runs produce real latency baselines.

## Verification

- `python -m py_compile scripts/smoke_content_ops_faq_saas_demo_route_e2e.py tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py`
  - passed.
- `python -m pytest tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py -q`
  - 8 passed.
- `git diff --check`
  - passed.
- `python scripts/audit_plan_doc.py plans/PR-FAQ-SaaS-Demo-Artifact-Status-Gate.md`
  - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-FAQ-SaaS-Demo-Artifact-Status-Gate.md`
  - passed.
- `bash scripts/check_ascii_python.sh`
  - passed.
- `ATLAS_CURRENT_PR_BODY_FILE=/home/juan-canfield/Desktop/atlas-pr-bodies/faq-saas-demo-artifact-status-gate.md bash scripts/local_pr_review.sh`
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 85 |
| E2E wrapper | 29 |
| Tests | 70 |
| **Total** | **184** |
