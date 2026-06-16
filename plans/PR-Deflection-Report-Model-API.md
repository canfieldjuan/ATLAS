# PR-Deflection-Report-Model-API

## Why this slice exists

Epic #1588 is moving the paid deflection report from one monolithic Markdown
artifact to a structured `deflection.v1` model that can feed web, PDF, email,
Markdown, and export surfaces. #1600 made the persisted model intentional at
the store boundary with `DeflectionReportAccessRecord.report_model()`, but the
hosted result page and future renderers still have no narrow API contract for
that model. They must fetch the full paid artifact, including Markdown and the
raw FAQ result, just to reach the structured section data.

The root cause is now at the API boundary: the model exists and is persisted,
but it is only incidental JSON inside `/artifact`. This slice exposes the
model as its own paid read route so downstream surfaces can consume the
structured contract without parsing Markdown or depending on full-artifact
shape.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Add a paid `GET /deflection-reports/{request_id}/report-model` route that
   returns the safe persisted `deflection.v1` projection from
   `DeflectionReportAccessRecord.report_model()`.
2. Keep the existing public snapshot and paid full-artifact routes unchanged.
3. Prove the new route is tenant-scoped, locked until the Stripe webhook marks
   the report paid, returns the persisted model after unlock, and returns 404
   for legacy/malformed artifacts with no supported model.
4. Update the frontend report contract docs to name the narrow model route.
5. Archive the now-merged #1600 plan doc and refresh `plans/INDEX.md`.

### Review Contract

- Acceptance criteria:
  - [ ] The new route is registered as
        `GET /deflection-reports/{request_id}/report-model`.
  - [ ] It uses the same account-scoped store lookup as `/artifact`.
  - [ ] It returns 403 while the report is unpaid.
  - [ ] It returns the persisted `deflection.v1` model after payment unlock.
  - [ ] It returns 404 when the report exists but the stored artifact has no
        supported report model, so consumers fail closed on historical or
        drifted rows.
  - [ ] Public snapshot payloads and the existing full artifact route remain
        unchanged.
- Affected surfaces: extracted package control-surface API, Atlas route wiring
  tests, frontend contract docs, plan archive/index.
- Risk areas: paid-artifact leakage, tenant/account scope, historical artifact
  compatibility, route dependency classification.
- Reviewer rules triggered: R1, R2, R5, R10, R14.

### Files touched

- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Report-Model-API.md`
- `plans/archive/PR-Deflection-Report-Model-Persistence.md`
- `tests/test_atlas_billing_content_ops_deflection_paid_flow.py`
- `tests/test_atlas_content_ops_generated_assets_api.py`
- `tests/test_content_ops_faq_report_contract_docs.py`

## Mechanism

The control-surface router already has the three pieces this route needs:

- tenant/account scope via `_resolve_scope(...)` and
  `_required_scope_account_id(...)`;
- paid-state enforcement in `/artifact`;
- the #1600 safe projection helper exposed as
  `DeflectionReportAccessRecord.report_model()`.

This PR adds a sibling route to `/artifact`:

```python
@router.get("/deflection-reports/{request_id}/report-model", ...)
async def deflection_report_model(...):
    record = await store.get_artifact_record(account_id=..., request_id=...)
    if record is None: raise 404
    if not record.paid: raise 403
    model = record.report_model()
    if model is None: raise 404
    return model
```

The route returns only the validated model projection. It does not return
Markdown, `faq_result`, source IDs outside the model's own paid section data,
or fallback-converted legacy shapes.

## Intentional

- No new storage/migration: #1600 already persists the model in artifact JSONB.
- No renderer migration in this slice. This is the API contract future
  portfolio/PDF/email renderers consume; moving each renderer is a follow-up
  and should stay reviewable on its own.
- A stored paid report with no supported model returns 404 from this route
  rather than falling back to Markdown parsing. That keeps the structured API
  honest while `/artifact` remains available for legacy full-report access.

## Deferred

- Move the hosted paid result page to prefer this `report-model` route.
- Move the curated PDF/email renderers to consume model sections instead of
  capping/parsing Markdown.

Parked hardening: none.

## Verification

- Focused route/docs pytest:
  python -m pytest tests/test_atlas_billing_content_ops_deflection_paid_flow.py::test_deflection_paid_flow_locks_snapshot_until_stripe_webhook_unlocks tests/test_atlas_billing_content_ops_deflection_paid_flow.py::test_deflection_report_model_route_fails_closed_without_supported_model tests/test_atlas_content_ops_generated_assets_api.py::test_content_ops_public_deflection_routes_use_rate_limit_gate tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_report_contract_links_example -q
  -- 4 passed.
- Touched test files:
  python -m pytest tests/test_atlas_billing_content_ops_deflection_paid_flow.py tests/test_atlas_content_ops_generated_assets_api.py tests/test_content_ops_faq_report_contract_docs.py -q
  -- 30 passed.
- Compile check:
  python -m compileall extracted_content_pipeline/api/control_surfaces.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py tests/test_atlas_content_ops_generated_assets_api.py tests/test_content_ops_faq_report_contract_docs.py
  -- passed.
- Full extracted CI check: bash scripts/run_extracted_pipeline_checks.sh --
  4371 passed, 10 skipped.
- Plan sync check:
  python scripts/sync_pr_plan.py plans/PR-Deflection-Report-Model-API.md --check
  -- passed.
- Local PR review bundle: bash scripts/local_pr_review.sh -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_report_contract.md` | 6 |
| `extracted_content_pipeline/api/control_surfaces.py` | 25 |
| `plans/INDEX.md` | 1 |
| `plans/PR-Deflection-Report-Model-API.md` | 139 |
| `plans/archive/PR-Deflection-Report-Model-Persistence.md` | 0 |
| `tests/test_atlas_billing_content_ops_deflection_paid_flow.py` | 50 |
| `tests/test_atlas_content_ops_generated_assets_api.py` | 1 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 2 |
| **Total** | **224** |
