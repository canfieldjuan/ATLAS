# PR-Content-Ops-Hosted-Asset-Workflow

## Why this slice exists

The host install runbook now tells customers to mount the generated asset
router beside the existing campaign operations and B2B campaign review routers.
The standalone router has focused tests, but the hosted workflow regression
still only mounts the campaign routers. We should lock the combined host shape
before adding more surfaces.

## Scope (this PR)

1. Mount `create_generated_asset_router` in the hosted workflow test fixture.
2. Add one regression proving generated asset list/review routes share the host
   auth, tenant scope, and database pool providers.

### Files touched

- `tests/test_extracted_campaign_api_hosted_workflow.py`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Hosted-Asset-Workflow.md`

## Intentional

- Test-only product contract coverage.
- No runtime implementation changes.
- No docs changes beyond the plan and coordination row.

## Deferred

- Seller + generated asset combined workflow coverage.
- Frontend/UI tests that call the generated asset router.

## Verification

Planned:

- `python -m pytest tests/test_extracted_campaign_api_hosted_workflow.py tests/test_extracted_content_asset_api.py`
- `git diff --check`

## Estimated diff size

- Tests/plan/coordination: ~90 LOC.
