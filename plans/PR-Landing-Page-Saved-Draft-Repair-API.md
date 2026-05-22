# PR-Landing-Page-Saved-Draft-Repair-API

## Why this slice exists

The saved-draft landing-page repair service now exists, but nothing in the
generated asset review API can call it yet. Operators need a backend route
that repairs the draft already in the review queue instead of starting a new
generation run.

This slice wires the service into the generated-assets router while keeping the
UI button deferred. The route stays landing-page-only, tenant-scoped, and uses
the same review row shape the existing drawer already consumes.

## Scope (this PR)

Ownership lane: content-ops/landing-page-saved-draft-repair-api

1. Add optional LLM and skill providers to the generated-assets router.
2. Add `POST /content-assets/landing_page/drafts/{id}/repair`.
3. Fetch the tenant-scoped draft, reject approved drafts, call the repair
   service, refetch the draft, and return the review row plus repair result.
4. Add API tests for success, missing LLM wiring, approved draft blocking, and
   non-landing-page rejection.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Saved-Draft-Repair-API.md` | Plan doc for this API slice. |
| `extracted_content_pipeline/api/generated_assets.py` | Add repair providers and landing-page repair route. |
| `atlas_brain/api/__init__.py` | Wire host LLM and skill providers into generated-assets routes. |
| `tests/test_extracted_content_asset_api.py` | Add repair route tests and local LLM/skills fakes. |

## Mechanism

The repair route resolves the database pool and tenant scope exactly like the
existing edit route. It loads the draft through `PostgresLandingPageRepository`,
rejects missing or approved drafts, resolves host-provided LLM and skill ports,
and calls `LandingPageGenerationService.repair_draft(...)`.

Successful repairs update the same row through the service. The route refetches
that row and returns `landing_page_draft_export_row(...)` with a
`repair_result` object attached so the UI can update the drawer without learning
a second response shape.

## Intentional

- No UI button in this slice.
- No non-landing-page repair route.
- No auto-repair on edit or list. Repair remains an explicit operator action.
- No fallback LLM creation inside the router. The host must wire the LLM and
  skill providers so production behavior is explicit.

## Deferred

- `PR-Landing-Page-Saved-Draft-Repair-UI` should add the review-drawer repair
  action after this route lands.
- A later audit slice can add richer repair metrics/events if needed.

## Verification

- Python compile for `extracted_content_pipeline/api/generated_assets.py` and
  `atlas_brain/api/__init__.py` -> passed.
- Focused pytest for `tests/test_extracted_content_asset_api.py` -> 49 passed.
- `scripts/validate_extracted_content_pipeline.sh` -> passed.
- `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` for
  `extracted_content_pipeline` -> passed.
- `scripts/audit_extracted_standalone.py` for `extracted_content_pipeline` ->
  passed with 0 findings.
- `scripts/check_ascii_python.sh` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~70 |
| API | ~75 |
| Tests | ~130 |
| Total | ~275 |
