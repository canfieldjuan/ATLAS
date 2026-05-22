# PR-Landing-Page-Saved-Draft-Repair-Service

## Why this slice exists

Landing pages now have save-time readiness gates, review UI readiness panels,
and a manual edit path. The remaining operator gap is repairing a saved draft
that is already in the review queue without starting a brand-new run.

This slice adds the service-level repair primitive first. It reuses the
existing landing-page generator, readiness checks, and quality repair prompt
loop, and updates the same saved draft row when the repair passes. The API
button is intentionally deferred until the service contract is pinned.

## Scope (this PR)

Ownership lane: content-ops/landing-page-saved-draft-repair-service

1. Add prompt guidance for saved-draft repair mode.
2. Add `LandingPageGenerationService.repair_draft(...)` for existing saved
   landing-page drafts.
3. Persist repaired content back to the same tenant-scoped draft row.
4. Persist trusted repair metadata/history through the repository update path.
5. Add focused service and repository tests.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Saved-Draft-Repair-Service.md` | Plan doc for this service slice. |
| `extracted_content_pipeline/skills/digest/landing_page_generation.md` | Add saved-draft repair prompt guidance. |
| `extracted_content_pipeline/landing_page_ports.py` | Clarify that update callers must pass trusted metadata. |
| `extracted_content_pipeline/landing_page_generation.py` | Add saved-draft repair orchestration. |
| `extracted_content_pipeline/landing_page_postgres.py` | Persist trusted metadata on draft updates. |
| `tests/test_extracted_landing_page_generation.py` | Add service repair tests. |
| `tests/test_extracted_landing_page_postgres.py` | Update repository update test for metadata persistence. |
| `tests/test_extracted_content_asset_api.py` | Keep edit API mass-assignment test aligned with trusted metadata persistence. |

## Mechanism

`repair_draft(...)` accepts a `LandingPageDraft`, evaluates its existing
readiness/quality state, and, when it fails, sends the current draft plus stable
repair issues through the existing landing-page generation prompt. The LLM must
return a full landing-page JSON object. The service then runs the same quality
and readiness checks used by generation. Passing repairs update the same row
through `LandingPageRepository.update_draft(...)`.

The repository update path now writes metadata from the trusted draft object.
The review edit API remains protected because it constructs the draft metadata
from the existing row and ignores user-supplied `metadata`.

## Intentional

- No generated-assets API route in this slice. The repair button should call a
  pinned service contract, not define service behavior inside the router.
- Approved landing pages are not repaired by this service.
- Repair updates the same draft id instead of creating a sibling draft.
- The LLM receives the current draft as context, but quality/readiness checks
  still decide whether the repair can persist.

## Deferred

- `PR-Landing-Page-Saved-Draft-Repair-API` should expose the service through an
  authenticated generated-assets route.
- `PR-Landing-Page-Saved-Draft-Repair-UI` should add the review-drawer button
  after the API route lands.
- `PR-Landing-Page-Edit-Audit-Trail` can add richer user/editor audit metadata.

## Verification

- Python compile for `extracted_content_pipeline/landing_page_generation.py`,
  `extracted_content_pipeline/landing_page_postgres.py`, and
  `extracted_content_pipeline/landing_page_ports.py` -> passed.
- Focused pytest for `tests/test_extracted_landing_page_generation.py`,
  `tests/test_extracted_landing_page_postgres.py`, and
  `tests/test_extracted_content_asset_api.py` -> 93 passed.
- `scripts/validate_extracted_content_pipeline.sh` -> passed.
- `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` for
  `extracted_content_pipeline` -> passed.
- `scripts/audit_extracted_standalone.py` for `extracted_content_pipeline` ->
  passed with 0 findings.
- `scripts/check_ascii_python.sh` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~90 |
| Prompt | ~15 |
| Service/repository | ~190 |
| Tests | ~180 |
| Total | ~475 |

This exceeds the 400 LOC target because the service method, trusted metadata
persistence, and tests need to land together for a usable repair primitive.
