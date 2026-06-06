# PR-Content-Ops-Support-Ticket-Input-Package

## Why this slice exists

PR-Content-Ops-Input-Provider-Contract and PR-Content-Ops-Input-Provider-API-Wiring created the generic handoff point between host ingestion and Content Ops generation. The next missing piece is a concrete, production-shaped adapter for uploaded/imported support tickets so the handoff can carry real ticket rows into the existing FAQ, landing-page, and blog planning paths without reimplementing any generator.

This slice stays in the Content Ops input-provider lane. It does not implement FAQ generation, article publishing, file upload, DB persistence, or hosted route behavior.

This is slightly over the normal soft diff budget because the smallest useful slice needs the adapter and the end-to-end control-surface tests in the same PR. Splitting them would leave either untested adapter code or tests for code that does not exist yet.

## Scope (this PR)

Ownership lane: content-ops/input-provider-ticket-package

1. Add a support-ticket input-package builder that accepts already-loaded ticket source material.
2. Normalize support-ticket rows into `source_material` that existing `faq_markdown` and signal paths already understand.
3. Populate landing-page SEO/GEO/AEO and blog topic inputs from the same ticket package using down-to-earth FAQ Report defaults.
4. Preserve explicit operator/request overrides through the existing `merge_content_ops_input_package` contract.
5. Add focused tests proving the package reaches `request_from_mapping`, `preview_control_surface`, and `build_generation_plan`.
6. Enroll the new test file in extracted pipeline CI.

### Files touched

- `extracted_content_pipeline/support_ticket_input_package.py`
- `tests/test_extracted_support_ticket_input_package.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Content-Ops-Support-Ticket-Input-Package.md`

## Mechanism

`build_support_ticket_input_package(...)` expands source material through the existing source-row bundle reader, keeps customer wording from ticket subject/body/comment fields, and returns a `ContentOpsInputPackage` with:

- `source_material`, `faq_window_days`, `faq_source_types`, and `faq_title` for the existing ticket FAQ service.
- `topic` and `filters` for the existing blog path.
- `campaign_name`, `offer`, `audience`, `target_keyword`, `faq_questions`, `source_period`, `cta_label`, and `cta_url` for the existing landing-page context path.

The adapter only prepares inputs. Existing control-surface validation and generation planning remain the source of truth.

## Intentional

- No new route or host wiring in this slice. The API already accepts an optional input provider; hosts can call this builder from their provider implementation.
- No FAQ generator changes. FAQ implementation and standalone FAQ-article ownership stay with the separate FAQ session.
- No file parsing. This builder accepts already-loaded rows or bundles; file upload/import stays with the ingestion lane.
- Defaults are opinionated around the FAQ Report offer, but callers can override them through either builder arguments or the existing request merge path.

## Deferred

- Future PR: host ticket-upload provider wiring can call this builder from the mounted Content Ops route once the owning ingestion session is ready.
- Future PR: FAQ standalone article contracts can consume the same `source_material` package without this slice implementing FAQ article generation.
- Future PR: richer ticket clustering or volume ranking can replace the current light question extraction after the FAQ owner defines that contract.
- Parked hardening: none.

## Verification

- `py_compile` for `extracted_content_pipeline/support_ticket_input_package.py`
  and `tests/test_extracted_support_ticket_input_package.py` - passed.
- `pytest` for `tests/test_extracted_support_ticket_input_package.py` and
  `tests/test_extracted_content_ops_input_provider.py` - 16 passed.
- `scripts/validate_extracted_content_pipeline.sh` - passed.
- `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` for
  `extracted_content_pipeline` - passed.
- `scripts/audit_extracted_standalone.py` with `--fail-on-debt` - passed.
- `scripts/check_ascii_python.sh` - passed.
- `scripts/run_extracted_pipeline_checks.sh` - 1896 passed, 1 skipped.
- `scripts/local_pr_review.sh` with `--allow-dirty` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~80 |
| Adapter module | ~335 |
| Tests | ~300 |
| CI enrollment | ~5 |
| **Total** | **~720** |

This is over the 400 LOC soft budget because the adapter and its tests are the smallest real end-to-end slice.
