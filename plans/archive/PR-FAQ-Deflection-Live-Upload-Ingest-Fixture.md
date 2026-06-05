# PR-FAQ-Deflection-Live-Upload-Ingest-Fixture

## Why this slice exists

PR-FAQ-Deflection-Live-Upload-Fixture added a synthetic CSV for operator live
upload testing. The reviewer noted that the fixture test proved the CSV's
shape, but not that the real support-ticket ingest builder accepts the rows
without warnings or skips. This slice closes that validation gap before the
fixture becomes the default live-upload test artifact. A PR review also caught
that the original test path was not enrolled in the extracted pipeline runner,
so the slice now puts the test under an enrolled content-ops filename family and adds it
to the explicit CI runner list.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Functional validation

1. Extend the live-upload fixture test to call the extracted package's real
   support-ticket ingest builder.
2. Assert the package metadata reports all fixture rows included, no skipped or
   truncated rows, and no warnings.
3. Move the test under an enrolled content-ops test filename family and
   add it to `scripts/run_extracted_pipeline_checks.sh`.
4. Keep the slice to test/runner coverage only; the fixture and production
   ingest code remain unchanged.

### Files touched

- `plans/PR-FAQ-Deflection-Live-Upload-Ingest-Fixture.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_faq_deflection_live_upload_fixture.py`

## Mechanism

The test loads the committed CSV fixture through the existing CSV row helper,
then passes those dictionaries to
`extracted_content_pipeline.support_ticket_input_package.build_support_ticket_input_package`.
The assertions lock the compatibility contract that matters for live upload:
the real ingest builder sees 12 rows, includes all 12, produces no warnings,
and keeps the normalized `source_material` count aligned with the CSV.
The test file name now matches the existing content-ops workflow filter, and
the extracted pipeline runner invokes that exact file.

## Intentional

- This imports the flat extracted package module rather than any
  `atlas_brain.services.*` module, so the test remains compatible with the
  extracted-checks environment that lacks torch and asyncpg.
- The fixture is not changed in this slice because the purpose is to prove the
  already-merged sample is ingest-compatible.
- The old `tests/test_faq_deflection_live_upload_fixture.py` path is renamed
  instead of adding a one-off workflow glob; this keeps fixture validation in
  the existing content-ops test family.

## Deferred

- Live hosted upload execution remains a manual runbook step because it needs
  deployed credentials and operator-provided environment values.

Parked hardening: none.

## Verification

- Python compile check for
  `tests/test_content_ops_faq_deflection_live_upload_fixture.py`
  - passed.
- Focused pytest for
  `tests/test_content_ops_faq_deflection_live_upload_fixture.py`
  - 3 passed.
- Extracted pipeline CI enrollment audit - passed, 140 matching tests enrolled.
- Local PR review with the prepared PR body file - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 83 |
| Extracted runner enrollment | 1 |
| Fixture test | 16 |
| **Total** | **100** |

Under the 400 LOC soft cap.
