# PR-FAQ-Deflection-Submit-Fixture-Default

## Why this slice exists

The checked live-upload CSV fixture now exists and is validated against the real
support-ticket ingest path, but the hosted submit smoke still requires the
operator to remember and export `ATLAS_DEFLECTION_SUBMIT_CSV_FILE` before the
snapshot proof can run. That keeps the new fixture one step away from the
default validation path.

This slice makes the hosted submit smoke use the checked fixture by default
when neither an explicit CSV file nor a legacy blob URL is provided.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Product polish

1. Add a single default fixture path constant in the hosted submit smoke.
2. Default `--csv-file` to that fixture when
   `ATLAS_DEFLECTION_SUBMIT_CSV_FILE` and `ATLAS_DEFLECTION_SUBMIT_BLOB_URL`
   are both omitted.
3. Update the runbook so operators know the checked fixture is now the default
   smoke input.
4. Add focused tests for parser defaults and the explicit blob override.

### Files touched

- `plans/PR-FAQ-Deflection-Submit-Fixture-Default.md`
- `docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md`
- `scripts/smoke_content_ops_deflection_submit_handoff.py`
- `tests/test_smoke_content_ops_deflection_submit_handoff.py`

## Mechanism

The smoke defines `DEFAULT_CSV_FILE` as the checked fixture under
`docs/extraction/validation/fixtures/faq_deflection_live_upload_sample.csv`.
The parser resolves CSV input in this order:

1. `ATLAS_DEFLECTION_SUBMIT_CSV_FILE`
2. `--csv-file`
3. the checked fixture, only when no legacy blob URL is provided

The blob URL remains an explicit rollback path. When
`ATLAS_DEFLECTION_SUBMIT_BLOB_URL` or `--blob-url` is set and no CSV is set,
the parser leaves `csv_file` unset so the smoke still exercises the legacy JSON
submit path.

## Intentional

- This does not change the hosted submit route, portfolio upload page, or
  generated report behavior. It only changes the operator smoke default.
- The legacy blob fallback stays available and wins over the default fixture
  when explicitly provided.
- Existing environment access in this script is left as-is; this slice does not
  broaden secret/config handling beyond the current smoke harness.

## Deferred

- Live hosted execution remains operator-driven because deployed API URL, auth,
  account id, company name, and contact email are runtime values.

Parked hardening: none.

## Verification

- Python compile check for
  `scripts/smoke_content_ops_deflection_submit_handoff.py` and
  `tests/test_smoke_content_ops_deflection_submit_handoff.py` - passed.
- Focused pytest for `tests/test_smoke_content_ops_deflection_submit_handoff.py`
  - 22 passed.
- Local PR review with the prepared PR body file - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 84 |
| Smoke script | 21 |
| Tests | 27 |
| Runbook | 13 |
| **Total** | **145** |

Under the 400 LOC soft cap.
