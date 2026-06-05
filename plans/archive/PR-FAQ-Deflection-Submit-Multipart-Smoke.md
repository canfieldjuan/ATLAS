# PR-FAQ-Deflection-Submit-Multipart-Smoke

## Why this slice exists

#1184 moved the production-safe portfolio submit contract from JSON `blob_url`
fetching to authenticated server-to-server multipart CSV upload. The hosted
handoff smoke still only exercises the legacy JSON shape, so operators cannot
prove the deployed production path without hand-rolling a multipart call.

This slice updates the existing hosted smoke and runbook to make a local CSV
file the preferred submit input while keeping the JSON `blob_url` path as a
compatibility fallback.

The diff is slightly over the 400 LOC target because the smoke is a checker
surface. The multipart encoder, input validation, mode-specific diagnostics,
diagnostics redaction, legacy fallback, and negative fixtures need to ship
together so the hosted proof cannot false-green the wrong submit mode.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Functional validation

1. Add `--csv-file` / `ATLAS_DEFLECTION_SUBMIT_CSV_FILE` support to
   `scripts/smoke_content_ops_deflection_submit_handoff.py`.
2. Send multipart form data when a CSV file is provided, including
   `csv_file`, `support_platform`, `company_name`, `contact_email`, and
   `limit`.
3. Keep the existing JSON `blob_url` mode for legacy hosted proof and rollback.
4. Extend smoke validation/tests so the multipart path fails closed on missing
   or unsafe file inputs, sends no JSON `blob_url`, redacts secrets, and
   requires multipart `uploaded_bytes` diagnostics.
5. Update the submit handoff runbook to document multipart as the preferred
   production smoke path.

### Files touched

- `plans/PR-FAQ-Deflection-Submit-Multipart-Smoke.md`
- `scripts/smoke_content_ops_deflection_submit_handoff.py`
- `tests/test_smoke_content_ops_deflection_submit_handoff.py`
- `docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md`

## Mechanism

The smoke chooses submit mode from inputs:

```text
--csv-file present -> multipart submit
otherwise          -> legacy JSON blob_url submit
```

Multipart mode reads the local CSV bytes, builds a bounded multipart request
with a generated boundary, posts it with the existing bearer token, and then
reuses the current execute-envelope, snapshot, and unpaid-artifact validation.
The execute-envelope validator requires `uploaded_bytes` in multipart mode and
`blob_bytes` in legacy JSON mode so an incomplete diagnostics envelope cannot
false-green the hosted proof.
The result artifact records only non-sensitive mode/size metadata. It does not
write the bearer token, signed blob query string, or CSV contents.

## Intentional

- The smoke still rejects localhost API hosts because this remains a hosted
  handoff proof, not a local route test.
- `blob_url` support remains because #1184 kept that backend contract for
  compatibility. The runbook will steer production validation to `--csv-file`.
- This slice does not add a live artifact. The operator still supplies deployed
  ATLAS auth plus a real support-ticket CSV when running the smoke.
- The script hand-builds multipart with the Python standard library so the
  extracted CI environment does not need `requests`.

## Deferred

- Parked hardening: none. `HARDENING.md` has no active entries touching this
  smoke or submit route lane.
- Removing the legacy JSON `blob_url` smoke mode remains deferred until the
  portfolio production path no longer needs rollback coverage.

## Verification

- `python -m py_compile scripts/smoke_content_ops_deflection_submit_handoff.py tests/test_smoke_content_ops_deflection_submit_handoff.py`
- `python -m pytest tests/test_smoke_content_ops_deflection_submit_handoff.py -q`
- `bash scripts/run_extracted_pipeline_checks.sh`
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-submit-multipart-smoke.md`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 97 |
| Smoke script | 195 |
| Tests | 212 |
| Runbook | 49 |
| **Total** | **553** |

This is over the soft cap for the checker-surface reason named in **Why this
slice exists**.
