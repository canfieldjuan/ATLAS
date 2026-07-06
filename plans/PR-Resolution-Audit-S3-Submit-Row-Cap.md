# PR-Resolution-Audit-S3-Submit-Row-Cap

## Why this slice exists

Issue #1993 S3 is the current-code remediation for the deflection submit
large-input boundary. The root cause is that sync deflection submit separated
from the existing sync execute row contract: `_MAX_DEFLECTION_SUBMIT_ROWS` is
defined as the 50 MiB byte cap, missing `limit` resolves to the full parsed row
count, and the internal submit path bypasses the generic
`faq_execute_max_source_material_rows` guard before building the report inline.
CSV loaders can parse-cap rows when given a bound, but the default bound is
missing; full-thread JSON loaders do not accept a row bound at all.

Review-fix root cause: the first push applied the correct default cap on the
request path, but two adjacent paths still carried pre-S3 assumptions. Raw
multipart `limit` values were parsed after form validation and therefore could
exceed the sync cap; parser truncation diagnostics inferred cap drops from
`source_row_count - rows`, which misclassified invalid Zendesk thread entries
as truncation; and the operator smoke profile still required 30,000 submitted
rows against a sync route that now intentionally admits at most 1,000.

Diff-size note: this is over the 400 LOC soft cap because the safe fix has to
land server admission, full-thread importer admission, smoke-helper validation,
route/importer/smoke regression tests, S3 tracking, S2 plan archival, and the
review-fix regressions together. Splitting would leave one of the submit paths,
diagnostic paths, or verifier paths temporarily false-green.

Correctness contract before code:

1. Define a real sync submit row cap that is independent of byte size and
   aligned with the existing 1,000-row sync FAQ source-material contract.
2. Apply that cap at importer admission for CSV uploads, CSV blobs,
   full-thread JSON uploads, and full-thread JSON blobs, with explicit `limit`
   able to lower the cap but never exceed it.
3. Preserve honest diagnostics: source rows count the parsed upload, submitted
   rows count the admitted subset, invalid Zendesk thread entries stay as
   parser warnings, and truncation warnings fire only for rows the cap actually
   drops.
4. Keep the existing sync endpoint shape and paid-gate flow. Do not introduce a
   background job in this slice; full-volume/background submit is the upstream
   large-upload path and remains deferred once sync admit is safe.
5. Do not change user-facing report, snapshot, landing, email, or PDF shape.
   This slice only changes submit admission/diagnostics, the operator smoke
   limit check, the S3 tracker, and teardown housekeeping for the merged S2
   plan.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Functional validation

1. Replace the accidental 50 MiB-as-row-cap submit limit with an explicit
   sync row cap and make missing `limit` use that cap.
2. Thread the same row cap into CSV and full-thread JSON upload/blob loaders.
3. Update the submit smoke helper's client-side limit validation to match the
   server contract instead of accepting byte-sized row limits.
4. Update the #1993 local tracker for S2 merged/S3 in progress and archive the
   merged S2 plan.
5. Add focused tests proving default-cap truncation, explicit lower limits,
   full-thread upload/blob cap behavior, diagnostics, and smoke validation.

### Review Contract

- Affected surfaces:
  - `extracted_content_pipeline.api.control_surfaces` submit admission and
    diagnostics.
  - `extracted_content_pipeline.support_ticket_zendesk_thread` full-thread row
    normalization admission.
  - `scripts/smoke_content_ops_deflection_submit_handoff.py` operator preflight
    validation.
  - S3 tracking docs/plan housekeeping only.
- Acceptance criteria:
  - Missing submit `limit` admits at most 1,000 rows and reports
    `max_source_material_rows=1000`.
  - Explicit `limit` below 1,000 still admits that lower count and keeps the
    existing truncation warning shape.
  - Explicit `limit` above 1,000 is rejected or clamped before importer
    admission on JSON-body, blob, and multipart upload paths; malformed and
    negative raw multipart limits fail closed instead of slicing oddly.
  - Full-thread JSON upload/blob importers count total source tickets while
    processing only the capped source-entry window, without counting invalid
    Zendesk entries as cap truncation.
  - The submit smoke helper's named profile reflects the capped sync route and
    no longer requires impossible 30,000-row submit admission.
  - Report/snapshot/landing/email/PDF output schemas and copy are untouched.
- Risk areas:
  - Off-by-one truncation diagnostics.
  - Full-thread JSON source counts vs processed/admitted rows.
  - Operator full-volume smoke assumptions that previously used the bad
    byte-sized limit.
- Reviewer rules triggered: R1 requirements match, R2 test evidence, R8
  contract drift, R10 detection/guard behavior, R13 unseen/class probes, R14
  codebase verification.

### Files touched

- `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/support_ticket_zendesk_thread.py`
- `plans/INDEX.md`
- `plans/PR-Resolution-Audit-S3-Submit-Row-Cap.md`
- `plans/archive/PR-Resolution-Audit-S2-Row-Key-Cache.md`
- `scripts/smoke_content_ops_deflection_submit_handoff.py`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_smoke_content_ops_deflection_submit_handoff.py`

## Mechanism

`_MAX_DEFLECTION_SUBMIT_ROWS` becomes an explicit 1,000-row sync cap.
`_deflection_submit_parse_max_rows(None)` returns that cap. Explicit limits are
integer-coerced, clamped to the cap, and invalid values raise a 422 before
importer admission. `_deflection_submit_max_rows(None, source_row_count)` also
clamps to it, so every missing-limit path uses the same bound before
package/report build.

CSV upload/blob paths already pass `max_rows` into the parser; they inherit the
new default cap. Full-thread JSON upload/blob loaders gain the same `max_rows`
parameter and pass it through `load_zendesk_full_thread_rows_from_json_*` into
`rows_from_zendesk_full_thread`, which stops normalizing after the processed
source-entry cap while preserving `source_row_count` as the total ticket-entry
count and returning parser truncation separately from invalid-row warnings.

The submit smoke helper's `SUBMIT_ROW_LIMIT_MAX` is changed to the same 1,000
row contract so operator preflight fails before sending a now-invalid
byte-sized row limit. Its named volume-gate profile is scoped to capped sync
submit admission; the true full-volume/background proof remains deferred.

## Intentional

- This PR does not add async/background large-upload processing. Once sync
  submit is safely capped, the large-upload path is a separate vertical slice
  because it needs progress/state semantics and may touch product flow.
- The 1,000-row cap matches the existing sync execute source-material limit and
  `build_support_ticket_input_package` default. It is also below the 2,000-row
  token-set clustering preview skip, so S3 does not leave sync submit in the
  known degraded large-clustering band.
- The full-thread JSON parser still decodes the JSON artifact before row
  admission. This slice caps normalization/report-build work; streaming JSON
  decode is deferred unless profiling proves JSON parse memory is still the
  binding launch risk.

## Deferred

- Full-volume/background deflection submit job with progress/status, replacing
  the old sync full-volume proof route for uploads above the sync cap.
- Streaming or incremental full-thread JSON decode if profiling shows JSON
  parse memory remains a blocker after sync row admission is capped.

Parked hardening: none.

## Verification

- Review-fix focused selectors for parser cap, Zendesk invalid diagnostics, and
  smoke profile thresholds - 19 passed.
- Full affected submit/importer/smoke files - 247 passed.
- Python compile for changed modules/tests after review fixes - passed.
- Focused submit/importer pytest selectors - 5 passed.
- Focused full-thread normalizer pytest selectors - 3 passed.
- Focused submit-handoff smoke pytest selectors - 2 passed.
- Python compile for changed modules/tests - passed.
- Full `tests/test_extracted_content_deflection_submit.py` - 95 passed.
- Full `tests/test_extracted_support_ticket_input_package.py` - 91 passed.
- Full `tests/test_smoke_content_ops_deflection_submit_handoff.py` - 51 passed.
- Extracted manifest validation via `scripts/validate_extracted_content_pipeline.sh` - passed.
- Extracted reasoning import audit via
  `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` - passed.
- Extracted standalone audit via `scripts/audit_extracted_standalone.py` - passed.
- Extracted ASCII audit via `scripts/check_ascii_python.sh` - passed.
- Plan sync check via `scripts/sync_pr_plan.py` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md` | 15 |
| `extracted_content_pipeline/api/control_surfaces.py` | 92 |
| `extracted_content_pipeline/support_ticket_zendesk_thread.py` | 32 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Resolution-Audit-S3-Submit-Row-Cap.md` | 186 |
| `plans/archive/PR-Resolution-Audit-S2-Row-Key-Cache.md` | 0 |
| `scripts/smoke_content_ops_deflection_submit_handoff.py` | 8 |
| `tests/test_extracted_content_deflection_submit.py` | 187 |
| `tests/test_extracted_support_ticket_input_package.py` | 54 |
| `tests/test_smoke_content_ops_deflection_submit_handoff.py` | 50 |
| **Total** | **627** |
