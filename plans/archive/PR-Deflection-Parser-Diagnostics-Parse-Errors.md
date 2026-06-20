# PR-Deflection-Parser-Diagnostics-Parse-Errors

## Why this slice exists

#1463 still lists an ingestion diagnostics hardening gap: the inspect path
crashes on a bad file instead of reporting it as a structured finding. I
reproduced that on current `origin/main`: source-row CSV missing-header and
inconsistent-column inputs raise `CsvCustomerDataParseError`, while invalid
opportunity JSON raises `JSONDecodeError`.

Root cause: `inspect_ingestion_file(...)` delegates directly to the lower-level
loaders and only builds `IngestionDiagnosticsReport` after parsing succeeds.
The lower-level loaders are allowed to reject malformed files, but the
diagnostics boundary is the user/operator-facing inspection surface; it should
turn parser-level failures into a structured failed report with reason,
location, and safe how-to-fix text.

This PR fixes the root at the diagnostics boundary and carries that structured
failure through the Content Ops operator UI. It does not hide parser errors in
the loaders or turn bad files into accepted uploads; it makes the inspect
API/CLI/UI fail closed with a report instead of an exception traceback or an
invisible backend-only field.

The synced diff is over the 400 LOC soft cap because the review exposed two
inseparable halves of the same boundary defect: backend parse failures must be
converted safely, and the UI must not drop the converted field. Splitting would
leave either an escaping parser branch or an invisible diagnostic in place.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser-testing
Slice phase: Production hardening

1. Add an optional top-level `parse_error` payload to diagnostics reports.
2. Convert parser-level file errors from `inspect_ingestion_file(...)` into a
   failed diagnostics report instead of raising.
3. Cover source-row CSV parse failures, decode failures, raw `csv.Error`, and
   invalid JSON parse failures through the inspect path.
4. Thread `parse_error` through atlas-intel-ui wire/domain mapping and render it
   in the Content Ops inspect/import result panels.
5. Preserve successful inspect behavior, including existing source-row
   admission diagnostics.

### Review Contract

Acceptance criteria:
- Missing-header and inconsistent-column source-row CSVs return `ok: false`
  plus `parse_error`; they do not raise.
- Invalid JSON, undecodable input, and raw CSV parser failures return
  `ok: false` plus `parse_error` without raw file content.
- CLI `--json` emits `parse_error` and exits non-zero without a traceback.
- atlas-intel-ui preserves `parse_error` across wire/domain mapping and renders
  the parser issue instead of dropping the backend diagnostic.
- Successful inspect output remains unchanged except no `parse_error`.
- Lower-level loaders keep raising their existing structured errors; this PR
  only changes the inspection/reporting boundary.

Affected surfaces: `ingestion_diagnostics.py`, the existing inspect CLI
serialization, atlas-intel-ui ingestion diagnostics mapping/rendering, and
diagnostics tests.

Risk areas: over-catching programmer errors, leaking raw file content,
marking parse failures as `ok`, or regressing parse-success CSV admission.

Reviewer rules triggered: R1, R2, R9, R10, R12, R13, R14.

### Files touched

- `atlas-intel-ui/scripts/content-ops-upload-source-run-handoff.test.mjs`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/domain/contentOps/fromWire.ts`
- `atlas-intel-ui/src/domain/contentOps/types.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `extracted_content_pipeline/ingestion_diagnostics.py`
- `plans/PR-Deflection-Parser-Diagnostics-Parse-Errors.md`
- `tests/test_extracted_content_ingestion_diagnostics.py`

## Mechanism

Add a `parse_error` field to `IngestionDiagnosticsReport`; serialize it only
when present, and force `ok` false when it exists. Wrap the parser-loading
section of `inspect_ingestion_file(...)` in a narrow conversion layer:

- `CsvCustomerDataParseError` -> use its existing safe `as_dict()` shape and
  add `location` based on mode/format.
- `json.JSONDecodeError` -> emit a bounded `json_parse_error` with line/column
  and generic UTF-8/JSON how-to-fix text, without raw payload text.
- `UnicodeDecodeError` -> emit `file_decode_error` with encoding + byte offset,
  without echoing the raw bytes.
- `csv.Error` -> emit `csv_parse_error` with the parser's bounded message for
  raw CSV parser failures such as field-size-limit breaches.

The CLI already serializes `report.as_dict()` and returns `1` when
`report.ok` is false, so no separate CLI exception handler is needed.
The atlas-intel-ui adapter adds the snake_case wire field, the domain mapper
camelCases it, and the new-run page renders the parse error beside the existing
warning/admission diagnostics.

## Intentional

- This PR does not make malformed files importable. Parse failures remain
  `ok: false`; the change is traceback -> structured finding.
- This PR catches only parser-level file errors known to come from the loader
  boundary. It does not catch arbitrary `Exception`, so programmer bugs still
  fail loudly.
- JSON parse diagnostics include parser reason + line/column only, not raw
  source text.
- The UI change stays on the operator Content Ops ingestion panels; no buyer
  surface or full-report QA surface is touched.

## Deferred

- #1467 low non-zero usable-ratio reject threshold and #1458 streaming upload
  memory hardening remain separate.

Parked hardening: none.

## Verification

- `/home/juan-canfield/Desktop/Atlas/.venv/bin/python -m pytest tests/test_extracted_content_ingestion_diagnostics.py -q` -- 28 passed.
- `npm --prefix atlas-intel-ui ci` -- installed local UI dependencies for this worktree.
- `npm --prefix atlas-intel-ui run test:content-ops-upload-source-run-handoff` -- 5 passed.
- `scripts/run_extracted_pipeline_checks.sh` via bash -- 4619 passed, 10 skipped, 1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-intel-ui/scripts/content-ops-upload-source-run-handoff.test.mjs` | 46 |
| `atlas-intel-ui/src/api/contentOps.ts` | 13 |
| `atlas-intel-ui/src/domain/contentOps/fromWire.ts` | 13 |
| `atlas-intel-ui/src/domain/contentOps/types.ts` | 13 |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | 25 |
| `extracted_content_pipeline/ingestion_diagnostics.py` | 254 |
| `plans/PR-Deflection-Parser-Diagnostics-Parse-Errors.md` | 137 |
| `tests/test_extracted_content_ingestion_diagnostics.py` | 209 |
| **Total** | **710** |
