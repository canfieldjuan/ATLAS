# PR-Deflection-See-Something-Invariant

## Why this slice exists

#1467's remaining parser-admission question was whether low non-zero coverage
should become a hard reject. The operator decided the product rule: **keep
ACCEPT + warning** for partial coverage, because showing a warning and still
producing the snapshot/report is preferable when at least some ticket text is
usable.

Root cause: the accept/reject boundary is behavior rather than a durable
contract. The backend rejects zero-usable CSV uploads and warns on partial
coverage, but a future threshold slice could change that accidentally. The
zero-usable reject also carries only a machine reason code, so the one genuine
"nothing usable" path does not tell a customer how to fix the upload.

This PR fixes the root in safe scope by testing the boundary and adding
customer-facing guidance to the existing zero-usable reject. It does not add a
low-coverage reject threshold.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser
Slice phase: Production hardening

1. Add `message` and `how_to_fix` to the existing zero-usable
   `source_row_admission.admission_decision` reject.
2. Carry those fields through atlas-intel-ui and render them on operator
   ingestion diagnostics.
3. Add regressions proving partial coverage still accepts with a warning, while
   zero-usable and parse-error paths give guidance without raw row echoes.

### Review Contract

Acceptance criteria:
- Partial source-row CSV coverage remains `ACCEPT` and still returns
  `ok: true` with at least one opportunity plus a
  `partial_source_row_coverage` warning.
- Zero usable source-row CSV coverage remains the only admission-level
  `REJECT` in this slice and includes non-empty `message` and `how_to_fix`.
- Reject guidance does not echo raw row content or private/internal sentinels.
- atlas-intel-ui preserves and renders the admission reject guidance.
- Parse-error guidance from #1675 remains intact and covered.

Affected surfaces:
- Extracted content pipeline ingestion diagnostics.
- atlas-intel-ui operator content-ops upload/import diagnostics.

Risk areas:
- Accidentally changing the product boundary from ACCEPT+warning to hard reject
  for low but non-zero coverage.
- Echoing raw ticket/private text in guidance.
- Frontend silently dropping new diagnostic fields.

Reviewer rules triggered: R1, R2, R9, R10, R12, R13, R14.

### Files touched

- `atlas-intel-ui/scripts/content-ops-upload-source-run-handoff.test.mjs`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/domain/contentOps/fromWire.ts`
- `atlas-intel-ui/src/domain/contentOps/types.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `extracted_content_pipeline/campaign_source_adapters.py`
- `plans/PR-Deflection-See-Something-Invariant.md`
- `tests/test_extracted_content_ingestion_diagnostics.py`

## Mechanism

`SourceRowAdmissionDiagnostics.admission_decision` already centralizes the
source-row CSV admission rule. The condition stays unchanged:

- no decision for non-CSV/header-only inputs;
- `REJECT` only when a non-empty CSV has zero usable source rows;
- `ACCEPT` for any CSV with one or more usable source rows;
- `coverage_warnings` reports partial coverage without blocking the report.

The zero-usable reject gains static `message` and `how_to_fix` fields so it can
guide the operator/customer without including row values.

atlas-intel-ui types, maps, and renders the admission guidance beside the
existing parse-error notice.

## Intentional

- Shape-preserving: this PR intentionally does not add a low non-zero coverage
  reject threshold. The operator chose ACCEPT + warning for partial coverage.
- Guidance is static and does not include raw CSV values, source ids, private
  notes, or snippets from rejected rows.
- Buyer-facing atlas-portfolio rendering is not touched in this Atlas PR.

## Deferred

- #1582 remains optional evidence gathering for provider calibration, but no
  longer blocks the current ACCEPT + warning policy.
- Buyer-facing atlas-portfolio rendering remains a separate repo/lane if needed.

Parked hardening: none.

## Verification

- pytest tests/test_extracted_content_ingestion_diagnostics.py -q - 30 passed.
- npm --prefix atlas-intel-ui run test:content-ops-upload-source-run-handoff - 7 passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py - OK, 185 matching tests enrolled.
- python -m py_compile extracted_content_pipeline/campaign_source_adapters.py tests/test_extracted_content_ingestion_diagnostics.py - passed.
- bash scripts/run_extracted_pipeline_checks.sh - reasoning core 295 passed; extracted content pipeline 4638 passed, 10 skipped, 1 existing torch warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-intel-ui/scripts/content-ops-upload-source-run-handoff.test.mjs` | 64 |
| `atlas-intel-ui/src/api/contentOps.ts` | 22 |
| `atlas-intel-ui/src/domain/contentOps/fromWire.ts` | 26 |
| `atlas-intel-ui/src/domain/contentOps/types.ts` | 22 |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | 32 |
| `extracted_content_pipeline/campaign_source_adapters.py` | 7 |
| `plans/PR-Deflection-See-Something-Invariant.md` | 120 |
| `tests/test_extracted_content_ingestion_diagnostics.py` | 106 |
| **Total** | **399** |
