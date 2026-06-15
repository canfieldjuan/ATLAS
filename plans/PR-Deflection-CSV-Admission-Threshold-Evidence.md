# PR-Deflection-CSV-Admission-Threshold-Evidence

## Why this slice exists

#1467 now has the admission boundary pieces for source-row CSV uploads: #1575
rejects non-empty CSVs with zero usable rows, and #1576 warns when accepted
CSVs skip some rows. The remaining question is not another diagnostic shape;
it is whether real product-shaped uploads justify a low non-zero coverage reject
threshold. The reviewer note on #1576 called that sequencing out directly:
before adding more policy, run the surface on a real/product-shaped export and
record threshold evidence.

This slice is over the 400 LOC target because it adds the evidence runner, its
CI-enrolled tests, and the committed proof artifact together. Splitting those
would leave either untested evidence tooling or an unrepeatable artifact.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser-admission
Slice phase: Functional validation

1. Add a deterministic CSV admission evidence runner for the committed Zendesk
   product-proof corpus.
2. Commit the generated threshold-evidence artifact showing how the current
   source-row CSV admission diagnostics behave on product-shaped Zendesk CSV
   projections.
3. Keep policy unchanged: no low non-zero usable-row reject threshold is added
   in this PR.

### Review Contract

- Acceptance criteria:
  - [ ] The runner converts the committed Zendesk product-proof corpus into
        source-row CSV projections and inspects them through the real ingestion
        diagnostics path.
  - [ ] The committed artifact records raw row count, usable row count, usable
        ratio, admission status, coverage warnings, mapped fields, ignored
        private fields, and unmapped populated fields for each projection.
  - [ ] Clean product-shaped Zendesk projections remain accepted with no partial
        coverage warning.
  - [ ] A focused synthetic partial-coverage fixture proves the evidence runner
        records the warning branch instead of hiding it.
  - [ ] The PR does not add or change a hard low-coverage reject threshold.
- Affected surfaces: scripts, validation docs, validation fixtures, tests.
- Risk areas: false confidence from evidence artifacts, CI enrollment,
  parser-admission scope drift.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `docs/extraction/validation/deflection_csv_admission_threshold_evidence_2026-06-15.md`
- `docs/extraction/validation/fixtures/deflection_csv_admission_threshold_evidence_20260615/summary.json`
- `plans/PR-Deflection-CSV-Admission-Threshold-Evidence.md`
- `scripts/evaluate_csv_admission_threshold_evidence.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_evaluate_csv_admission_threshold_evidence.py`

## Mechanism

The runner loads
`docs/extraction/validation/fixtures/zendesk_product_proof_corpus.json`,
projects each ticket into two CSV shapes that match common Zendesk exports, and
passes each temporary CSV through
`inspect_ingestion_file` with `source_rows=True` and `source_format=csv`.

The product-shaped projections are:

1. public thread CSV: ticket id, subject, public comments, and internal notes;
2. description CSV: ticket id, subject, description, and internal notes.

The summary records the serialized `source_row_admission` block for each case
plus a short interpretation. A clean run is evidence that current Zendesk-shaped
CSV exports are accepted at full coverage; it is not evidence that a low
non-zero reject threshold is safe. The tests therefore include a small
partial-coverage CSV fixture that exercises the warning branch without treating
that synthetic ratio as production threshold data.

## Intentional

- No hard threshold is added. The only product-shaped corpus available in-repo
  is clean enough to produce full coverage, so using it to choose a low-coverage
  cutoff would be false precision.
- The partial-coverage fixture is test evidence for the runner, not threshold
  evidence. The proof document names that boundary so the next policy slice does
  not mistake a synthetic ratio for an observed customer-export distribution.
- The runner writes a compact artifact rather than storing generated CSV files;
  the generated rows are reproducible from the committed corpus.

## Deferred

- Product-backed low-coverage policy: gather observed partial-coverage uploads
  from real provider exports before choosing a warn-only vs reject threshold.
- #1458 streaming upload memory remains separate.

Parked hardening: none.

## Verification

- Command: .venv/bin/python scripts/evaluate_csv_admission_threshold_evidence.py --json
  - Passed; generated the committed summary/doc artifact and reported both
    Zendesk-shaped CSV projections at 50/50 usable rows.
- Command: .venv/bin/python -m pytest tests/test_evaluate_csv_admission_threshold_evidence.py tests/test_extracted_content_ingestion_diagnostics.py
  - Passed: 19 passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh
  - Passed: 4285 passed, 10 skipped.
- Local PR review wrapper with the planned PR body file.
  - Passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/deflection_csv_admission_threshold_evidence_2026-06-15.md` | 41 |
| `docs/extraction/validation/fixtures/deflection_csv_admission_threshold_evidence_20260615/summary.json` | 80 |
| `plans/PR-Deflection-CSV-Admission-Threshold-Evidence.md` | 121 |
| `scripts/evaluate_csv_admission_threshold_evidence.py` | 347 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_evaluate_csv_admission_threshold_evidence.py` | 181 |
| **Total** | **771** |
