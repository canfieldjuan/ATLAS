# PR-Deflection-CSV-Admission-Observed-Evidence

## Why this slice exists

#1577 proved clean Zendesk-shaped CSV projections admit at full coverage, then
explicitly deferred low-coverage policy until observed partial provider exports
exist. Its final review also called out the next implementation constraint:
the runner currently marks non-full-coverage cases as `observed`, but the
summary still treats any non-`ok` case as blocking. That is fail-safe, but it
prevents the next evidence slice from recording real partial-coverage exports
without turning the proof red.

This slice makes observed evidence a first-class, non-gating case type while
keeping expected-full-coverage failures blocking. It does not set a threshold.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser-admission
Slice phase: Vertical slice

1. Add optional observed CSV inputs to the CSV admission evidence runner.
2. Include observed cases in the generated summary/doc without making them
   fail the run.
3. Keep expected-full-coverage cases fail-closed when they reject, skip rows,
   or emit coverage warnings.
4. Keep the default committed Zendesk artifact clean: observed case count is
   zero until an operator supplies observed partial exports.

### Review Contract

- Acceptance criteria:
  - [ ] The runner accepts optional observed CSV evidence files and inspects
        them through the same source-row CSV diagnostics path.
  - [ ] Observed cases with partial coverage are recorded with raw/usable
        counts, usable ratio, admission status, and coverage warnings.
  - [ ] Observed cases do not add blocking violation codes or make the CLI
        return non-zero.
  - [ ] Expected-full-coverage cases still fail closed on partial coverage.
  - [ ] The default committed artifact remains policy-neutral and does not
        add a low-coverage reject threshold.
- Affected surfaces: scripts, validation docs, validation fixtures, tests.
- Risk areas: false-green evaluator behavior, threshold overclaiming, CI
  enrollment.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `docs/extraction/validation/deflection_csv_admission_threshold_evidence_2026-06-15.md`
- `docs/extraction/validation/fixtures/deflection_csv_admission_threshold_evidence_20260615/summary.json`
- `plans/PR-Deflection-CSV-Admission-Observed-Evidence.md`
- `scripts/evaluate_csv_admission_threshold_evidence.py`
- `tests/test_evaluate_csv_admission_threshold_evidence.py`

## Mechanism

The runner keeps the two built-in Zendesk product-proof projections as
expected-full-coverage cases. It also accepts repeatable observed CSV inputs
from the CLI using name/path pairs. Each observed file is inspected through the
same `inspect_ingestion_file` source-row CSV path and appears in the same
`cases` array.

Case status stays explicit:

- `ok`: expected-full-coverage case admitted every row with no partial warning.
- `failed`: expected-full-coverage case rejected, skipped rows, or warned.
- `observed`: operator-supplied evidence case. It records what happened, but
  does not block the run.

The generated summary separates `blocking_violation_codes` from
`observed_case_count` so the next threshold-policy slice can look at observed
partial ratios without confusing evidence collection with enforcement.

## Intentional

- This does not commit a synthetic partial-coverage artifact as product
  evidence. Tests use synthetic partial CSVs to prove behavior; the committed
  default artifact still reflects the clean Zendesk corpus only.
- Observed cases are non-gating by design. A later threshold-policy PR can
  promote an observed ratio into a reject rule after the operator has real
  provider-export evidence.
- No source CSV content is committed. Observed CSV files are read locally and
  summarized as counts, statuses, warning codes, and field names.
- Local cross-layer caller hints are generic script symbol name collisions
  (`main`, `_parse_args`, `_proof_doc`), not real non-diff callers of this
  standalone evaluator. The touched behavior is covered through the enrolled
  evaluator tests.

## Deferred

- Product-backed low-coverage policy: choose warn-only vs reject after observed
  partial provider exports are collected.
- #1458 streaming upload memory remains separate.

Parked hardening: none.

## Verification

- Pending before push:
  - Command: /home/juan-canfield/Desktop/Atlas/.venv/bin/python scripts/evaluate_csv_admission_threshold_evidence.py --json
    - Passed; regenerated the committed summary/doc artifact with observed case count zero.
  - Command: /home/juan-canfield/Desktop/Atlas/.venv/bin/python -m pytest tests/test_evaluate_csv_admission_threshold_evidence.py tests/test_extracted_content_ingestion_diagnostics.py
    - Passed: 21 passed.
  - Command: bash scripts/run_extracted_pipeline_checks.sh
    - Passed: 4287 passed, 10 skipped.
  - Local PR review wrapper with the planned PR body file.
    - Passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/deflection_csv_admission_threshold_evidence_2026-06-15.md` | 16 |
| `docs/extraction/validation/fixtures/deflection_csv_admission_threshold_evidence_20260615/summary.json` | 5 |
| `plans/PR-Deflection-CSV-Admission-Observed-Evidence.md` | 117 |
| `scripts/evaluate_csv_admission_threshold_evidence.py` | 111 |
| `tests/test_evaluate_csv_admission_threshold_evidence.py` | 80 |
| **Total** | **329** |
