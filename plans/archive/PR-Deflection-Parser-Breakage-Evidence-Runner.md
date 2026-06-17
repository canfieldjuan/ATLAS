# PR-Deflection-Parser-Breakage-Evidence-Runner

## Why this slice exists

#1467 defines the parser-admission boundary, and the operator asked to "break
it" so we know where parser mechanics fail, why, and which failures can be
fixed without waiting for a real provider CSV. The root cause is not a missing
threshold; it is that the existing evidence runner proves clean Zendesk-shaped
CSV acceptance but does not score adversarial parser mechanics as fail-closed,
warned, no-policy, or known fail-open. This slice fixes that root for synthetic
mechanics by adding a repeatable breakage matrix to the existing runner while
leaving product threshold policy unchanged.

This slice may exceed the 400 LOC soft cap because it commits the generated
proof JSON alongside the runner/test changes; the generated artifact is the
reviewable evidence for the matrix.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser-testing
Slice phase: Robust testing

1. Extend `scripts/evaluate_csv_admission_threshold_evidence.py` with a
   synthetic parser-breakage matrix over the real source-row CSV diagnostics
   path.
2. Record zero-usable, private-only, status/timestamp-only, partial-coverage,
   header-only, and JSON-blob-message cases in the committed summary/doc.
3. Make expected guard failures blocking, while recording known fail-open gaps
   as non-blocking evidence.
4. Archive the merged #1624 plan as same-lane housekeeping.

### Review Contract

Acceptance criteria:
- Each breakage case runs through `inspect_ingestion_file(..., source_rows=True,
  source_format="csv")`, not a mocked parser.
- Cases that should fail closed assert the exact `REJECT` reason/location or
  coverage warning shape.
- The JSON-blob-message case is recorded as a known fail-open/gap, not hidden
  and not used to set threshold policy.
- The summary JSON and proof doc clearly separate clean product-shaped CSV
  threshold evidence from synthetic parser-breakage mechanics.
- #1467 low non-zero threshold policy remains deferred until real partial
  provider CSV evidence exists.

Affected surfaces:
- CSV admission evidence runner and tests.
- Committed offline validation artifact/doc.
- Plans archive/index housekeeping for #1624.

Risk areas:
- Accidentally making synthetic gaps block CI before product policy exists.
- Blurring "mechanics proved" with "threshold justified."
- Letting known fail-open evidence disappear from the artifact.

Reviewer rules triggered: R1, R2, R10, R12, R13, R14.

### Files touched

- `HARDENING.md`
- `docs/extraction/validation/deflection_csv_admission_threshold_evidence_2026-06-15.md`
- `docs/extraction/validation/fixtures/deflection_csv_admission_threshold_evidence_20260615/summary.json`
- `plans/INDEX.md`
- `plans/PR-Deflection-Parser-Breakage-Evidence-Runner.md`
- `plans/archive/PR-Deflection-Parser-Invariant-Test-Pack.md`
- `scripts/evaluate_csv_admission_threshold_evidence.py`
- `tests/test_evaluate_csv_admission_threshold_evidence.py`

## Mechanism

- Add a small `CsvBreakageCase` table to the existing evidence script.
- Reuse `_evaluate_csv_path(...)` for every case, then classify the observed
  outcome as `reject`, `accept_with_warning`, `no_policy_decision`, or
  `accept_clean`.
- Compare that observed outcome to the expected mechanic. Expected guard
  mismatches become blocking violations; explicit `KNOWN_FAIL_OPEN` cases are
  counted and documented but do not fail the runner.
- Regenerate the committed summary JSON and Markdown proof artifact.

## Intentional

- No admission threshold changes. Synthetic ratios can prove detectors, but
  they cannot justify product policy for low non-zero usable coverage.
- No live provider calls, no customer CSVs, and no PII. The matrix is synthetic
  by design.
- The JSON-blob-message case is intentionally non-blocking: it names a current
  parser gap instead of silently converting the runner into a policy PR.

## Deferred

- #1467 low non-zero reject threshold: still blocked on real partial provider
  CSV evidence.
- A future policy/fix slice can promote the JSON-blob-message known gap into a
  fail-closed guard once the desired parser behavior is agreed; tracked in
  `HARDENING.md` as "CSV source-row admission accepts machine JSON in mapped
  message fields."

Parked hardening:
- `HARDENING.md`: "CSV source-row admission accepts machine JSON in mapped
  message fields" because this PR records the breakage evidence but does not
  choose parser policy for machine payload rejection.

## Verification

- `python scripts/evaluate_csv_admission_threshold_evidence.py --json`
  - Passed; generated a 6-case breakage matrix with 0 blocking cases and 1
    known fail-open gap.
- `pytest tests/test_evaluate_csv_admission_threshold_evidence.py -q`
  - 10 passed in 0.15s.
- `./scripts/run_extracted_pipeline_checks.sh`
  - 4563 passed, 10 skipped, 1 warning in 74.93s.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 11 |
| `docs/extraction/validation/deflection_csv_admission_threshold_evidence_2026-06-15.md` | 19 |
| `docs/extraction/validation/fixtures/deflection_csv_admission_threshold_evidence_20260615/summary.json` | 242 |
| `plans/INDEX.md` | 1 |
| `plans/PR-Deflection-Parser-Breakage-Evidence-Runner.md` | 125 |
| `plans/archive/PR-Deflection-Parser-Invariant-Test-Pack.md` | 0 |
| `scripts/evaluate_csv_admission_threshold_evidence.py` | 195 |
| `tests/test_evaluate_csv_admission_threshold_evidence.py` | 116 |
| **Total** | **709** |
