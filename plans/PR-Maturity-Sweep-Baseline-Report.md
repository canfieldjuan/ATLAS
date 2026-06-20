# PR-Maturity-Sweep-Baseline-Report

## Why this slice exists

Issue #1689 deferred "baseline debt burndown/trend reporting across lanes"
after the maturity-sweep ratchet finished rolling out. Phase C4 completed the
last explicit enrollment lane (`scripts/**`); the repo now has many committed
baselines but no quick way to see which lanes carry the most debt or which
finding classes dominate.

Root cause: maturity-sweep baselines are stored as per-lane JSON artifacts, but
there is no deterministic report that summarizes total score, file count, and
finding counts across those baselines. This change treats observability, not
gate behavior: it adds a read-only report over existing baselines so operators
can prioritize burndown without reverse-engineering JSON files.

## Scope (this PR)

Ownership lane: ci/maturity-sweep
Slice phase: Production hardening

1. Add a read-only baseline summary script for maturity-sweep baseline JSON
   files.
2. Add focused unit tests for deterministic aggregation, sorting, and malformed
   baseline rejection.
3. Keep CI gate behavior unchanged.

### Review Contract

Acceptance criteria:
- The helper reads maturity-sweep baseline JSON files and summarizes per-lane
  file count, total score, top score, and finding counts.
- Output is deterministic and useful in text by default, with JSON output
  available for later automation.
- Malformed baseline JSON fails closed with a non-zero exit.
- Tests cover aggregation, ordering, JSON output, and malformed input.

Affected surfaces:
- `scripts/summarize_maturity_sweep_baselines.py`
- `tests/test_summarize_maturity_sweep_baselines.py`

Risk areas:
- Misleading totals from malformed baseline entries.
- Report nondeterminism that makes trend diffs noisy.

Reviewer rules triggered: R1, R2, R10, R14.

### Files touched

- `plans/PR-Maturity-Sweep-Baseline-Report.md`
- `scripts/summarize_maturity_sweep_baselines.py`
- `tests/test_summarize_maturity_sweep_baselines.py`

## Mechanism

The script scans a baseline glob, validates each baseline object, and aggregates
only numeric `score` and `counts` fields. Lanes are sorted by total score
descending, then lane name ascending for stable output. Text output is compact
for humans; `--json` emits the same computed summaries for future trend tooling.

## Intentional

- No workflow enrollment in this slice. This is an operator/reporting helper,
  not a new gate.
- No historical trend storage yet. This produces the current snapshot that a
  later scheduled artifact or trend slice can consume.

## Deferred

- Scheduled trend artifact publishing remains deferred.
- Threshold tightening remains deferred until operators use the report to pick
  lanes for burndown.

Parked hardening: none.

## Verification

- Command passed: python -m py_compile for
  `scripts/summarize_maturity_sweep_baselines.py` and
  `tests/test_summarize_maturity_sweep_baselines.py`.
- Command passed: pytest for `tests/test_summarize_maturity_sweep_baselines.py`
  and `tests/test_maturity_sweep.py` with `--noconftest -q`; 19 passed.
- Command passed: scripts-lane maturity-sweep ratchet for
  `tests/maturity_sweep/baseline_scripts.json`; 262 files scanned and no new
  brittleness above baseline.
- Command passed: summary report for `tests/maturity_sweep/baseline_scripts.json`
  with top three finding counts; reports `scripts 260 2385 63`.
- Command passed: full summary report over all maturity-sweep baselines; reports
  `scripts`, `extracted_content_pipeline`, and `atlas_brain_autonomous` as the
  top three total-score lanes.
- Command passed: plan sync for
  `plans/PR-Maturity-Sweep-Baseline-Report.md`.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Maturity-Sweep-Baseline-Report.md` | 101 |
| `scripts/summarize_maturity_sweep_baselines.py` | 129 |
| `tests/test_summarize_maturity_sweep_baselines.py` | 106 |
| **Total** | **336** |
