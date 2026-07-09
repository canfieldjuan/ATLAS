# PR-Resolution-Audit-S7-Date-Window

Ownership lane: resolution-audit-csv

## Why this slice exists

S7 of the #1993 remediation arc, launch blocker #6 (C3 + M7, both in the
audit's top criticals and both verified failing on `origin/main` before
coding). C3: `parse_support_ticket_source_date` is per-value and US-only, so
a day-first (UK/EU/AU) upload silently transposes `02-01-2026` into Feb 1
when the day is <= 12 and goes unparseable when it is > 12 -- corrupting the
recency/annualization basis of the paid number without any signal. M7: the
date window gate is all-or-nothing (`missing_count == 0`), so ONE blank or
unparseable export cell flips the whole report onto the dateless x12
run-rate basis (verified: 19/20 dated rows -> window disabled). The
annualized-field EXPOSURE half of S7 is an operator product-shape decision
and is deliberately not here.

### Problem-derived contract

- Root cause: no upload-level date-convention inference exists (the parser
  sees one value at a time, defaulting to US month-first), and window
  validity is a boolean on perfect coverage rather than a coverage
  threshold.
- Correct fix must touch/change:
  1. `extracted_content_pipeline/support_ticket_dates.py`:
     `infer_support_ticket_date_convention(values)` -- an upload proves
     day-first (any first field > 12), month-first (any second field > 12),
     contradicts itself (`ambiguous`), or is undecidable (`unknown`).
     `parse_support_ticket_source_date` gains a `convention` parameter:
     day-first formats when proven; the historical US month-first default
     when proven or unknown; and NUMERIC dates refuse to parse under a
     contradictory convention -- a silently transposed date is worse than a
     missing one. ISO/date/datetime inputs parse regardless.
  2. `extracted_content_pipeline/support_ticket_input_package.py`: infer the
     convention once per upload from admitted rows' `created_at`; NORMALIZE
     parseable `created_at` values to ISO at admission (the convention
     decision is made exactly once; downstream markdown/recency re-parsers
     can never transpose); window validity becomes a coverage threshold
     (`DATE_WINDOW_MIN_COVERAGE` = 0.9 of included rows dated); a
     `support_ticket_date_convention_ambiguous` warning when the upload
     contradicts itself; diagnostics carry `date_convention`.
  3. Tests: inference matrix (day-first/month-first/contradictory/unknown),
     transposition fixed end-to-end (day-first upload normalizes `02/01` to
     Jan 2, never Feb 1), ambiguous warns and does not guess, one-blank-date
     keeps the window, sparse dates do not fake a window, the 90% edge both
     sides, dateless uploads keep the dateless basis, diagnostics counts;
     CI enrollment in the same PR.
- Must not change:
  - Report/user-facing wording; the annualized/dateless run-rate EXPOSURE in
    hosted/PDF/email payloads (operator-gated, tracked on #1993).
  - `window_days` semantics (caller-declared window length).
  - Privacy (S6A), hygiene (S6B), junk (S6E), evidence/status (S6D/M9),
    clustering (S5) logic.
  - Final-output scrubber grammar; product shape.

Review-loop guard: HARD cap of 3 Codex rounds counted by ROUNDS (the
operator-backed rule after #2053/#2054); at cap, fix what is written,
resolve/waive remaining threads, merge on required-green.

## Scope (this PR)

Slice phase: Vertical slice

Max files: 7

1. `extracted_content_pipeline/support_ticket_dates.py` -- inference +
   convention-aware parsing.
2. `extracted_content_pipeline/support_ticket_input_package.py` -- one-time
   inference, ISO normalization at admission, coverage-threshold window,
   diagnostics/warnings.
3. `tests/test_support_ticket_dates_window.py` -- the contract tests.
4. `tests/test_extracted_support_ticket_input_package.py` -- three pins
   updated from raw `created_at` passthrough to the ISO canonical value
   (same dates; the normalization is the intended contract change).
5. `scripts/run_extracted_pipeline_checks.sh` -- CI enrollment, same PR.
6. `tests/maturity_sweep/baseline_extracted_content_pipeline.json` --
   ratchet acceptance (the dates module grew with the inference machinery;
   behavior pinned by 15 dedicated both-direction tests).
7. This plan doc.

### Files touched

- `extracted_content_pipeline/support_ticket_dates.py`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Resolution-Audit-S7-Date-Window.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/maturity_sweep/baseline_extracted_content_pipeline.json`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_support_ticket_dates_window.py`

### Review Contract

Acceptance criteria:
1. A day-first upload (proven by any day > 12) parses `02/01/2026` as
   Jan 2 end-to-end; the transposed Feb 1 value appears nowhere.
2. A contradictory upload warns (`support_ticket_date_convention_ambiguous`)
   and leaves numeric dates unparsed rather than guessing.
3. One blank date in twenty keeps the dated window; 8/10 dated does not;
   the 90% edge holds on both sides; dateless uploads keep the dateless
   basis.
4. `created_at` on admitted rows is canonical ISO whenever parseable.
5. US-default behavior for undecidable uploads is unchanged (regression
   pins).

Reachability proof: wired on the live `build_support_ticket_input_package`
path; the end-to-end tests assert package output (window, source_period,
normalized rows, warnings), not just the parser.

Affected surfaces: date diagnostics, window validity, `source_period` basis
selection, row `created_at` canonicalization feeding markdown recency.

Risk areas: threshold semantics (money-basis selection -- pinned both sides
at the edge); ISO canonicalization changing raw passthrough expectations
(three pins updated deliberately).

Reviewer rules triggered: R1, R2, R10, R14 (extracted package change; test evidence; gate predicate change; admission-adjacent parser probed both directions).

## Mechanism

`infer_support_ticket_date_convention` scans numeric two-field dates across
the upload and returns the proven convention, `ambiguous` on contradiction,
or `unknown`. The package builder infers once, rewrites parseable
`created_at` to ISO on admitted rows, and computes diagnostics from the
canonical values. `_date_window_is_valid` replaces the `missing_count == 0`
rule with dated/included >= `DATE_WINDOW_MIN_COVERAGE`.

## Intentional

- Refuse-to-guess on contradictory uploads: unparsed dates are visible in
  diagnostics; transposed dates corrupt the money basis silently.
- ISO canonicalization at admission: the convention decision happens once,
  upstream, instead of in every downstream re-parser.
- 0.9 coverage threshold: one export artifact cannot flip the money basis,
  while majority-undated uploads still fall back to the dateless x12 basis.
  The threshold is a named constant with both edges pinned.

## Deferred

- Annualized/dateless run-rate exposure in hosted/PDF/email (operator
  decision, #1993).
- S8a/S8b runtime scorecard + money reconciliation (#2051/#2052).

## Verification

- `python -m pytest tests/test_support_ticket_dates_window.py -q` (15 passed)
- Adjacent suites: input-package + smoke + junk + hygiene + privacy
  (512 passed)
- Full CI mirror scripts/run_extracted_pipeline_checks.sh (5909 passed, 21
  skipped)
- scripts/validate_extracted_content_pipeline.sh (mapped files match);
  maturity ratchet green after documented baseline acceptance.
- Both defects reproduced on `origin/main` before coding (transposition and
  the 19/20 window kill).

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_dates.py` | 98 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 57 |
| `plans/PR-Resolution-Audit-S7-Date-Window.md` | 167 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/maturity_sweep/baseline_extracted_content_pipeline.json` | 3 |
| `tests/test_extracted_support_ticket_input_package.py` | 6 |
| `tests/test_support_ticket_dates_window.py` | 144 |
| **Total** | **476** |
