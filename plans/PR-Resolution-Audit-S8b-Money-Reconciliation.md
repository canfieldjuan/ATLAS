# PR-Resolution-Audit-S8b-Money-Reconciliation

Ownership lane: resolution-audit-csv

## Why this slice exists

S8b of the #1993 remediation arc (issue #2052, launch blocker #7 second
half; P5-7 in the audit). No model-level money reconciliation exists:
the headline `estimated_support_cost`, the per-row money in
`ranked_questions.data.rows`, and the annualized figures are computed
independently at build time and never cross-checked afterward, so
sum-of-rendered-rows vs headline can silently drift (with x16
annualized amplification). S4/#2031 shipped DISPLAY reconciliation
only. Separately, the S4-deferred delta-money path still rounds with
float semantics: `deflection_delta.py` uses `round(x, 2)` (half-even)
and the delta email's `_whole_dollar_money` uses raw f-string float
formatting (half-even), while canonical money in `deflection_money.py`
is Decimal ROUND_HALF_UP.

### Problem-derived contract

- Root cause: money figures are computed once and trusted forever --
  no verifier ties headline, rows, and annualized figures back to the
  billed-repeat predicate; and the delta path never adopted the
  canonical Decimal rounding.
- Correct fix must touch/change:
  1. `extracted_content_pipeline/faq_deflection_report.py` (owned):
     extract the per-item billed-repeat test into ONE named predicate
     (`_is_billed_repeat_item`, the existing `_ticket_count(item) >= 2`
     rule) and make `_repeat_ticket_count` use it; add
     totals-reconciliation assertions to
     `build_deflection_full_report_qa_scorecard` that VERIFY (never
     compute): headline `repeat_ticket_count` equals the predicate
     applied to the model's own `ranked_questions.data.rows`; headline
     `estimated_support_cost` equals
     `support_cost_usd(repeat_ticket_count)`; the billed rows' money
     sums exactly to the headline; each row's money equals
     `support_cost_usd(row.ticket_count)`; the annualized figure equals
     `annualized_support_cost_usd(repeat, window_days)` on the dated
     basis or `support_cost_usd(repeat * 12)` on the dateless run-rate
     basis. Because S8a's gate runs this scorecard at persist (raw and
     projected), the guard is live at the runtime boundary with no new
     wiring.
  2. `extracted_content_pipeline/deflection_money.py` (owned): a signed
     quantizer (`signed_support_cost_delta_usd`) using the SAME Decimal
     ROUND_HALF_UP rule, sign-preserving (no zero-clamp). (A signed
     FORMATTER was drafted and removed at cold-diff review: no consumer
     exists -- the delta email keeps its own signed whole-dollar shape.)
  3. `extracted_content_pipeline/deflection_delta.py` (owned): replace
     both float `round(x, 2)` call sites with the signed quantizer.
  4. `atlas_brain/content_ops_deflection_delivery.py`:
     `_whole_dollar_money` quantizes via Decimal ROUND_HALF_UP instead
     of f-string float rounding; display shape (signed whole dollars)
     unchanged.
  5. Tests: drifted fixtures proving the guard trips at the scorecard
     AND through the S8a persist gate (inflated headline count,
     tampered headline cost, tampered row money, wrong annualized
     basis); healthy fixture still passes raw + projected; delta
     rounding boundary cases (.005 cents, .5 whole dollars) proving
     half-up on both signs.
- Must not change:
  - Money COMPUTATION and display on the report path (S4 owns display;
    the guard verifies, it does not compute).
  - The billed-repeat rule itself (>=2 with source-ids fallback) --
    it moves into a named function, semantics identical.
  - Email/delta display shape (same strings; only the rounding rule at
    the half-cent / half-dollar boundary).
  - Scorecard schema_version; S8a gate mechanics.

Empirical pre-checks (run before coding): on the real fixture the
invariant holds exactly on BOTH the raw and the projected model
(headline repeat=8, cost=108.0 == billed-row sum == recompute; rows are
stored uncapped, so summing `data.rows` is sound); money is linear
(13.50 x integer is always exact at 2dp), so the equality is exact, not
tolerance-based. `_signed_email_money` currently formats via
`f"{float(value):,.0f}"` (half-even) and `deflection_delta.py` uses
float `round` (half-even), both diverging from the canonical
ROUND_HALF_UP.

Review-loop guard: HARD cap of 3 Codex rounds counted by ROUNDS; at
cap, fix what is written, resolve/waive remaining threads, merge on
required-green.

## Scope (this PR)

Slice phase: Vertical slice

Max files: 10

1. `extracted_content_pipeline/faq_deflection_report.py` -- named
   billed-repeat predicate + scorecard reconciliation assertions.
2. `extracted_content_pipeline/deflection_money.py` -- signed
   quantizer.
3. `extracted_content_pipeline/deflection_delta.py` -- canonical
   rounding at both delta call sites.
4. `atlas_brain/content_ops_deflection_delivery.py` -- half-up whole
   dollar signed display.
5. `tests/test_content_ops_deflection_report.py` -- reconciliation
   guard both directions.
6. `tests/test_atlas_content_ops_deflection_delivery.py` -- delta
   rounding boundary pins.
7-9. `tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py`,
   `tests/test_smoke_content_ops_deflection_pdf_export_validators.py`,
   `tests/test_run_deflection_full_report_qa_live_runner.py` --
   hand-rolled fixtures made money-consistent (rows without
   ticket_count/money against a nonzero headline are exactly the drift
   the guard now refuses; intended contract change, same shape as the
   S8a PII-fixture rebuild).
10. This plan doc.

### Files touched

- `atlas_brain/content_ops_deflection_delivery.py`
- `extracted_content_pipeline/deflection_delta.py`
- `extracted_content_pipeline/deflection_money.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Resolution-Audit-S8b-Money-Reconciliation.md`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_run_deflection_full_report_qa_live_runner.py`
- `tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py`
- `tests/test_smoke_content_ops_deflection_pdf_export_validators.py`

### Review Contract

Acceptance criteria:
1. A model whose headline money, row money, repeat count, or annualized
   figure drifted from the billed-repeat predicate fails the scorecard
   with a named `money.*` assertion, and therefore 502s at the S8a
   persist gate.
2. The healthy fixture passes raw and projected, exactly (no
   tolerance).
3. The billed-repeat rule exists in exactly ONE function used by both
   the count computation and the reconciliation guard.
4. Signed delta money quantizes ROUND_HALF_UP on both signs at compute
   (`deflection_delta.py`) and display (`_whole_dollar_money`); shape
   unchanged.
5. Report-path money computation and display are byte-identical.

Reachability proof: the assertions live in
`build_deflection_full_report_qa_scorecard`, which S8a's
`check_deflection_report_artifact_qa` runs at the persist boundary on
both the stored payload and the paid projection; tests drive the
drifted artifacts through the gate, not just the scorecard.

Affected surfaces: QA scorecard, persist gate verdicts, delta compute,
delta delivery email rounding.

Risk areas: false positives on legitimately-shaped models (bounded by
exact linearity of the money rule and the uncapped stored rows,
verified empirically; existing scorecard fixtures re-checked); the
half-even -> half-up boundary shifts some delta displays by one unit at
exact .5 boundaries (intended, canonical).

Reviewer rules triggered: R1, R2, R3, R8, R10, R14 (extracted package change; test evidence; money-path change; billing figures; gate predicate change; both-direction probes).

## Mechanism

`_is_billed_repeat_item(item)` owns the `>= 2` rule;
`_repeat_ticket_count` sums `_ticket_count` over items passing it. The
scorecard's new `money.*` assertions re-apply the predicate to the
model's own `ranked_questions.data.rows` and verify every money figure
against `deflection_money` -- pure verification, no computation of new
values. The delta path swaps float rounding for the shared signed
Decimal quantizer.

## Intentional

- The guard lives in the SCORECARD (not a second gate): S8a already
  runs the scorecard at persist on raw + projected payloads, so the
  money guard rides the existing fail-closed boundary with zero new
  wiring.
- Exact equality, not tolerance: 13.50 x integer is exact at 2dp and
  annualized figures are quantized once at compute -- a tolerance would
  mask exactly the drift class P5-7 documents.
- One named predicate: the audit's second-side trap is a guard that
  re-hardcodes `tc >= 2` and diverges from the billing rule later.
- Delta rounding fixed at compute AND display: fixing only display
  leaves half-even sums in persisted deltas.

## Deferred

- Annualized-field EXPOSURE decision (operator, #1993).
- Ops sweep over already-persisted artifacts (S8a deferred item).
- Delta-report money reconciliation guard (deltas have no scorecard
  today; follow-up if delta drift materializes).

## Verification

- `python -m pytest tests/test_content_ops_deflection_report.py -q`
- `python -m pytest tests/test_atlas_content_ops_deflection_delivery.py -q`
- `python -m pytest tests/test_extracted_content_deflection_submit.py -q`
- Full CI mirror: bash `scripts/run_extracted_pipeline_checks.sh`
- Drifted fixtures reproduced failing through the persist gate before
  the fix was called done.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/content_ops_deflection_delivery.py` | 14 |
| `extracted_content_pipeline/deflection_delta.py` | 6 |
| `extracted_content_pipeline/deflection_money.py` | 12 |
| `extracted_content_pipeline/faq_deflection_report.py` | 119 |
| `plans/PR-Resolution-Audit-S8b-Money-Reconciliation.md` | 211 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 23 |
| `tests/test_content_ops_deflection_report.py` | 96 |
| `tests/test_run_deflection_full_report_qa_live_runner.py` | 15 |
| `tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py` | 5 |
| `tests/test_smoke_content_ops_deflection_pdf_export_validators.py` | 15 |
| **Total** | **516** |
