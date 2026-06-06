# PR — Content-Ops Triage + Experiment Contract (operating-model slice 2)

## Why this slice exists

`docs/content_ops_operating_model.md` stages the build; this is **slice 2 — triage gate +
experiment contract**. It adds the two lifecycle bookends the model hinges on but slice 1
(vocabulary) did not provide: **stage 0 "should this exist?"** triage, and the **stage 7
experiment contract** (the measurement plan frozen *before* publish). Like slice 1 these
are additive, behavior-neutral schemas that consume slice 1's vocabulary; the routing /
enforcement that *requires* them is a later slice.

Diff total runs slightly over the 400-LOC soft cap (see *Estimated diff size*). The excess
is test surface, not product: the module itself is ~130 LOC, and the tests grew with review
hardening (None/non-`str` completeness, the required-field set). The shippable surface
stays small.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

Extends the owned module `extracted_content_pipeline/review_contract.py` (no new file, no
manifest change) plus unit tests. Pure value types + pure helpers; no I/O, no Atlas
imports, no DB, no LLM:

- `TriageDecision` (StrEnum) — CREATE / CLONE_WINNER / DEFER / REJECT (the stage-0 verdict
  that keeps the pipeline from efficiently producing landfill).
- `TriageRequest` (frozen dataclass) — the stage-0 inputs: audience_segment,
  lifecycle_stage, business_goal, expected_behavior_change, channel, opportunity_size,
  reuse_potential, `risk_tier` (slice 1 `RiskTier`), why_now. `missing_fields()` /
  `is_complete()` so triage can't be waved through blank.
- `ExperimentContract` (frozen dataclass) — the stage-7 measurement plan: hypothesis,
  primary_metric, secondary_metric, attribution_window_days, audience, comparison,
  min_sample_size, success_definition, inconclusive_definition, decision_if_works,
  decision_if_not (required set mirrors the doc's stage-7 list). `missing_fields()` /
  `is_complete()` so a piece can't publish against an empty plan; completeness treats
  `None`/non-`str` as missing rather than raising.


### Files touched

- `extracted_content_pipeline/review_contract.py`
- `plans/PR-Content-Ops-Triage-Experiment.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_content_triage_experiment.py`

## Mechanism

Frozen dataclasses + pure validation helpers, same conventions as slice 1 (`StrEnum` with
the Python-3.10 fallback already imported in the module). `missing_fields()` returns the
empty required-field names in declaration order; `is_complete()` is `not missing_fields()`.
`TriageRequest.risk_tier` reuses `RiskTier` so triage and the later risk-tier routing
table speak the same vocabulary. No existing code path changes.

## Intentional

- Additive only — the generated-asset review API and existing flows are untouched.
- Validation is *completeness*, not *quality*: these helpers check that required fields are
  filled, not whether the hypothesis is good. Quality stays a human/market call per the doc.
- `attribution_window_days` and `min_sample_size` are typed numbers with light bounds
  (`> 0`); richer statistical validation is out of scope.

## Deferred

- Routing that *requires* a passing triage before drafting, and a frozen experiment
  contract before publish (consumes these schemas) — later slice.
- Wiring `ReviewDecision` into `api/generated_assets.py` + risk-tier routing table
  enforcement — later slice.
- Claims map (slice 3); Content-PR coverage matrix (slice 4); calibration library +
  adversarial pass (slice 5). Multi-model disagreement orchestration stays parked.

## Verification

- pytest `tests/test_extracted_content_triage_experiment.py`
- `scripts/check_ascii_python.sh` (run via bash) -- ASCII gate
- `scripts/check_extracted_imports.py` (run via python3) -- import structure
- `scripts/audit_extracted_pipeline_ci_enrollment.py` -- new test enrolled
- `scripts/audit_pr_session_drift.py` + `scripts/sync_pr_plan.py` -- plan shape/drift

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/review_contract.py` | 131 |
| `plans/PR-Content-Ops-Triage-Experiment.md` | 88 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_extracted_content_triage_experiment.py` | 236 |
| **Total** | **456** |
