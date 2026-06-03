# PR: Content Ops Marketer Stat Card Output

## Why this slice exists

The marketer reviews-as-input lane has completed social posts, ad copy, and
quote cards as executable marketer assets. The remaining deferred card-family
output from the quote-card handoff is `stat_card`: a short visual-card draft
that turns a source-backed numeric metric into a bounded campaign asset.

This slice adds the first stat-card vertical slice as a deterministic,
source-evidence-only generator. It mirrors the first `quote_card` output slice:
register an executable output, generate bounded drafts from `source_material`,
wire host execution, and prove preview/plan/run behavior. The extra invariant
for this output is numeric-claim validation: a stat card only emits when the
numeric value is present in the source evidence text, so the asset cannot
invent a metric from an unsupported field.

The diff may exceed the 400 LOC soft cap for the same indivisible first-output
reason as prior marketer-output slices: the service, manifest entry, catalog
exposure, generation plan, executor dispatch, host wiring, reasoning no-op
policy, UI catalog fixture, and focused tests need to land together or
`stat_card` is either invisible, not runnable, or not protected by the numeric
validator.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input

Slice phase: Vertical slice

1. Add a package-owned deterministic `StatCardGenerationService` that turns
   source material into bounded stat-card drafts with source ids, metric label,
   numeric value, claim text, evidence text, and non-fatal source-material
   warnings.
2. Validate every emitted numeric claim against the source evidence text. Rows
   with non-numeric metric values, no known metric, or a value that is not
   present in evidence are skipped with explicit warnings.
3. Register `stat_card` in the control-surface catalog, generation plan,
   executor service bundle, host service factory, and no-op reasoning policy.
4. Keep stat-card generation source-evidence-only: no LLM prompt, persistence,
   generated-asset review API/UI branch, package-default enrollment, visual
   export, or #1268 variant fan-out in this slice.
5. Add focused tests for service output/warnings/limits, numeric-claim failure
   detection, preview/plan mapping, executor dispatch, host factory wiring, and
   catalog readiness.

### Files touched

- `plans/PR-Content-Ops-Marketer-Stat-Card-Output.md`
- `extracted_content_pipeline/stat_card_generation.py`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/control_surfaces.py`
- `extracted_content_pipeline/docs/control_surface_preview_api.md`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/reasoning_policy.py`
- `atlas_brain/_content_ops_services.py`
- `tests/test_extracted_stat_card_generation.py`
- `tests/test_extracted_content_control_surfaces.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_extracted_content_reasoning_policy.py`
- `tests/test_atlas_content_ops_execution_services.py`
- `.github/workflows/extracted_pipeline_checks.yml`
- `scripts/audit_extracted_pipeline_ci_enrollment.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_audit_extracted_pipeline_ci_enrollment.py`
- `atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json`

## Mechanism

`StatCardGenerationService.generate(...)` accepts `source_material`, expands the
same source-material bundles as quote cards through
`source_material_to_source_rows(...)`, then normalizes each row with
`source_row_to_campaign_opportunity(...)`. It evaluates a small allowlist of
numeric source fields (`nps_score`, `csat_score`, `opportunity_score`,
`urgency_score`, `rating`, and count/rate style fields) and emits at most one
card per usable row.

For every candidate metric, the generator:

```python
value = _numeric_value(opportunity.get(field))
if value is None:
    warn("invalid_stat_card_metric")
elif not _evidence_snippet_for_value(evidence, value, max_evidence_chars):
    warn("unsupported_numeric_claim")
else:
    emit_stat_card(...)
```

The generated card is deterministic and no-cost:

```python
{
    "id": source_id,
    "theme": "customer_metric",
    "metric_label": "NPS score",
    "metric_value": 42,
    "metric_display": "42",
    "claim": "NPS score: 42",
    "headline": "Customer metric for Zendesk",
    "supporting_text": "Use this stat to frame customer sentiment.",
    "evidence": "NPS score dropped to 42 after renewal.",
    "source_id": source_id,
    "source_type": "survey_response",
}
```

The bounded `evidence` field is selected around the matched numeric value rather
than blindly truncating from the front, so a card that claims `NPS score: 42`
still includes `42` in its returned evidence snippet even when the source row is
long.

`ContentOpsExecutionServices` gets a `stat_card` slot and dispatch handler that
passes `scope`, `target_mode`, `source_material`, `limit`, and
`max_text_chars`. `stat_card` uses `reasoning_requirement="absent"` because the
slice is deterministic and evidence-bound.

## Intentional

- No persistence or generated-asset review branch. This first slice proves the
  stat-card output shape and numeric guard before adding durable review/export
  state.
- No package-default enrollment yet. Review/competitive package defaults should
  add `stat_card` only after the executable output shape is accepted.
- No visual template/export generation. The card payload includes copy and
  evidence fields that a later creative export can consume.
- No LLM prompt. Numeric claims are too easy to hallucinate; this slice keeps
  them deterministic and source-evidence-only.
- No #1268 output-variations work. `variant_count` and style/channel fan-out
  remain outside this lane.

## Deferred

- Add `stat_card` to review and competitive input-package defaults after the
  output shape is accepted.
- Persist generated stat-card drafts into the generated-assets review queue in
  a follow-up handoff slice.
- Add visual template/export generation for quote cards and stat cards after
  review/export rows exist.
- LLM/style/channel variants and #1268-style output variations remain future
  product polish.

## Parked hardening

None.

## Verification

- Passed: `pytest tests/test_extracted_stat_card_generation.py tests/test_audit_extracted_pipeline_ci_enrollment.py tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py tests/test_extracted_content_reasoning_policy.py tests/test_extracted_content_control_surface_api.py tests/test_atlas_content_ops_execution_services.py -q` (356 passed, 1 skipped)
- Passed: `python -m json.tool atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json >/dev/null`
- Passed: `python -m py_compile extracted_content_pipeline/stat_card_generation.py extracted_content_pipeline/control_surfaces.py extracted_content_pipeline/generation_plan.py extracted_content_pipeline/content_ops_execution.py extracted_content_pipeline/reasoning_policy.py atlas_brain/_content_ops_services.py scripts/audit_extracted_pipeline_ci_enrollment.py tests/test_extracted_stat_card_generation.py tests/test_audit_extracted_pipeline_ci_enrollment.py`
- Passed: `git diff --check`
- Passed: `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main` (OK: 146 matching tests are enrolled.)
- Passed: `bash scripts/validate_extracted_content_pipeline.sh`
- Passed: `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
- Passed: `python scripts/audit_extracted_standalone.py --fail-on-debt`
- Passed: `bash scripts/check_ascii_python.sh`
- Passed: `bash scripts/run_extracted_pipeline_checks.sh` (3044 passed, 10 skipped, 1 warning)
- Passed: `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-stat-card-output-pr-body.md`

## Estimated diff size

Actual: 21 files, +1201 / -5. This is above the 400 LOC soft cap for the indivisible
first-output wiring reason described in **Why this slice exists**.

| Area | Estimated LOC |
|---|---:|
| Stat-card service and validator | ~435 |
| Catalog/plan/executor/host/reasoning/docs/fixture wiring | ~95 |
| CI enrollment audit/workflow protection | ~33 |
| Focused tests | ~462 |
| Plan doc and CI enrollment | ~176 |
| **Total** | **~1201** |
