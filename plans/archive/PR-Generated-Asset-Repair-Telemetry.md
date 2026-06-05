# PR-Generated-Asset-Repair-Telemetry

## Why this slice exists

The landing-page repair-attempt chain is now wired end to end: generator,
dispatch, input override, UI control, and cost preview. Operators can choose
repair attempts and see the cost impact, but the generated output still does
not explain what happened inside the repair loop.

When a landing-page draft repairs successfully, reviewers should be able to see
which earlier candidate failed the quality gate. When repair still fails,
operators should see the sequence of quality issues that were fed back to the
model.

## Scope (this PR)

1. Record a compact landing-page quality-repair history per generation run.
2. Include each parsed candidate's repair attempt number, pass/fail state,
   blockers, and repair issues.
3. Expose the repair history in `LandingPageGenerationResult.as_dict()`.
4. Persist the repair history in saved landing-page draft metadata.
5. Attach the repair history to quality-blocked and repair-unparseable errors.
6. Add focused landing-page generation tests.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Generated-Asset-Repair-Telemetry.md` | Plan doc for this telemetry slice. |
| `extracted_content_pipeline/landing_page_generation.py` | Record and expose landing-page quality-repair history. |
| `tests/test_extracted_landing_page_generation.py` | Cover success, quality-blocked, and repair-unparseable telemetry. |

## Mechanism

The landing-page generator already evaluates each parsed candidate through the
deterministic quality pack before deciding whether to repair or persist. This
slice records that decision as a small serializable row:

- `attempt`
- `passed`
- `blockers`
- `repair_issues`

The row is appended only after a candidate parses and receives a quality
report. Parse failures after a previous quality block include the history from
the earlier parsed candidate so operators can still see what drove the repair
prompt.

## Intentional

- No changes to repair behavior, prompts, quality-gate decisions, or cost
  estimates.
- No new database columns; draft metadata already carries generation
  diagnostics.
- No separate UI renderer; the existing execution result JSON and generated
  asset metadata can carry the field.
- No repair telemetry for blog posts or other generated assets in this slice.

## Deferred

- A polished generated-asset UI panel for repair history.
- Shared repair telemetry if other generated assets adopt the same repair-loop
  contract.
- Token-level or provider-cost telemetry per repair attempt.

## Verification

- pytest tests/test_extracted_landing_page_generation.py -q
  -> passed 25/25 tests.
- Python compile command over `extracted_content_pipeline/landing_page_generation.py`
  and `tests/test_extracted_landing_page_generation.py` -> passed 2/2 files.
- git diff --check -> passed.
- bash scripts/validate_extracted_content_pipeline.sh -> passed mapped-file
  and hard-import checks.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  -> passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt -> passed with
  0 Atlas runtime import findings.
- bash scripts/check_ascii_python.sh -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Landing-page generation | ~45 |
| Tests | ~70 |
| Total | ~190 |
