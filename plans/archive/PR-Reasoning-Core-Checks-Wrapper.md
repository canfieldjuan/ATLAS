# Reasoning Core Checks Wrapper

## Why this slice exists

`extracted_reasoning_core` has its own manifest, standalone smoke, import
guards, and product tests, but its full check path is currently embedded inside
`scripts/run_extracted_pipeline_checks.sh`. That keeps the reasoning product
operationally tied to the Content Ops runner even though the products are now
separate.

## Scope

1. Add a dedicated `scripts/run_extracted_reasoning_core_checks.sh` product
   check wrapper.
2. Have the broader Content Ops check runner delegate reasoning-core validation
   to that wrapper instead of duplicating the commands inline.
3. Remove merged #576 from the in-flight coordination table and claim this
   slice.
4. Update reasoning-core coordination state to reflect #576 as the latest
   merged reasoning-core documentation closeout.

### Files touched

- `scripts/run_extracted_reasoning_core_checks.sh`
- `scripts/run_extracted_pipeline_checks.sh`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Reasoning-Core-Checks-Wrapper.md`

## Mechanism

The new wrapper runs the same reasoning-core gates already present in the full
extracted pipeline runner:

- manifest validation;
- ASCII check;
- no-Atlas-fallback import check;
- forbidden Atlas reasoning import guard;
- standalone reasoning-core smoke;
- focused reasoning-core pytest suite plus the import-guard test.

The Content Ops runner keeps its Content Ops checks and Atlas wrapper tests,
but calls the reasoning-core wrapper for the core product checks.

## Intentional

- No runtime behavior changes.
- No new reasoning capability.
- No product API changes.
- No broad backlog or audit refresh.

## Deferred

- Any new reasoning-core provider ports or AI Content Ops reasoning behavior
  remain trigger-driven by a concrete runtime/product need.
- Removing the in-flight row for this PR after merge remains the standard
  post-merge coordination step.

## Verification

- Command: python -m pytest -q <reasoning-core test glob> tests/test_forbid_atlas_reasoning_imports.py; result: 295 passed.
- Command: bash scripts/run_extracted_reasoning_core_checks.sh; result: 295 passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh; result: 1361 passed, 1 existing torch/pynvml warning.
- Command: bash scripts/local_pr_review.sh; result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| New reasoning-core wrapper | ~20 |
| Pipeline runner delegation | ~25 |
| Coordination docs | ~8 |
| Plan doc | ~70 |
| **Total** | ~125 |
