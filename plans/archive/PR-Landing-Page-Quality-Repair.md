# PR-Landing-Page-Quality-Repair

## Why this slice exists

The landing-page generator now has a readiness contract, export/readiness
summaries, quality-gate validators, and a prompt aligned to those validators.
The remaining source gap is generation behavior: when the LLM returns valid
JSON that fails the deterministic quality gate, the service currently stops
immediately and skips persistence.

That is wasteful for repairable defects like a missing CTA block or invalid
slug. This slice adds one targeted quality-repair pass that feeds the gate
issues back to the model and persists only if the repaired draft passes.

## Scope (this PR)

1. Add a landing-page `quality_repair_attempts` config default of 1.
2. Let callers override `quality_repair_attempts` per generation call.
3. Retry parsed-but-quality-blocked landing-page JSON with the quality issues
   in the user prompt.
4. Keep parse retry and quality repair separate.
5. Accumulate token usage and parse-attempt counts across repair passes.
6. Persist only the passing repaired draft.
7. Add tests for successful repair, failed repair, and legacy block behavior
   with repair disabled.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Quality-Repair.md` | Plan doc for this implementation slice. |
| `extracted_content_pipeline/landing_page_generation.py` | Add quality-repair retry flow and repair metadata. |
| `tests/test_extracted_landing_page_generation.py` | Cover repair success, repair failure, and disabled-repair blocking. |

## Mechanism

The generator still calls `_generate_one(...)` for JSON generation and parse
retry. The new outer loop runs around that parsed output:

1. Generate and parse a landing-page JSON candidate.
2. Run the deterministic landing-page quality gate.
3. If it passes, build and save the draft.
4. If it fails and repair attempts remain, call the LLM again with the quality
   issues included in the user prompt.
5. If the repaired candidate still fails, return `quality_blocked` and do not
   save.

Usage metadata is accumulated across all LLM calls. The saved draft stores the
final model, total usage, total parse attempts, and
`generation_quality_repair_attempts`.

## Intentional

- No prompt-file changes.
- No quality-pack changes.
- No export/API changes.
- No frontend changes.
- No persistence of failed intermediate drafts.
- No repair loop for unparseable responses beyond the existing parse retry
  mechanism.

## Deferred

- Threading `quality_repair_attempts` through the Content Ops run-plan/control
  surface.
- Adding the same quality-repair behavior to other generated asset services.
- Adding richer operator telemetry for every failed intermediate candidate.

## Verification

- `pytest tests/test_extracted_landing_page_generation.py -q` -> passed 24/24
  tests.
- `pytest tests/test_extracted_landing_page_generation.py tests/test_extracted_landing_page_export.py tests/test_extracted_quality_gate_landing_page_pack.py -q`
  -> passed 62/62 tests.
- Python compile command over `extracted_content_pipeline/landing_page_generation.py`
  and `tests/test_extracted_landing_page_generation.py` -> passed 2/2 files.
- `git diff --check` -> passed with 0 whitespace errors.
- `bash scripts/local_pr_review.sh origin/main` -> passed 3/3 top-level
  checks: pre-push audit wrapper, plan/code consistency, and `git diff
  --check`.
- The pre-push audit wrapper inside local review reported all 8 internal checks
  passed: MCP tool counts, MCP port assignments, MCP tool-name inventories,
  extracted manifest sync, plan shape, plan files touched, plan diff size, and
  ASCII Python policy.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~95 |
| Landing-page generation | ~135 |
| Tests | ~95 |
| Total | ~325 |
