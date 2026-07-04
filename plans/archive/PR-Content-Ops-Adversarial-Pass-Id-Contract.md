# PR-Content-Ops-Adversarial-Pass-Id-Contract

## Why this slice exists

Issue #1493 records an undocumented edge in adversarial corroboration: repeated
`pass_id` rows are treated as one independent pass for corroboration, while all
substantiated rows still fold into non-blocking Objections comments. The current
direction is conservative because malformed duplicate IDs cannot inflate
corroboration, but the behavior is only implied by the loop. This slice turns
that first-occurrence-wins behavior into an explicit package contract and adds
regressions for the two logged edges.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Production hardening

1. Document that adversarial corroboration counts only the first submitted row
   for each `pass_id`.
2. Add a regression where a later row reuses a `pass_id` with different
   categories and remains excluded from corroboration.
3. Add a regression where direct callers submit blank `pass_id` values and the
   blank ID is still treated as one duplicated pass identity.
4. Leave decode-layer duplicate rejection and per-category pass-id surfacing out
   of this slice.

### Files touched

- `plans/PR-Content-Ops-Adversarial-Pass-Id-Contract.md`
- `extracted_content_pipeline/adversarial_pass.py`
- `tests/test_extracted_content_adversarial_pass.py`

### Review Contract

- Acceptance criteria:
  - [ ] The adversarial-pass package documents first-occurrence-wins behavior
        for reused `pass_id` values.
  - [ ] A divergent reused-`pass_id` fixture proves later-row categories do not
        count toward corroboration.
  - [ ] A blank-`pass_id` direct-call fixture proves blank IDs dedupe like any
        other repeated pass identity.
  - [ ] Existing distinct-pass corroboration behavior remains unchanged.
- Affected surfaces: extracted package contract and unit tests.
- Risk areas: backcompat, maintainability.
- Reviewer rules triggered: R1, R2, R10, R13, R14.

## Mechanism

`corroborated_categories_across` already tracks seen pass IDs and skips later
rows with the same ID. This PR keeps that behavior and expands the docstring so
callers understand that the first occurrence is authoritative for
corroboration. The new tests lock the issue's malformed-input edge directly and
also cover the direct-host blank-ID case that the MCP decode fallback avoids.

## Intentional

- First occurrence wins is preserved instead of switching to union-by-ID. A
  duplicate ID most plausibly represents a retry, and the conservative failure
  direction is to under-count rather than inflate an advisory corroboration
  signal.
- The MCP decode layer is not changed here. Rejecting duplicate submitted IDs is
  a broader transport contract decision and would touch host/MCP surfaces rather
  than the deterministic package contract.
- Existing Objections folding remains unchanged. Findings are still evidence
  for editors, not blocking verdicts.

## Deferred

- Per-category pass-id/source surfacing remains deferred from the corroboration
  surfacing work. That future slice can decide whether the UI/result artifact
  should expose skipped duplicate IDs or reject them earlier.
- Decode-layer duplicate-ID rejection remains a possible future follow-up if
  transport clients need hard validation instead of conservative counting.
- Parked hardening: none.

## Verification

- pytest tests/test_extracted_content_adversarial_pass.py -- 28 passed.
- bash scripts/validate_extracted_content_pipeline.sh -- passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt -- passed.
- bash scripts/check_ascii_python.sh -- passed.
- bash scripts/run_extracted_pipeline_checks.sh -- 3893 passed, 10 skipped.
- git diff --check -- passed.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_adversarial_pass_id_contract_pr_body.md -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-Adversarial-Pass-Id-Contract.md` | 99 |
| `extracted_content_pipeline/adversarial_pass.py` | 6 |
| `tests/test_extracted_content_adversarial_pass.py` | 47 |
| **Total** | **152** |

Under the 400 LOC target.
