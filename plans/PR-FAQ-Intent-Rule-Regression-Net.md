# PR-FAQ-Intent-Rule-Regression-Net

## Why this slice exists

PR-FAQ-Deflection-Report-Intent-Rule-Input shipped the hosted custom intent-rule
path. The review was LGTM, but it identified two useful regression gaps: the
double-normalize plan/dispatch path was manually verified but not pinned, and
the deflection-report execute path was manually verified but only plan-tested.

This slice adds the smallest regression net for those two behaviors without
changing product code.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-ui

Slice phase: Functional validation

1. Add an idempotency test for `normalize_intent_rules`.
2. Add an execute-level `faq_deflection_report` test proving hosted custom
   intent rules affect report grouping.
3. In that same report execute test, prove the fail-closed answer-evidence
   promise remains intact when no resolution evidence is uploaded.

### Files touched

| File | Purpose |
|---|---|
| `tests/test_extracted_ticket_faq_markdown.py` | Adds the intent-rule idempotency regression test. |
| `tests/test_extracted_content_ops_execution.py` | Adds the deflection-report execute regression test. |
| `plans/PR-FAQ-Intent-Rule-Regression-Net.md` | Documents this slice contract. |

## Mechanism

The idempotency test normalizes mixed line/object rules, feeds the normalized
tuple back through the same function, and asserts the result is unchanged.

The report execute test runs `faq_deflection_report` through
`execute_content_ops_from_mapping` with a hosted intent rule and no
`resolution_text`. It asserts the generated report groups both tickets under
the custom topic while summary counts and item statuses remain no-proven.

## Intentional

- No production code changes are included. The reviewed behavior was already
  correct; this slice only locks it.
- No UI changes are included because the UI input helper was covered in the
  prior PR and the gap was backend execute coverage.

## Deferred

- Parked hardening considered: none. The slice addresses the two relevant
  review follow-ups directly.

## Verification

- Command: pytest tests/test_extracted_ticket_faq_markdown.py::test_normalize_intent_rules_is_idempotent tests/test_extracted_content_ops_execution.py::test_execute_applies_hosted_faq_intent_rules_to_deflection_report_without_inventing_answers
  - Result: 2 passed.
- Command: pytest tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_content_ops_execution.py
  - Result: 204 passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - Result: passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Result: passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - Result: passed.
- Command: bash scripts/check_ascii_python.sh
  - Result: passed.
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Intent-Rule-Regression-Net.md
  - Result: passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Intent-Rule-Regression-Net.md
  - Result: passed.
- Command: git diff --check
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| FAQ normalizer test | 15 |
| Deflection report execute test | 55 |
| Plan doc | 83 |
| **Total** | **147** |
