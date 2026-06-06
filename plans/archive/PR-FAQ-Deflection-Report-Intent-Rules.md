# FAQ Deflection Report Intent Rules

## Why this slice exists

The FAQ generator and hosted service support custom intent rules, but the
customer-facing deflection report CLI cannot pass them through. That blocks the
report workflow from using customer-specific symptom language to cluster repeat
questions when the source rows do not already carry a useful taxonomy.

This is a vertical slice because it adds one end-to-end operator control to the
real report CLI and proves the generated report changes shape through the
existing FAQ generator, report renderer, output-check gate, and result artifact.

## Scope (this PR)

Ownership lane: content-ops/deflection-report
Slice phase: Vertical slice

1. Add repeatable `--intent-rule topic=keyword,keyword` support to
   `scripts/build_content_ops_deflection_report.py`.
2. Pass custom intent rules ahead of the default FAQ rules so operator-provided
   customer language takes precedence.
3. Record the custom intent rules in `--result-output` config metadata.
4. Prove custom rules cluster weakly-taxonomized support rows into one FAQ
   opportunity in the customer-facing report.
5. Prove malformed intent rules fail before writing a report.

### Files touched

| File | Purpose |
|---|---|
| `scripts/build_content_ops_deflection_report.py` | Add CLI parsing, config metadata, and generator pass-through for custom intent rules. |
| `tests/test_content_ops_deflection_report.py` | Add end-to-end CLI coverage for custom intent rules and malformed rule rejection. |
| `plans/PR-FAQ-Deflection-Report-Intent-Rules.md` | Plan and verification contract. |

## Mechanism

The CLI accepts repeatable `--intent-rule` values shaped as
`topic=keyword,keyword`. Parsing trims blanks and de-duplicates keywords
case-insensitively within each rule. The report builder passes:

```python
intent_rules = (*custom_intent_rules, *DEFAULT_INTENT_RULES)
```

into `build_ticket_faq_markdown(...)`, matching the FAQ CLI precedence rule:
custom customer language is evaluated before default taxonomy rules.

## Intentional

- No JSON `--rule-file` parity in this slice. The thinnest useful path is the
  inline intent rule operator control; broader rules-file parity can follow if
  operators need it.
- No change to default report behavior. When no `--intent-rule` is provided,
  the report still uses the generator defaults.
- Custom rules prepend, not append, because customer-provided wording should win
  over generic defaults for a paid report run.

## Deferred

- Parked hardening: none.
- Future vertical slice: add deflection-report CLI `--rule-file` parity only if
  operators need reusable customer rule bundles instead of inline command flags.

## Verification

- Command: python -m pytest tests/test_content_ops_deflection_report.py -q
- Command: python -m py_compile scripts/build_content_ops_deflection_report.py tests/test_content_ops_deflection_report.py
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Report-Intent-Rules.md
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Report-Intent-Rules.md
- Command: python scripts/audit_plan_doc_diff_size.py plans/PR-FAQ-Deflection-Report-Intent-Rules.md origin/main
- Command: git diff --check
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-report-intent-rules.md

## Estimated diff size

| Area | Estimate |
|---|---:|
| `scripts/build_content_ops_deflection_report.py` | 42 LOC |
| `tests/test_content_ops_deflection_report.py` | 94 LOC |
| `plans/PR-FAQ-Deflection-Report-Intent-Rules.md` | 82 LOC |
| **Total** | **218 LOC** |
