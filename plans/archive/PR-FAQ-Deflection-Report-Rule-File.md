# FAQ Deflection Report Rule File

## Why this slice exists

The FAQ Markdown CLI supports reusable JSON rule files for customer-specific
intent and vocabulary mappings, but the customer-facing deflection report CLI
only accepts inline `--intent-rule` and `--vocabulary-gap-rule` values. Paid
report runs need the same reusable customer mapping path without asking
operators to repeat long command lines.

This slice extracts the existing FAQ CLI rule-file parser into one shared script
helper before wiring the report CLI to it. That keeps rule parsing, validation,
and precedence in one place instead of reimplementing the same contract in a
second CLI.

The estimated diff is over the 400 LOC target because the parser is being moved
to one owner while preserving the existing FAQ CLI validation coverage and
adding report-level proof. Keeping two independent parser copies would be
smaller in this PR but would recreate the cross-layer drift pattern this lane
has been tightening.

## Scope (this PR)

Ownership lane: content-ops/deflection-report
Slice phase: Vertical slice

1. Move the existing JSON rule-file parser from the FAQ CLI into a shared script
   helper.
2. Keep the FAQ CLI behavior and tests on the shared parser.
3. Add repeatable JSON rule-file support to the deflection report CLI.
4. Combine explicit CLI rules before file-loaded rules so command-line values
   still win when multiple mappings match.
5. Record rule-file paths and resolved rule config in the report result JSON.

### Files touched

| File | Purpose |
|---|---|
| `scripts/content_ops_faq_cli_rules.py` | Shared JSON rule-file and inline-rule parser. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Import the shared parser instead of carrying a private copy. |
| `scripts/build_content_ops_deflection_report.py` | Add JSON rule-file flag handling and shared parser routing. |
| `tests/test_content_ops_deflection_report.py` | Prove report CLI rule files affect clustering/vocabulary and reject bad files. |
| `plans/PR-FAQ-Deflection-Report-Rule-File.md` | Plan and verification contract. |

## Mechanism

The shared helper exposes:

```python
load_rule_files(paths)
parse_intent_rules(values)
parse_vocabulary_gap_rules(values)
```

Both CLIs use those functions. The deflection report CLI resolves:

```python
intent_rules = (*inline_intent, *file_intent, *DEFAULT_INTENT_RULES)
vocabulary_gap_rules = (*inline_vocab, *file_vocab)
```

and passes the resolved rules to `build_ticket_faq_markdown(...)`.

## Intentional

- JSON rule files only. This matches the existing FAQ CLI rule-file contract.
- This touches the FAQ CLI to remove duplicate parser ownership, but the
  existing FAQ CLI tests continue to cover the parser's validation branches.
- No docs update in this slice; the report CLI behavior is covered by `--help`
  and focused tests. Operator docs can follow once the CLI surface settles.

## Deferred

- Parked hardening: none.
- Future product-polish slice: add report CLI examples to the extracted README
  and host runbook after this shared parser lands.

## Verification

- Command: python -m pytest tests/test_content_ops_deflection_report.py tests/test_extracted_ticket_faq_markdown.py -q
- Command: python -m py_compile scripts/content_ops_faq_cli_rules.py scripts/build_content_ops_deflection_report.py scripts/build_extracted_ticket_faq_markdown.py tests/test_content_ops_deflection_report.py tests/test_extracted_ticket_faq_markdown.py
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Report-Rule-File.md
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Report-Rule-File.md
- Command: python scripts/audit_plan_doc_diff_size.py plans/PR-FAQ-Deflection-Report-Rule-File.md origin/main
- Command: git diff --check
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-report-rule-file.md

## Estimated diff size

| Area | Estimate |
|---|---:|
| `scripts/content_ops_faq_cli_rules.py` | 193 LOC |
| `scripts/build_extracted_ticket_faq_markdown.py` | 193 LOC |
| `scripts/build_content_ops_deflection_report.py` | 68 LOC |
| `tests/test_content_ops_deflection_report.py` | 118 LOC |
| `plans/PR-FAQ-Deflection-Report-Rule-File.md` | 97 LOC |
| **Total** | **669 LOC** |
