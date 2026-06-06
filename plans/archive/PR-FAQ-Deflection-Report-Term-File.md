# PR-FAQ-Deflection-Report-Term-File

## Why this slice exists

The customer-facing `faq_deflection_report` CLI now accepts custom intent and
vocabulary rule files, but it still requires documentation terms to be repeated
inline with `--documentation-term`. The underlying FAQ CLI already supports
documentation-term files, and the deflection report docs describe the same
glossary workflow around the report builder. That creates operator friction and
one more place where the report deliverable can drift from the FAQ generator it
wraps.

This slice keeps the report product moving without touching the hosted-search
blocker or the macro-writeback branch.

The diff is over the 400 LOC soft cap because the indivisible part of the slice
is moving the existing documentation-term-file parser into the shared CLI module
instead of copying it into the report CLI. The large deletion from the FAQ CLI
is the drift-reduction part of the change.

## Scope (this PR)

Ownership lane: content-ops/deflection-report

Slice phase: Product polish

1. Move the existing FAQ CLI documentation-term-file parser into the shared FAQ
   CLI rules module.
2. Keep the FAQ CLI behavior unchanged while importing the shared parser.
3. Add `--documentation-term-file` and `--documentation-term-format` to the
   deflection report CLI.
4. Add focused deflection report CLI tests for a JSON documentation-term file
   and an invalid documentation-term file.

### Files touched

| File | Purpose |
|---|---|
| `scripts/content_ops_faq_cli_rules.py` | Hosts the shared documentation-term-file parser. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Imports the shared parser and keeps existing CLI behavior. |
| `scripts/build_content_ops_deflection_report.py` | Adds documentation-term-file flags and resolved config metadata. |
| `tests/test_content_ops_deflection_report.py` | Proves deflection report term-file success and fail-closed parsing. |
| `plans/PR-FAQ-Deflection-Report-Term-File.md` | Documents this slice contract. |

## Mechanism

`content_ops_faq_cli_rules.py` already owns shared FAQ CLI parsing for intent
and vocabulary rules. This slice moves the existing documentation-term loader
there too:

```python
documentation_terms = parse_documentation_terms(
    args.documentation_term,
    args.documentation_term_file,
    args.documentation_term_format,
)
```

Both CLIs then call the same parser. The deflection report CLI passes the
resolved terms into `build_ticket_faq_markdown`, and the result JSON records the
term files, requested format, and resolved terms for operator audit.

## Intentional

- This does not add a new rule-file schema for documentation terms. Rule files
  stay limited to intent and vocabulary-gap rules so existing validation remains
  strict.
- This does not change report Markdown rendering or the hosted execute route.
  The route already accepts resolved `faq_documentation_terms` through request
  inputs; this slice is only offline report CLI ergonomics.
- Existing parser behavior and error messages are preserved so the large FAQ
  CLI test set remains the compatibility guard.

## Deferred

- Parked hardening: none.
- Hosted/deployed deflection report validation remains deferred until the
  operator-provisioned Atlas host and JWT tracked outside this PR are available.

## Verification

- Command: python -m py_compile scripts/content_ops_faq_cli_rules.py scripts/build_extracted_ticket_faq_markdown.py scripts/build_content_ops_deflection_report.py tests/test_content_ops_deflection_report.py tests/test_extracted_ticket_faq_markdown.py
  - Result: passed.
- Command: python -m pytest tests/test_content_ops_deflection_report.py tests/test_extracted_ticket_faq_markdown.py -q -k "deflection_report_cli or documentation_term_file"
  - Result: 26 passed, 133 deselected.
- Command: python -m pytest tests/test_content_ops_deflection_report.py -q
  - Result: 18 passed.
- Command: python -m pytest tests/test_extracted_ticket_faq_markdown.py -q
  - Result: 141 passed.
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Report-Term-File.md
  - Result: passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Report-Term-File.md
  - Result: passed.
- Command: git diff --check
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Shared parser extraction | 219 |
| Deflection report CLI wiring | 29 |
| FAQ CLI import cleanup | 223 |
| Tests | 101 |
| Plan doc | 106 |
| **Total** | **678** |
