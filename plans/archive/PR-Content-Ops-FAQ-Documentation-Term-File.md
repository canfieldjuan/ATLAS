# PR-Content-Ops-FAQ-Documentation-Term-File

## Why this slice exists

Vocabulary-gap detection can compare customer language against documentation
terms, and custom rule files now make intent and alias rules reusable. The
remaining operator friction is documentation terms: larger SMB and mid-market
docs exports should not require dozens of repeated `--documentation-term`
flags.

This slice adds a small documentation-term file input to the existing FAQ CLI
so help-center headings or glossary terms can be loaded from a checked or
exported text file.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add repeatable `--documentation-term-file` support to the FAQ Markdown CLI.
2. Read one documentation term per nonblank line, with `#` comment lines
   ignored for copyable examples.
3. Combine explicit `--documentation-term` values before file-loaded terms and
   expose both the file paths and resolved terms in compact result JSON.
4. Add a checked example documentation-term file and CLI tests for success and
   missing-file failure.
5. Document the term-file command in the extracted README and host runbook.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Documentation-Term-File.md` | Plan doc for this slice. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Adds term-file CLI parsing, result config, and validation. |
| `extracted_content_pipeline/examples/faq_documentation_terms.txt` | Checked one-term-per-line example. |
| `extracted_content_pipeline/README.md` | Documents documentation-term file usage. |
| `extracted_content_pipeline/docs/host_install_runbook.md` | Mirrors the operator command in host docs. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers term-file success and missing-file failure. |

## Mechanism

The CLI accepts `--documentation-term-file PATH` repeatedly. Each file is read
as UTF-8 plain text. Blank lines and lines whose first non-space character is
`#` are ignored; all other lines become documentation terms. Inline
`--documentation-term` values are placed before file values, then cleaned and
deduplicated case-insensitively before calling `build_ticket_faq_markdown`.

The result JSON records `documentation_term_files` and the resolved
`documentation_terms` list so large-upload runs remain auditable.

## Intentional

- Plain text only in this slice. JSON or CSV documentation exports can be added
  later if a host export requires them.
- No vocabulary-gap matching changes. The existing deterministic mapping logic
  consumes the resolved documentation-term list.
- Inline `--documentation-term` values stay first because command-line values
  are the most local operator intent.
- CLI-only helper names are prefixed with `_cli_` so they do not look like the
  FAQ library's private documentation-term helpers in cross-layer audits.

## Deferred

- CSV/JSON documentation-term imports.
- Hosted UI upload/entry fields for documentation terms and rule files.
- Larger customer glossary templates beyond the checked copyable example.

## Verification

- Focused documentation-term-file FAQ CLI pytest - passed, 2 tests.
- Full FAQ pytest for `tests/test_extracted_ticket_faq_markdown.py` - passed,
  118 tests.
- Py compile for affected Python files - passed.
- Git whitespace check - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.
- Local PR review against origin/main - passed; cross-layer caller hints for
  generic script helpers were inspected and no external caller impact was found.
- Reviewer NIT test update: focused documentation-term-file FAQ CLI pytest
  passed, 3 tests; git whitespace check passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| CLI parser/result changes | ~55 |
| Example file | ~5 |
| README/runbook docs | ~25 |
| Tests | ~55 |
| **Total** | ~220 |
