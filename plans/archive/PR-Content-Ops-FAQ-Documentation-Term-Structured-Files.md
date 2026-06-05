# PR-Content-Ops-FAQ-Documentation-Term-Structured-Files

## Why this slice exists

PR-Content-Ops-FAQ-Documentation-Term-File made documentation terms reusable
through a plain text file, but the next real customer source is likely a docs
export from a help center or knowledge base. Those exports commonly arrive as
CSV, JSON, or JSONL rows with `title`, `heading`, or `term` fields.

This slice keeps the existing `--documentation-term-file` flag and teaches it
to load common structured documentation-term exports by suffix.

This is over the 400 LOC soft budget after review because the structured-file
surface needs the parser and error-path fixture coverage to ship together:
JSON, JSONL, CSV, malformed JSON, malformed JSONL, BOM CSV headers, quoted
multiline CSV cells, and nested JSON term values are all part of the same
operator input contract.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Extend `--documentation-term-file` auto-detection to JSON, JSONL, and CSV
   files while preserving existing plain-text behavior.
2. Extract terms from common keys such as `documentation_term`, `term`,
   `heading`, `title`, `page_title`, `name`, and `label`.
3. Accept JSON bundles with list keys such as `documentation_terms`, `terms`,
   `headings`, `documents`, `pages`, `articles`, `rows`, and `data`.
4. Fail cleanly for malformed JSON/JSONL and structured files that contain no
   usable terms.
5. Add focused CLI tests for JSON, JSONL, CSV, and malformed JSONL handling.
6. Document the supported documentation-term file formats in the extracted
   README and host runbook.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Documentation-Term-Structured-Files.md` | Plan doc for this structured term-file slice. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Adds structured term-file loading for JSON/JSONL/CSV. |
| `extracted_content_pipeline/README.md` | Documents structured documentation-term file support. |
| `extracted_content_pipeline/docs/host_install_runbook.md` | Mirrors host-facing format notes. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers JSON, JSONL, CSV, and malformed JSONL term files. |

## Mechanism

The existing `--documentation-term-file` path now resolves by suffix. JSON files
load either a list or an object bundle. JSONL files load one JSON value per
nonblank line. CSV files use `csv.DictReader`. Unknown suffixes keep the
existing plain text behavior.

Structured rows are intentionally conservative: only common documentation-term
keys are extracted, and bundle traversal only follows known list keys. This
avoids treating unrelated metadata as headings while still supporting typical
help-center exports.

## Intentional

- No new CLI flag. Operators can keep using `--documentation-term-file` and let
  suffix-based detection choose the parser.
- No vocabulary-gap matching behavior changes. This only changes how
  documentation terms enter the existing deterministic mapper.
- JSON/CSV parsing is local to the FAQ CLI for now; a shared docs-export loader
  can be introduced if another command needs the same contract.

## Deferred

- Explicit `--documentation-term-format` override if hosts need suffixless
  structured files.
- Hosted UI upload/entry fields for documentation terms and rule files.
- Richer docs export metadata, such as URLs or section paths, in result JSON.

## Verification

- Focused structured documentation-term-file FAQ CLI pytest - passed, 5 tests.
- Full FAQ pytest for `tests/test_extracted_ticket_faq_markdown.py` - passed,
  124 tests.
- Py compile for affected Python files - passed.
- Git whitespace check - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.
- Reviewer feedback update: focused structured term-file regressions passed, 7
  tests; full FAQ pytest passed, 126 tests; py compile and git whitespace check
  passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~90 |
| CLI structured parser | ~130 |
| README/runbook docs | ~15 |
| Tests | ~230 |
| **Total** | ~465 |
