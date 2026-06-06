# PR-Content-Ops-FAQ-Documentation-Term-Format

## Why this slice exists

Vocabulary-gap detection depends on loading the customer's existing
documentation terms. The CLI already accepts documentation-term files as text,
JSON, JSONL, or CSV, but format detection is suffix-based. Real exports often
arrive as suffixless files from help-center or docs pipelines, which currently
fall back to text parsing and can miss structured headings.

This slice adds an explicit documentation-term format override so suffixless
docs exports can exercise the real vocabulary-gap flow without renaming files.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add `--documentation-term-format` with `auto`, `text`, `json`, `jsonl`, and
   `csv` choices.
2. Thread the selected format through the existing documentation-term file
   loader.
3. Record the selected format in compact CLI result config.
4. Add focused CLI coverage for suffixless CSV exports and malformed override
   failures.
5. Update README guidance for documentation-term file formats.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Documentation-Term-Format.md` | Plan doc for this CLI ergonomics slice. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Adds and threads the documentation-term format override. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers suffixless CSV parsing and malformed override errors. |
| `extracted_content_pipeline/README.md` | Documents the explicit format override. |

## Mechanism

The parser gets a new option:

```bash
--documentation-term-format auto|text|json|jsonl|csv
```

`auto` preserves the current suffix-based behavior. Any explicit value bypasses
suffix detection and calls the matching existing parser for every
`--documentation-term-file`. Errors continue to use the same
`--documentation-term-file ...` messages because only parser selection changes.

## Intentional

- CLI-only behavior. FAQ generation, vocabulary-gap scoring, Markdown rendering,
  and rule-file parsing do not change.
- One format applies to all documentation-term files in the invocation. Mixed
  file formats still work with `auto` when suffixes are present.
- No new parser or field names are introduced; this only exposes parser
  selection for existing formats.

## Deferred

- Hosted UI upload controls for documentation terms and rule files remain a
  separate product slice.
- Per-file format overrides can be considered later if a real customer export
  requires mixed suffixless files in one command.
- Parked hardening considered: current `HARDENING.md` entries are landing-page
  repair items and do not touch this FAQ lane.

## Verification

- Focused documentation-term format pytest - passed, 3 tests.
- Full FAQ pytest for `tests/test_extracted_ticket_faq_markdown.py` - passed,
  133 tests.
- Py compile for affected Python files - passed.
- CLI demo with suffixless CSV documentation export - passed. The result config
  recorded `documentation_term_format=csv`, loaded 3 documentation terms, and
  produced `term_mapping_count=2`.
- Git whitespace check - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.
- Full extracted pipeline checks - passed, 1,758 tests, 1 existing torch/pynvml
  warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~90 |
| CLI parser/loader/config | ~40 |
| Tests | ~70 |
| README | ~5 |
| **Total** | ~205 |
