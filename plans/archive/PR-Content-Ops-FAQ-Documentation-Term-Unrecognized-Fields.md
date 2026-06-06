# PR-Content-Ops-FAQ-Documentation-Term-Unrecognized-Fields

## Why this slice exists

PR-Content-Ops-FAQ-Documentation-Term-Structured-Files added CSV, JSON, and
JSONL documentation-term imports, but parseable structured exports with the
wrong field names still fall through to the generic "contains no terms" error.
That slows down operator triage because it does not say whether the file was
empty or the export used unrecognized columns or keys.

This slice adds small diagnostics for that failure path so larger docs exports
are easier to correct without changing the vocabulary-gap matching behavior.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Detect CSV documentation-term files that contain rows but none of the
   supported term columns.
2. Detect JSON and JSONL documentation-term files that contain structured rows
   or bundles but none of the supported term keys.
3. Raise clean CLI errors that list the expected term keys and avoid raw
   tracebacks.
4. Add focused CLI regression tests for unrecognized CSV, JSON, and JSONL
   structured files.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Documentation-Term-Unrecognized-Fields.md` | Plan doc for this diagnostic slice. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Adds structured term-file field diagnostics. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers unrecognized CSV, JSON, and JSONL term-file failures. |

## Mechanism

The CLI keeps the same suffix-based loaders. CSV now distinguishes "no rows" from
"rows present but no recognized term column." JSON and JSONL reuse the existing
structured traversal but record whether a mapping-shaped row or bundle was seen
without yielding a term.

When a structured file has parseable rows but no supported field, the loader
raises `SystemExit` with the expected field list. Empty files and plain text
files keep the existing generic "contains no terms" behavior.

## Intentional

- No new accepted field names in this slice. The goal is clearer diagnostics,
  not a broader import contract.
- No result JSON changes. The failure happens before a FAQ run result exists,
  and stderr is the operator surface for bad CLI input.
- The diagnostic stays CLI-local because documentation-term file parsing is
  currently a CLI feature.

## Deferred

- Format override flags for suffixless structured files.
- Hosted UI upload diagnostics for documentation term files.
- Rich docs export metadata such as URLs or section paths in result JSON.

## Verification

- Focused documentation-term-file FAQ CLI pytest - passed, 11 tests.
- Full FAQ pytest for `tests/test_extracted_ticket_faq_markdown.py` - passed,
  129 tests.
- Py compile for affected Python files - passed.
- Git whitespace check - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~75 |
| CLI diagnostics | ~65 |
| Tests | ~75 |
| **Total** | ~220 |
