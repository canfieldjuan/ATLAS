# PR-Content-Ops-FAQ-Rule-File-Import

## Why this slice exists

The FAQ CLI can now accept custom vocabulary-gap rules and custom intent rules,
but larger SMB and mid-market runs should not require long command lines. Teams
need a small rule-file path for reusable product glossary and intent mappings
that can travel with demos, offline customer samples, and repeatable validation
runs.

This slice adds JSON rule-file import for the existing CLI custom-rule seams.
It is over the 400 LOC soft budget after review because the delimiter
round-trip guard and operator-error branches need to ship with fixture coverage
in the same slice; splitting those tests out would recreate the under-tested
error-path pattern this family of PRs has been correcting.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add a repeatable `--rule-file` CLI flag for JSON rule files.
2. Accept `intent_rules` and `vocabulary_gap_rules` from the JSON file.
3. Let explicit CLI flags take precedence over file rules for deterministic
   first-match intent routing.
4. Include configured rule-file paths in compact result JSON.
5. Add focused CLI tests for rule-file routing, precedence, and validation.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Rule-File-Import.md` | Plan doc for this rule-file slice. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Adds JSON rule-file parsing and combines file rules with explicit CLI rules. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers rule-file import, precedence, and invalid payload handling. |

## Mechanism

The CLI reads each `--rule-file` as a JSON object with optional
`intent_rules` and `vocabulary_gap_rules` arrays. Intent rules use objects shaped
as `{"topic": "...", "keywords": ["...", "..."]}`. Vocabulary-gap rules use
arrays of aliases such as `["SSO", "single sign-on"]`.

File rules are parsed through the same validation helpers as the existing CLI
flags. Explicit CLI rules are placed before file rules so command-line overrides
win in first-match intent routing and first-listed vocabulary mappings.

## Intentional

- JSON only in this slice. CSV import is deferred until the rule-file contract
  proves useful.
- Explicit CLI flags win over file rules because they are the most local
  operator intent.
- JSON rule-file values must be strings and cannot contain the CLI delimiters
  used for the corresponding rule type; the CLI rejects those cases rather than
  silently re-splitting a rule into the wrong shape.
- The builder/service contracts remain unchanged; this is a CLI ergonomics
  layer over existing rule inputs.
- Local review's cross-layer caller hints for generic script helper names such
  as `_parse_args`, `main`, and `_result_payload` were inspected as unrelated
  same-name helpers in other scripts; this slice does not change shared library
  behavior.

## Deferred

- CSV rule-file import.
- Hosted UI upload/entry fields for custom FAQ rules.
- A shared custom-rule parser module if another CLI adopts this shape.

## Verification

- Focused FAQ rule-file/custom-rule CLI pytest - passed, 33 tests.
- Full FAQ pytest for tests/test_extracted_ticket_faq_markdown.py - passed,
  115 tests.
- Py compile for affected Python files - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.
- Git whitespace check - passed.
- Local PR review against origin/main - passed; advisory cross-layer caller
  hints inspected, no actionable external caller impact.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| CLI parser/result changes | ~155 |
| Tests | ~220 |
| **Total** | ~455 |
