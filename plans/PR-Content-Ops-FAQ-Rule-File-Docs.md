# PR-Content-Ops-FAQ-Rule-File-Docs

## Why this slice exists

PR-Content-Ops-FAQ-Rule-File-Import added reusable JSON rule files for custom
FAQ intent and vocabulary-gap rules, but the checked repo still lacks a
copyable example and a host-facing command that shows the schema in context.
Operators should not have to reverse-engineer the shape from CLI tests or the
prior plan doc before running customer glossary and intent mappings.

This slice documents the rule-file contract and adds a small checked example
that is exercised by the existing FAQ CLI test suite.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add a checked JSON example for FAQ custom intent and vocabulary-gap rules.
2. Document the `--rule-file` schema and precedence in the extracted package
   README.
3. Mirror the operator command in the host install runbook.
4. Add a CLI smoke test proving the checked example file changes generated
   intent routing and vocabulary-gap diagnostics.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Rule-File-Docs.md` | Plan doc for this docs/example slice. |
| `extracted_content_pipeline/examples/faq_custom_rules.json` | Checked JSON example for reusable FAQ rules. |
| `extracted_content_pipeline/README.md` | Documents rule-file schema, command usage, and precedence. |
| `extracted_content_pipeline/docs/host_install_runbook.md` | Adds the same operator-facing FAQ rule-file command. |
| `tests/test_extracted_ticket_faq_markdown.py` | Smoke test for the committed example file. |

## Mechanism

The checked example uses the existing JSON contract:

```json
{
  "intent_rules": [
    {"topic": "data freshness", "keywords": ["warehouse sync", "connector lag"]}
  ],
  "vocabulary_gap_rules": [
    ["SSO", "single sign-on"]
  ]
}
```

The CLI smoke test loads that file through `--rule-file`, supplies a temporary
ticket row that contains the example terms, and writes compact result JSON. The
assertions verify that the file path appears in config diagnostics, the custom
intent topic is used for the generated FAQ item, and the custom vocabulary rule
produces a term mapping against an existing documentation term.

## Intentional

- No parser or generator behavior changes. This slice only makes the existing
  rule-file capability easier to run and harder to regress.
- The example is intentionally short so operators can copy and adapt it without
  deleting demo-specific noise.
- Explicit CLI rules still take precedence over file rules; this slice documents
  the behavior rather than changing it.
- Rule-file delimiter restrictions are documented here because the parser
  already rejects those values with a clean operator error.

## Deferred

- CSV rule-file import remains deferred until the JSON contract has more use.
- Hosted UI upload/entry fields for custom FAQ rules remain a later product
  slice.
- Broader customer glossary templates can be added once a real customer import
  path needs them.

## Verification

- Focused checked-rule-file FAQ CLI pytest - passed, 1 test.
- Full FAQ pytest for `tests/test_extracted_ticket_faq_markdown.py` - passed,
  116 tests.
- Py compile for the affected Python test file - passed.
- Git whitespace check - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.
- Local PR review against origin/main - passed with `--allow-dirty` because the
  worktree already had unrelated untracked user audit notes outside this PR.
- Reviewer NIT docs update: git whitespace check passed; local PR review against
  origin/main passed with the same unrelated untracked user audit note present.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~75 |
| JSON example | ~12 |
| README docs | ~35 |
| Host runbook docs | ~20 |
| CLI smoke test | ~35 |
| **Total** | ~177 |
