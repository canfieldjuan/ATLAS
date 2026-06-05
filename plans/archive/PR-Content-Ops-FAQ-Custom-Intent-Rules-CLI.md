# PR-Content-Ops-FAQ-Custom-Intent-Rules-CLI

## Why this slice exists

Intent-to-FAQ mapping is already partially implemented in the FAQ generator:
library and service callers can pass `intent_rules`, and tests prove custom
rules can group customer language into product-specific FAQ opportunities. The
CLI path still cannot pass those rules, so offline runs and customer-data demos
remain stuck on the built-in taxonomy.

This slice exposes custom intent rules in the CLI and result diagnostics without
changing the existing builder/service contract.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add repeatable CLI parsing for custom intent rules.
2. Prepend custom CLI intent rules before built-in defaults so customer-specific
   language wins deterministic topic matching.
3. Include configured custom intent rules in compact result JSON.
4. Add focused CLI tests for custom intent routing and invalid rule input.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Custom-Intent-Rules-CLI.md` | Plan doc for this CLI intent-rule slice. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Adds CLI custom intent-rule parsing, routing, and result config output. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers CLI custom intent routing and validation. |

## Mechanism

The CLI accepts repeated `--intent-rule` values shaped as
`topic=keyword,keyword`. Each rule must include a non-empty topic and at least
one non-empty keyword after case-insensitive dedupe. Parsed custom rules are
stored on the parsed args for result diagnostics, then prepended to
`DEFAULT_INTENT_RULES` before calling `build_ticket_faq_markdown`.

The builder/service `intent_rules` parameter remains an exact rule sequence for
existing host callers. This slice only adds a CLI convenience layer.

## Intentional

- Custom CLI rules take precedence over defaults so customer intent language is
  not hidden by generic built-in topics.
- The builder contract is unchanged; callers that already pass exact
  `intent_rules` keep the same behavior.
- Result JSON records only configured custom CLI rules, not the full default
  catalog, to keep diagnostics compact.

## Deferred

- Hosted UI fields for custom intent rules.
- JSON/CSV intent-rule imports for larger glossary-style rule sets.
- A future shared parser if multiple CLIs adopt the same custom-rule shape.

## Verification

- Focused custom intent-rule CLI pytest - passed, 7 tests.
- Full FAQ pytest for tests/test_extracted_ticket_faq_markdown.py - passed,
  96 tests.
- Py compile for affected Python files - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.
- Git whitespace check - passed.
- Local PR review against origin/main - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| CLI parser/result changes | ~45 |
| Tests | ~125 |
| **Total** | ~250 |
