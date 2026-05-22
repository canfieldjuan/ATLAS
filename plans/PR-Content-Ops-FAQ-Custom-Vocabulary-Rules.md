# PR-Content-Ops-FAQ-Custom-Vocabulary-Rules

## Why this slice exists

Vocabulary-gap detection can now rank mismatches between customer wording and
documentation wording, but the alias rules are still limited to the generator's
built-in generic SaaS and support terms. SMB and mid-market teams need to bring
their own glossary terms, acronyms, product names, and synonym sets before the
FAQ wedge feels production-ready for their actual support language.

This slice adds a small custom vocabulary-rule input path while preserving the
current default behavior.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add configurable vocabulary-gap alias groups to the FAQ builder and service.
2. Keep the built-in vocabulary-gap rules active, then append caller-provided
   rules for product-specific glossary terms.
3. Add a repeatable CLI flag for custom vocabulary-gap rules.
4. Include the configured custom rules in compact CLI result diagnostics.
5. Add focused tests for builder, service, CLI diagnostics, and CLI validation.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Custom-Vocabulary-Rules.md` | Plan doc for this custom vocabulary-rule slice. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Threads custom vocabulary-gap rules through builder/service and mapping detection. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Adds CLI parsing and result config output for custom vocabulary-gap rules. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers custom rule generation, service config, CLI diagnostics, and invalid rule input. |

## Mechanism

The builder accepts `vocabulary_gap_rules` as a sequence of alias groups, where
each group must contain at least two cleaned aliases. The generator prepends
custom groups before the built-in defaults so team glossary terms win the
three-mapping display cap. Vocabulary-gap mapping then walks the resolved rules
exactly as before: it looks for customer terms in source text, skips terms
already present in documentation headings, and suggests the matching
documentation term from the same alias group.

The CLI exposes this as repeated `--vocabulary-gap-rule` values using a
comma-separated alias group such as `SSO,single sign-on`. Invalid one-term rules
and case-only duplicates fail before generation so a malformed glossary cannot
silently produce a green empty diagnostic or leak a traceback.

## Intentional

- Custom rules are evaluated before defaults so customer glossary terms are not
  starved by generic default mappings under the display cap.
- Built-in defaults remain active after custom rules so existing callers keep
  today's vocabulary-gap behavior when they do not pass custom rules.
- Rule parsing is deterministic and local; this does not introduce an LLM,
  runtime dependency, or external glossary fetch.
- Result JSON records only configured custom rules, not the built-in defaults,
  to keep diagnostics compact.

## Deferred

- A later hosted upload slice can expose custom glossary entry fields in the
  Content Ops Station UI.
- A later glossary import slice can support JSON or CSV rule files if repeated
  CLI flags become too awkward for larger teams.
- A later audit slice can add broader cross-generator glossary contract checks
  if other content generators adopt the same rule shape.

## Verification

- Focused custom vocabulary-rule pytest for builder, service, validation, and
  CLI behavior - passed, 9 tests.
- Full FAQ pytest for tests/test_extracted_ticket_faq_markdown.py - passed,
  83 tests.
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
| Plan doc | ~85 |
| Generator, CLI, and tests | ~285 |
| **Total** | ~370 |
