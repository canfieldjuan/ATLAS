# PR-Content-Ops-FAQ-Vocabulary-Gap-Term-Mapping

## Why this slice exists

The FAQ product wedge still needs first-pass Vocabulary Gap Detection. The
generator can rank FAQ opportunities from customer language, but it does not yet
show when customers use one term and the existing documentation uses another.

That gap matters for SMB and mid-market teams because zero-result searches and
support tickets often come from wording mismatch, not missing functionality.
This slice adds deterministic term-mapping suggestions to the FAQ artifact so a
generated FAQ can say which customer terms should be added as synonyms or
alternate headings.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add optional documentation-term input to the FAQ service, core builder, and
   CLI.
2. Extract documentation terms from explicit configuration and documentation-like
   source rows without using those rows as FAQ evidence by default.
3. Add per-FAQ-item vocabulary-gap suggestions when customer evidence uses a
   known customer term and documentation terms use a different known term.
4. Render compact "Vocabulary gaps" Markdown bullets and expose compact counts
   in the CLI result JSON.
5. Add focused tests for explicit documentation terms, documentation source
   rows, CLI result diagnostics, and no-gap behavior.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Vocabulary-Gap-Term-Mapping.md` | Plan doc for this vocabulary-gap slice. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Adds documentation-term input, term mapping detection, item metadata, and Markdown rendering. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Adds repeatable CLI documentation-term input and compact term-mapping diagnostics. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers generator, service, and CLI vocabulary-gap behavior. |

## Mechanism

The core builder receives optional documentation terms. It also scans
documentation-like rows already present in normalized source opportunities, such
as document/help-article rows, for source titles and evidence text. Those rows
continue to be excluded from default FAQ evidence because the existing
source-type filter still controls which rows can produce FAQ items.

For each generated FAQ item, the builder compares customer evidence against a
small deterministic synonym-rule table. When customer evidence contains a
customer term and the documentation terms contain a different term from the same
rule, the item gets a compact term-mapping suggestion. The Markdown renderer
prints these under "Vocabulary gaps", and the CLI result JSON records counts and
compact mapping summaries without duplicating the full Markdown body.

## Intentional

- No LLM call, embedding dependency, search index dependency, or hosted UI
  change.
- No new output check gate; missing documentation terms should not make FAQ
  generation fail.
- Documentation rows can inform term mapping, but they do not become FAQ
  evidence unless callers explicitly change the source-type filter.
- The synonym table is deliberately small and deterministic. It covers common
  support/doc mismatches first and can grow in later slices.

## Deferred

- A later hosted upload slice can expose documentation-term inputs in the UI.
- A later search-log slice can import zero-result search terms directly from a
  site-search export and rank mappings by zero-result frequency.
- A later semantic slice can replace or supplement deterministic synonym rules
  with embeddings or a customer-provided glossary.

## Verification

- Focused vocabulary-gap pytest for builder, service, and CLI behavior - passed, 6 tests.
- Full FAQ pytest for tests/test_extracted_ticket_faq_markdown.py - passed, 74 tests.
- Py compile for affected Python files - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.
- Git whitespace check - passed.
- Local PR review wrapper against origin/main - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 97 |
| FAQ generator | 184 |
| CLI result diagnostics | 39 |
| Tests | 147 |
| **Total** | **~470** |

This intentionally exceeds the soft 400 LOC target because the slice needs the
generator behavior, service/CLI entry points, and tests together to make the
feature usable and reviewable.
