# PR-Content-Ops-Helpdesk-Source-Aliases

## Why this slice exists

The Content Ops ingestion path now accepts source-row JSON/JSONL/CSV and ships packaged support-ticket examples, but common help desk exports still require hosts to rename fields before import. This slice closes that friction by teaching the existing source adapter a small set of provider-style support aliases while preserving the current opportunity contract.

## Scope (this PR)

1. Extend the existing source-row alias tables for common help desk fields.
2. Preserve the existing normalized-key lookup and source-row conversion flow.
3. Add focused regression tests for help desk CSV-style field names.

### Files touched

- `extracted_content_pipeline/campaign_source_adapters.py`
- `tests/test_extracted_campaign_source_adapters.py`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Helpdesk-Source-Aliases.md`

## Mechanism

The adapter already normalizes source field labels by snake-case and compact comparisons. This PR adds aliases to the existing tuples rather than adding a new parser:

- ticket/case/conversation number aliases map to source ids and source-type precedence.
- requester/customer/user aliases map to contact fields.
- organization/account aliases map to company fields.
- issue/latest-comment/customer-message aliases map to evidence text.
- ticket/case title aliases map to source title and are stripped from buyer/contact fields like existing `subject` and `title`.

## Intentional

- No new ingestion command. The existing file loader and Postgres smoke keep using the same adapter.
- No provider-specific dependency or schema module. These are generic help desk export aliases.
- No changes to function signatures or output shape.

## Deferred

- Provider-specific mappings for tools like Zendesk, Intercom, Salesforce Service Cloud, or Freshdesk can land later if real exports need fields beyond this generic set.
- Multi-message support transcripts still use the existing `comments` / `messages` thread path.

## Verification

- Focused source-adapter pytest suite - 60 passed.
- Source adapter and test py_compile - passed.
- git diff whitespace check - passed.
- Extracted content pipeline manifest validation - passed.
- Extracted content pipeline Atlas reasoning import guard - passed.
- Extracted standalone audit - passed.
- Extracted content pipeline ASCII check - passed.
- Local PR review - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Source adapter aliases | ~85 |
| Tests | ~72 |
| Plan + coordination | ~58 |
| **Total** | **~215** |

This is below the 400 LOC review budget.
