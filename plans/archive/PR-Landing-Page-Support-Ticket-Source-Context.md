# PR: Landing Page Support Ticket Source Context

## Why this slice exists

Live support-ticket validation proved the blog-post path keeps CSV-derived
counts and clusters, but the landing-page path only used the general support
ticket theme and lost most source specificity. Landing pages need the same
source facts carried through the input contract so generated copy can use
ticket clusters and customer wording without exposing unrelated request fields.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: functional validation

1. Add explicit support-ticket source fact keys to the landing-page context
   allowlist.
2. Have the support-ticket input package populate row counts, question counts,
   truncation counts, top ticket clusters, and customer wording examples.
3. Preserve those support-ticket source fields in saved landing-page draft
   metadata for review/export inspection.
4. Update the landing-page prompt to use supplied support-ticket facts as
   service evidence.
5. Add focused tests for provider output, executor context threading, prompt
   payload visibility, and saved draft metadata.

### Files touched

- `plans/PR-Landing-Page-Support-Ticket-Source-Context.md`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `extracted_content_pipeline/landing_page_input_contract.py`
- `extracted_content_pipeline/landing_page_generation.py`
- `extracted_content_pipeline/skills/digest/landing_page_generation.md`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_extracted_support_ticket_input_provider.py`
- `tests/test_smoke_content_ops_live_generation.py`
- `tests/test_extracted_content_ops_live_execute_harness.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_landing_page_generation.py`

## Mechanism

The support-ticket provider derives compact facts from normalized ticket rows
and adds them to request inputs under explicit landing-page-safe keys. The
executor already builds `MarketingCampaign.context` from the landing-page
allowlist, so adding those keys there makes them visible to the prompt without
reopening the prior broad input leak. The landing generator then copies only
those allowlisted source keys into draft metadata for auditability.

Cluster labels are built only from genuine category/title signals. Rows without
a real label are counted as `uncategorized`, top-cluster overflow is counted as
`remaining`, and truncated rows are reported separately from examined rows that
were skipped for missing text. Source-period language only claims a date window
when all included rows have parseable dates.

## Intentional

- No FAQ generator changes. FAQ output remains owned by the FAQ session.
- No file-ingestion route changes. This slice starts from already-loaded
  support-ticket rows, matching this lane.
- No broad `source_material` leak into `MarketingCampaign.context`; only the
  summarized source facts are allowed.
- No nested `support_ticket_source_summary` context blob. Landing context keeps
  an explicit flat schema so request overrides cannot smuggle unrelated nested
  fields into the prompt or saved metadata.

## Deferred

- No live smoke rerun in this PR. The next slice should rerun the Haiku landing
  and blog smoke with `--export-saved-draft` to compare artifacts after this
  source-context fix lands.
- Parked hardening: none. `HARDENING.md` was scanned; current entries are FAQ
  scale and file-ingestion concurrency work outside this support-ticket
  provider validation slice.

## Verification

- Focused source-context tests:
  `pytest tests/test_extracted_support_ticket_input_package.py tests/test_extracted_content_ops_execution.py::test_marketing_campaign_context_does_not_leak_unrelated_inputs tests/test_extracted_landing_page_generation.py::test_generate_threads_seo_geo_aeo_context_into_system_prompt_payload tests/test_extracted_landing_page_generation.py::test_build_draft_preserves_support_ticket_source_context_metadata -q`
  - 20 passed.
- Relevant file suites:
  `pytest tests/test_extracted_support_ticket_input_package.py tests/test_extracted_content_ops_execution.py tests/test_extracted_landing_page_generation.py -q`
  - 107 passed.
- Expanded source-period and live-smoke suites:
  `pytest tests/test_extracted_support_ticket_input_package.py tests/test_extracted_support_ticket_input_provider.py tests/test_smoke_content_ops_live_generation.py tests/test_extracted_content_ops_live_execute_harness.py tests/test_extracted_content_ops_execution.py tests/test_extracted_landing_page_generation.py -q`
  - 147 passed.
- Py compile for changed Python files - passed.
- Git whitespace check - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~100 |
| Provider + contract + generation | ~90 |
| Prompt | ~10 |
| Tests | ~150 |
| **Total** | **~420** |
