# PR-Support-Ticket-Resolution-Evidence-Contract

## Why this slice exists

Live support-ticket generation proved a source-contract gap: uploaded tickets can
show repeated customer questions, but question-only exports do not prove the
correct procedural answer. The current prompts tell the model not to invent
outcomes, but they do not give it a first-class signal for whether support
resolution text exists. That lets blog and landing-page FAQ copy fill missing
answer steps from genre convention.

This slice closes the source-level gap by making resolution evidence explicit in
the support-ticket package and generation contexts. It addresses the parked
hardening item "Support-ticket FAQ drafts can invent procedural answer steps
when tickets lack resolutions."

## Scope (this PR)

Ownership lane: content-ops/support-ticket-resolution-evidence

Slice phase: Production hardening

1. Detect explicit support-resolution fields while packaging uploaded
   support-ticket source material.
2. Expose resolution-evidence presence, count, and bounded examples in the
   package inputs and metadata.
3. Thread the evidence fields into landing-page `MarketingCampaign.context` and
   blog `data_context` without exposing full source rows.
4. Update blog and landing-page prompt contracts so question-only support-ticket
   data can draft questions and answer shells, but cannot invent concrete
   procedural steps.
5. Update the parked hardening entry to show this slice closes the contract
   portion and leaves no detector follow-up in this PR.

### Files touched

- `ATLAS-HARDENING.md`
- `atlas_brain/skills/digest/blog_post_generation.md`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/landing_page_input_contract.py`
- `extracted_content_pipeline/skills/digest/blog_post_generation.md`
- `extracted_content_pipeline/skills/digest/landing_page_generation.md`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Support-Ticket-Resolution-Evidence-Contract.md`
- `scripts/smoke_content_ops_support_ticket_package.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_landing_page_generation.py`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_smoke_content_ops_support_ticket_package.py`
- `tests/test_support_ticket_provider_landing_blog_execute.py`

## Mechanism

The support-ticket package now recognizes explicit top-level resolution fields
such as `resolution`, `resolution_text`, `solution`, `answer`,
`agent_response`, and `support_response`. When present, normalized source rows
carry a clipped `resolution_text`; the package exposes:

```python
support_ticket_resolution_evidence_present: bool
support_ticket_resolution_evidence_count: int
support_ticket_resolution_examples: list[dict[str, str]]
```

Landing pages receive these fields through the existing
`LANDING_PAGE_SUPPORT_TICKET_SOURCE_INPUT_KEYS` allowlist. Blog generation
receives the same fields through `_support_ticket_blog_data_context_from_inputs`.

The prompt contracts use the evidence signal as the generation rule:

- If resolution evidence is absent, write FAQ questions and review-needed answer
  shells, not concrete UI paths, menu names, timing, permissions, or exact
  procedural instructions.
- If resolution evidence is present, summarize only the supplied resolution
  examples and do not add missing product steps.

## Intentional

- This does not parse every possible conversation transcript. The first
  production contract is explicit top-level resolution fields because they are
  stable across CSV/JSON exports and avoid treating customer wording as support
  truth.
- This does not add a broad detector for invented procedural steps. The durable
  fix is to tell the generator what evidence exists before writing. Existing
  generated-content detectors remain regression backstops for outcome claims.
- The new examples are bounded and clipped so large CSVs do not leak full
  source rows into landing-page or blog prompts.

## Deferred

Parked hardening: none.

Future PR, if support platforms export threaded conversations instead of
resolution columns: add role-aware transcript parsing that extracts only
agent/staff resolution messages into the same `resolution_text` contract.

Future PR, if we want the broader evidence-or-placeholder contract: add explicit
date-window and measured-outcome evidence flags such as `has_dated_window`,
`has_measured_outcomes`, and bounded measured-outcome examples. This slice stays
focused on the immediate fabricated procedural-answer issue.

## Verification

- `python -m pytest tests/test_extracted_support_ticket_input_package.py tests/test_smoke_content_ops_support_ticket_package.py tests/test_support_ticket_provider_landing_blog_execute.py -q` - 36 passed.
- `python -m pytest tests/test_extracted_content_ops_execution.py::test_marketing_campaign_context_does_not_leak_unrelated_inputs tests/test_extracted_landing_page_generation.py::test_generate_threads_seo_geo_aeo_context_into_system_prompt_payload tests/test_extracted_landing_page_generation.py::test_build_draft_preserves_support_ticket_source_context_metadata -q` - 3 passed.
- `python -m pytest tests/test_extracted_landing_page_generation.py tests/test_extracted_content_ops_execution.py tests/test_extracted_blog_generation.py::test_support_ticket_blog_context_detection_markers_bite tests/test_extracted_blog_generation.py::test_support_ticket_blog_context_detection_rejects_false_positives -q` - 108 passed.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline` - passed; extracted files retained the resolution-evidence contract.
- `python -m pytest tests/test_extracted_support_ticket_input_package.py tests/test_smoke_content_ops_support_ticket_package.py tests/test_support_ticket_provider_landing_blog_execute.py tests/test_extracted_landing_page_generation.py tests/test_extracted_content_ops_execution.py tests/test_extracted_blog_generation.py::test_support_ticket_blog_context_detection_markers_bite tests/test_extracted_blog_generation.py::test_support_ticket_blog_context_detection_rejects_false_positives -q` - 144 passed after sync.
- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q` - 32 passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Resolution evidence packaging/context wiring | ~92 |
| Prompt contracts and hardening closeout | ~14 |
| Tests and smoke summary coverage | ~132 |
| Plan doc | ~116 |
| **Total** | **~354** |
