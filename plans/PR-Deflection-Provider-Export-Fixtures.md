# PR-Deflection-Provider-Export-Fixtures
## Why this slice exists
#1384 still needs provider-export-shaped validation for the help desks named in
the upload UI. This slice adds sanitized contract fixtures and proves CSV
parsing -> support-ticket packaging -> deflection diagnostics/report without
claiming they are operator-supplied live exports.
## Scope (this PR)
Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Functional validation
1. Add Zendesk/Freshdesk/Help Scout/Intercom full-thread fixtures plus a
   Zendesk index-only negative fixture.
2. Expand only provider-specific customer-comment and agent/admin/support-reply aliases.
3. Prove real CSV loading, package diagnostics, FAQ/report output, submit metadata,
   and rejection of generic response/SLA/auto-ack aliases.
### Review Contract
- Acceptance criteria: committed CSV bytes use the real parser/package/report path;
  full-thread fixtures produce resolution diagnostics and a `resolution_evidence`
  item; HTML/entities normalize; index-only stays gap-list-only; generic
  response/SLA/auto-ack columns are rejected; plan/body do not overclaim live exports.
- Affected surfaces: support-ticket aliases, fixtures, extracted smoke tests,
  submit diagnostics.
- Risks: fixture realism, alias false positives, overclaiming, CI enrollment.
- Reviewer rules triggered: R1, R2, R5, R10, R12, R13.
### Files touched

- `extracted_content_pipeline/campaign_source_adapters.py`
- `extracted_content_pipeline/examples/support_ticket_provider_exports/README.md`
- `extracted_content_pipeline/examples/support_ticket_provider_exports/freshdesk_full_thread_export.csv`
- `extracted_content_pipeline/examples/support_ticket_provider_exports/help_scout_full_thread_export.csv`
- `extracted_content_pipeline/examples/support_ticket_provider_exports/intercom_conversation_export.csv`
- `extracted_content_pipeline/examples/support_ticket_provider_exports/zendesk_full_thread_export.csv`
- `extracted_content_pipeline/examples/support_ticket_provider_exports/zendesk_ticket_index_only.csv`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Provider-Export-Fixtures.md`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_smoke_content_ops_support_ticket_package.py`

## Mechanism
Fixtures live under `extracted_content_pipeline/examples/` and load through the
real CSV adapter. Alias expansion is limited to customer text fields and explicit
agent/support/admin reply fields. Generic `reply_text`, `first_response`, and
`last_response` are rejected in both package and markdown fallback tests so
SLA/auto-ack metadata cannot become publishable resolution evidence. The submit
test posts fixture bytes and checks pre-payment metadata plus no raw email leak.
## Intentional
- Sanitized provider-shaped fixtures are not a substitute for real operator exports.
- No live help-desk APIs are called; the product remains upload-based.
- Generic response/comment fields are rejected because they can be SLA metadata
  or auto-acks rather than actual agent resolutions.
## Deferred
- True sanitized operator exports remain the stronger #1384 follow-up.
- A richer inspect/pre-payment UI can render these diagnostics later.
Parked hardening: none.
## Verification
- python -m pytest tests/test_extracted_support_ticket_input_package.py tests/test_extracted_ticket_faq_markdown.py tests/test_smoke_content_ops_support_ticket_package.py tests/test_extracted_content_deflection_submit.py -q -- 239 passed in 4.30s.
- python -m py_compile extracted_content_pipeline/support_ticket_input_package.py extracted_content_pipeline/ticket_faq_markdown.py tests/test_extracted_support_ticket_input_package.py tests/test_extracted_ticket_faq_markdown.py -- passed.
- bash scripts/validate_extracted_content_pipeline.sh -- passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- clean.
- python scripts/audit_extracted_standalone.py --fail-on-debt -- Atlas runtime import findings: 0.
- bash scripts/check_ascii_python.sh -- passed.
- bash scripts/run_extracted_pipeline_checks.sh -- 3584 passed, 10 skipped, 1 warning in 56.55s.
## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/campaign_source_adapters.py` | 4 |
| `extracted_content_pipeline/examples/support_ticket_provider_exports/README.md` | 20 |
| `extracted_content_pipeline/examples/support_ticket_provider_exports/freshdesk_full_thread_export.csv` | 4 |
| `extracted_content_pipeline/examples/support_ticket_provider_exports/help_scout_full_thread_export.csv` | 4 |
| `extracted_content_pipeline/examples/support_ticket_provider_exports/intercom_conversation_export.csv` | 4 |
| `extracted_content_pipeline/examples/support_ticket_provider_exports/zendesk_full_thread_export.csv` | 4 |
| `extracted_content_pipeline/examples/support_ticket_provider_exports/zendesk_ticket_index_only.csv` | 3 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 15 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 9 |
| `plans/PR-Deflection-Provider-Export-Fixtures.md` | 83 |
| `tests/test_extracted_content_deflection_submit.py` | 52 |
| `tests/test_extracted_support_ticket_input_package.py` | 30 |
| `tests/test_extracted_ticket_faq_markdown.py` | 39 |
| `tests/test_smoke_content_ops_support_ticket_package.py` | 122 |
| **Total** | **393** |
