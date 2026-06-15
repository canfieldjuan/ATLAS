# PR-Deflection-Zendesk-Header-Admission

## Why this slice exists

GitHub issue #1457 is still open after the Zendesk product-proof corpus and
eval work: CSV exports can carry the real customer wording in provider-specific
comment/history columns while `Description` is blank. The current source-row
adapter preserves those columns, but the canonical text lookup does not admit
several Zendesk public-comment aliases as customer-visible text. That makes
product-shaped ticket exports look empty downstream and can silently drop rows
from the deflection proof path.

The same issue also names the privacy boundary: Zendesk internal/private notes
are not customer-visible and must not become customer wording. This slice fixes
the upstream parser admission boundary instead of patching report output after
the source text has already been lost.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser-admission
Slice phase: Production hardening

1. Admit scalar Zendesk public-comment/history aliases as source text when
   provider exports leave `Description` blank.
2. Keep internal/private note aliases out of customer-visible source text and
   out of normalized campaign opportunity context.
3. Prove the behavior at the source adapter boundary and through the
   support-ticket input package consumer path.
4. Filter structured comment/history objects with `public: false` so private
   Zendesk history cannot leak through newly admitted aliases.
5. Scrub `public: false` objects from preserved thread/comment fields before
   normalized campaign opportunity context can be serialized into LLM prompts.

### Files touched

- `extracted_content_pipeline/campaign_customer_data.py`
- `extracted_content_pipeline/campaign_source_adapters.py`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Deflection-Zendesk-Header-Admission.md`
- `tests/test_extracted_campaign_source_adapters.py`
- `tests/test_smoke_content_ops_support_ticket_package.py`

### Review Contract

- Acceptance criteria:
  - [ ] A Zendesk-style CSV row with `Ticket ID`, `Subject`, blank
        `Description`, and public `Ticket Comments` or `Ticket History` becomes
        a usable support-ticket source row.
  - [ ] A row whose only body-like content is `Internal Notes` or
        `Private Notes` still warns as missing source text and is not admitted
        as customer wording.
  - [ ] The support-ticket input package includes public scalar comments in the
        ticket text while excluding private/internal note text.
  - [ ] Structured comment/history lists skip `public: false` objects in both
        source-adapter evidence and support-ticket package customer wording.
  - [ ] Preserved thread/comment fields in normalized campaign opportunity
        context do not retain `public: false` objects.
  - [ ] Existing CSV delimiter/encoding behavior remains covered by the
        existing enrolled source-adapter tests.
- Affected surfaces: extracted source ingestion, support-ticket source shaping,
  tests.
- Risk areas: backcompat, privacy, data truthfulness.
- Reviewer rules triggered: R1, R2, R10, R12, R13, R14.

## Mechanism

Extend the source adapter's public thread alias list with Zendesk
customer-visible public comment/history names such as `ticket_comments`,
`public_comments`, `comment_body`, `ticket_history`, and `conversation`. The
existing provider-style key lookup already normalizes spaces and casing, and
the thread-text helper already handles scalar strings as well as structured
comment sequences, so the fix belongs in the alias table rather than in CSV
parsing.

Do not add private/internal note aliases to that public text list. Instead,
filter explicit private/internal note columns while constructing normalized
campaign opportunity context so those raw columns do not travel downstream as
prompt-visible metadata. Add explicit regression coverage that rows containing
only private/internal note fields are rejected as missing customer-visible text.
Then add a consumer-path smoke case that builds the support-ticket input
package from a Zendesk-shaped CSV and checks the resulting customer wording
includes public comments/history but not internal notes.

For structured comment/history values, mirror the dedicated Zendesk thread
importer's privacy rule: any mapping with `public` set to `False` is ignored
before text is extracted. Public objects and objects without a `public` flag
remain admissible, preserving the scalar/public-comment fix while closing the
private-history leak found in review.

Apply the same rule when preserving raw thread/comment fields on the normalized
campaign opportunity. Thread fields such as `ticket_history` and `comments`
remain available for context, but private objects are removed before the
opportunity dict can be serialized into campaign/report/sales-brief prompts.

## Intentional

- This does not build the full #1467 admission report. #1457's immediate P1
  failure is customer-visible text being lost or misclassified; a richer
  mapped/unmapped-column diagnostic remains a separate parser-admission slice.
- This keeps the fix in the shared source adapter rather than special-casing
  ticket FAQ rendering. Once the source row has correct text, the existing
  support-ticket package and report layers can consume it without downstream
  patching.

## Deferred

- Full mapped/unmapped CSV column admission diagnostics from #1467.
- Streaming upload memory hardening from #1458.

Parked hardening: none.

## Verification

- .venv/bin/python -m pytest tests/test_extracted_campaign_source_adapters.py tests/test_smoke_content_ops_support_ticket_package.py
  - 136 passed.
- bash scripts/run_extracted_pipeline_checks.sh
  - 4273 passed, 10 skipped.
- bash scripts/validate_extracted_content_pipeline.sh
  - Passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt
  - Passed.
- bash scripts/check_ascii_python.sh
  - Passed.
- bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline
  - Completed; intended extracted file edits remained.
- Pending before push:
  - bash scripts/local_pr_review.sh

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/campaign_customer_data.py` | 8 |
| `extracted_content_pipeline/campaign_source_adapters.py` | 84 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 49 |
| `plans/PR-Deflection-Zendesk-Header-Admission.md` | 141 |
| `tests/test_extracted_campaign_source_adapters.py` | 86 |
| `tests/test_smoke_content_ops_support_ticket_package.py` | 68 |
| **Total** | **436** |
