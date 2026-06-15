# PR-Deflection-Report-SEO-Target-Cap

## Why this slice exists

Issue #1579 flags the paid deflection report as too dense, with two remaining
high-volume drivers after #1584 collapsed the repeated per-question passes:
the uncapped Help-Desk SEO Targeting List and the still-complete per-question
evidence blocks.

This slice takes the safe half first. The SEO list is an index, not the
complete-evidence promise. Capping that index reduces browser/PDF bulk without
dropping ranked questions, source IDs, or evidence quotes from the paid report.
Evidence caps require a real complete-evidence export/download path before they
are safe; this PR deliberately does not pretend the internal persisted
`faq_result` is a buyer-facing export.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Product polish

1. Cap `## Your Help-Desk SEO Targeting List` to a deterministic top-N list.
2. Add buyer-facing copy that says how many additional source-backed phrases
   were omitted from the index for readability.
3. Preserve every ranked question, every per-question vocabulary mapping, and
   every complete evidence/source-ID block in `## Question Details and
   Evidence`.
4. Add regression coverage for capped and uncapped SEO-list behavior, including
   boundary cases where the list length equals the cap.

### Files touched

- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Report-SEO-Target-Cap.md`
- `tests/test_content_ops_deflection_report.py`

### Review Contract

Acceptance criteria:

- The SEO targeting section renders at most the configured cap of phrases.
- When phrases are omitted, the report states the omitted count and makes clear
  the cap is only for the SEO index.
- When phrases are at or below the cap, the report does not show an omitted
  count.
- Ranked questions remain uncapped.
- Complete source IDs and evidence quotes remain uncapped inside
  `Question Details and Evidence`.
- The cap is deterministic and based on the existing `_customer_phrase_list`
  order, not random sampling or LLM ranking.

Affected surfaces:

- `extracted_content_pipeline/faq_deflection_report.py`
- `tests/test_content_ops_deflection_report.py`
- committed report contract/proof fixtures that assert the SEO section text

Risk areas:

- Accidentally changing the paid-report completeness promise.
- Hiding full question/evidence data instead of only shortening the SEO index.
- Off-by-one cap behavior at exactly-N phrases.
- Stale proof fixtures that still imply the SEO list is complete.

Reviewer rules triggered: R1, R2, R10, R13, R14.

## Mechanism

Add a small renderer constant for the SEO index cap and apply it only in
`_help_desk_seo_targeting_section`. The section continues to source phrases
from `_customer_phrase_list(items)`, preserving the existing deterministic
question/term-mapping order, then renders only the first N phrases.

If the full phrase list is longer than N, the section appends a short note:
the list is capped for readability and the remaining count is still represented
in the per-question detail blocks. This avoids claiming the SEO index is
complete while keeping the full paid report data intact.

No evidence rendering code is changed in this slice.

## Intentional

- This PR does not cap evidence quotes or source IDs. Evidence capping is only
  safe once the complete evidence is exposed through a real buyer-facing export
  or download path; the persisted artifact JSON alone is not a product surface.
- This PR does not add browser/PDF anchor navigation. The hosted result page
  still renders Markdown inside a `<pre>`, and PDF navigation is a separate
  renderer/delivery concern.
- The cap is fixed in code for now, not a config field. This is a deterministic
  report-shape constant, not tenant-specific runtime behavior.

## Deferred

- #1579 follow-up: add a real complete-evidence export/download path, then cap
  inline evidence quotes/source IDs without breaking the buyer-facing
  completeness promise.
- #1579 follow-up: add true hosted-report navigation once the portfolio result
  page renders structured Markdown/HTML instead of escaped `<pre>` text.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_content_ops_deflection_report.py -q
  -- 43 passed.
- Command: python -m pytest tests/test_content_ops_deflection_report.py tests/test_content_ops_faq_report_contract_docs.py -q
  -- 48 passed.
- Command: python -m pytest tests/test_atlas_billing_content_ops_deflection_paid_flow.py tests/test_extracted_content_ops_execution.py tests/test_content_ops_deflection_resolution_live_proof.py tests/test_atlas_content_ops_deflection_delivery.py tests/test_deflection_pdf_renderer.py -q
  -- 98 passed, 1 warning.
- Command: python -m py_compile extracted_content_pipeline/faq_deflection_report.py tests/test_content_ops_deflection_report.py
  -- passed.
- Command: git diff --check -- passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh -- passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  -- passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt -- passed.
- Command: bash scripts/check_ascii_python.sh -- passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh -- 4300 passed, 10 skipped,
  1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_report_contract.md` | 5 |
| `extracted_content_pipeline/faq_deflection_report.py` | 18 |
| `plans/PR-Deflection-Report-SEO-Target-Cap.md` | 130 |
| `tests/test_content_ops_deflection_report.py` | 90 |
| **Total** | **243** |
