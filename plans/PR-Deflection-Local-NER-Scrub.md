# PR-Deflection-Local-NER-Scrub

## Why this slice exists

Issue #1734 closed the highest-risk live leak with #1730: the report storage
gate now runs a structural scrub over the paid artifact and free Snapshot before
persistence. That scrub is still regex-only. The root cause for the remaining
privacy gap is that the single report-text scrub seam only detects patterned
PII classes: emails, phone numbers, labeled identifiers, and prior redaction
artifacts. Names, street addresses, and high-entropy opaque customer identifiers
can still pass through answer/steps prose because they do not always carry a
stable label or delimiter.

This PR fixes the root within safe scope by widening the local detector behind
the existing report-payload scrub seam. It does not add cloud DLP, does not
change report shape, does not change paid source-ID policy, and does not touch
retention/deletion. The heavier local-model option from #1734 remains
replaceable behind the same seam if we later decide a spaCy/Presidio dependency
is worth the operational cost.

The diff is over the 400 LOC soft cap because the review-fix pass added
same-class negative fixtures for the detector branches, source-link parity, and
buyer-facing near-misses; splitting those probes away would weaken the guard
that makes this privacy hardening safe to merge.

## Scope (this PR)

Ownership lane: content-ops/deflection-privacy
Slice phase: Production hardening

1. Add a dependency-free local entity-style detector to the existing
   deflection report scrub path for:
   - context-labeled person names;
   - street-address shapes;
   - high-entropy opaque customer/system IDs with an explicit ID cue or known
     customer/system prefix.
2. Keep the detector conservative with explicit near-miss coverage for product
   names, ordinary numerics, ISO-style references, and paid source-link values.
3. Prove the detector runs before both stored paid artifact and free Snapshot
   persistence through the existing scrubber and storage-gate tests.

### Files touched

- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Local-NER-Scrub.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_extracted_content_deflection_submit.py`

### Review Contract

Acceptance criteria:
- Names in explicit customer/requester/contact/user contexts redact to
  `[redacted-name]`.
- Street-address shaped text redacts to `[redacted-address]`.
- Opaque mixed letter/digit customer IDs with explicit ID context or a known
  customer/system prefix redact to `[redacted-identifier]`.
- Paid `source_id` / `source_ids` linkage values remain governed by #1730:
  non-PII source IDs are preserved in source-link fields, while PII-shaped
  source ID values are still scrubbed.
- Near-misses remain intact: product names, ordinary counts, prices, years,
  ISO-like standards, and low-entropy ticket labels.
- The persisted paid artifact and free Snapshot are both built from the widened
  scrubbed payload.

Affected surfaces:
- Deflection paid report artifact JSON stored by the control-surface report
  gate.
- Free Snapshot JSON derived from the same artifact before persistence.

Risk areas:
- Over-redacting product/company names in buyer-facing answers.
- Accidentally changing the #1730 paid source-ID traceability contract.
- Adding broad regexes that catch legitimate numeric metrics.

Reviewer rules triggered: R1, R2, R10, R14

## Mechanism

The current recursive payload scrubber already funnels free text through one
local text scrub seam after source-link protection and known identifier token
replacement. This PR extends that seam with three deterministic detectors:

1. Context-labeled person names are redacted only when introduced by person
   context such as customer, requester, contact, user, client, or name labels.
   This intentionally avoids broad title-case matching.
2. Street-address shapes redact street numbers plus common street suffixes and
   optional unit markers.
3. Opaque IDs redact mixed letter/digit tokens only when they carry explicit ID
   context or a known customer/system prefix, while allowing common low-risk
   standards, CVEs, SKUs, product versions, count phrases, and ticket labels.

The implementation stays pure and local inside
`extracted_content_pipeline/faq_deflection_report.py`. Tests exercise both the
new detection branches and the false-positive side of each branch.

## Intentional

- This is a conservative local detector, not a full NLP model. It catches clear
  high-confidence leak shapes without introducing a large runtime dependency or
  cloud processing surface.
- The slice does not try to redact every title-cased human name in prose. Broad
  title-case matching would over-redact product names and report headings.
- Bare CVEs, SKUs, product-version labels, ISO-like labels, and count phrases
  are intentionally preserved unless they carry explicit customer/system ID
  context.
- Retention/deletion for `content_ops_deflection_reports` stays deferred as the
  next separate #1734 item.
- Paid source IDs remain visible as buyer linkage metadata per #1730.

## Deferred

- #1734 retention/deletion slice for `content_ops_deflection_reports`.
- Optional heavier local NLP backend behind the same scrub seam if conservative
  deterministic detection is not enough after live artifact review.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_supported_pii_before_snapshot_projection tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_identifier_fields_markdown_and_keys tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_local_entity_shapes tests/test_content_ops_deflection_report.py::test_deflection_report_payload_preserves_local_entity_near_misses tests/test_extracted_content_deflection_submit.py::test_deflection_report_storage_gate_scrubs_supported_pii -q -
  5 passed.
- Command: python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_local_entity_shapes tests/test_content_ops_deflection_report.py::test_deflection_report_payload_preserves_local_entity_near_misses tests/test_extracted_content_deflection_submit.py::test_deflection_report_storage_gate_scrubs_supported_pii -q -
  3 passed.
- Command: python -m pytest tests/test_content_ops_deflection_report.py tests/test_extracted_content_deflection_submit.py -q -
  142 passed.
- Command: python -m py_compile extracted_content_pipeline/faq_deflection_report.py tests/test_content_ops_deflection_report.py tests/test_extracted_content_deflection_submit.py -
  passed.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-deflection-local-ner-scrub.md -
  passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/faq_deflection_report.py` | 228 |
| `plans/PR-Deflection-Local-NER-Scrub.md` | 139 |
| `tests/test_content_ops_deflection_report.py` | 89 |
| `tests/test_extracted_content_deflection_submit.py` | 21 |
| **Total** | **477** |
