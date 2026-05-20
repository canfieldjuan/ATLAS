# Content Ops FAQ Complaint Narrative Questions

## Why this slice exists

The CFPB source-row path exposed a generic FAQ issue: complaint-shaped rows are
often first-person narratives, not literal question sentences. The FAQ output
check for customer vocabulary should not fail just because a support-ticket-like
source says "I was charged..." instead of "Why was I charged...?"

The fix belongs in the generic FAQ question extractor and billing intent rules,
not in the CFPB exporter or a CFPB-specific renderer branch.

## Scope (this PR)

1. Expand billing intent keywords so fees, charged, loans, statements, leases,
   interest, and disputes group with billing/payment complaints.
2. Derive human FAQ questions from first-person complaint narratives such as
   "I was ...", "my ...", "we were ...", and "our ...".
3. Add regression coverage using CFPB-shaped source rows through generic FAQ
   fields only.
4. Replace the merged FAQ financial-action coordination row with this slice.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Complaint-Narrative-Questions.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Claim this FAQ narrative-quality slice. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Align billing intent keywords and complaint narrative question extraction. |
| `tests/test_extracted_ticket_faq_markdown.py` | Regression coverage for complaint-narrative question extraction. |

## Mechanism

The FAQ builder already extracts direct question sentences and converts common
first-person request starts such as "I need to ..." into "How do I ..."
questions. This PR extends that same generic extractor to complaint starts:
"I was ...", "I got ...", "I received ...", "I paid ...", "my ...", "we were
...", "we got ...", "we received ...", "we paid ...", and "our ...".

Those become short "What should I/we do if ..." FAQ questions, preserving the
customer's vocabulary while making the heading readable.

The billing intent rule is also aligned with the financial action rule so a
fee or charge complaint does not split away from billing/payment topics.

## Intentional

- No CFPB-specific branch in the FAQ renderer.
- No LLM generation. FAQ output stays deterministic and easy to audit.
- No legal or financial advice; the existing action language still tells users
  to compare records and contact support when the issue remains unresolved.

## Deferred

- The CFPB-to-FAQ live smoke command is the next slice after this generic FAQ
  behavior lands.
- Product-specific action templates remain deferred until a host supplies real
  product docs or real customer ticket exports.

## Verification

- pytest tests/test_extracted_ticket_faq_markdown.py - 45 passed.
- python -m py_compile extracted_content_pipeline/ticket_faq_markdown.py tests/test_extracted_ticket_faq_markdown.py - passed.
- python scripts/build_extracted_ticket_faq_markdown.py extracted_content_pipeline/examples/support_ticket_sources.csv --source-format csv --support-contact 1-800-555-0100 --require-output-checks - passed.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- Local PR review: pending.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Complaint-Narrative-Questions.md` | 75 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 35 |
| `tests/test_extracted_ticket_faq_markdown.py` | 45 |
| **Total** | **159** |
