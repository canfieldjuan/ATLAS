# FAQ Output Proof Ingestion Bridge

## Why this slice exists

PR #1109 made generated FAQ output reusable as source material, and PR #1113
bridged resolution-backed FAQ steps into the canonical `resolution_text`
contract. The remaining gap is proof: the current FAQ output smoke demonstrates
Markdown generation, but it does not exercise the real downstream bridge from
generated FAQ output back into support-ticket source rows.

This slice turns that smoke into a thin end-to-end validation of the bridge so a
future drift in FAQ result shape, FAQ-output adapter behavior, or support-ticket
resolution evidence packaging fails in one place.

## Scope (this PR)

Ownership lane: content-ops/faq-output-ingestion
Slice phase: Functional validation

1. Add resolution-backed source rows to the existing FAQ output proof fixture.
2. Extend the proof summary to adapt the generated FAQ result through
   `faq_output_to_source_rows(...)`.
3. Build a support-ticket input package from that adapted output and assert the
   canonical `support_ticket_resolution_evidence_*` fields are present.
4. Add focused tests for the success path and the new failure predicate.

### Files touched

- `plans/PR-FAQ-Output-Proof-Ingestion-Bridge.md`
- `scripts/smoke_content_ops_faq_output_proof.py`
- `tests/test_smoke_content_ops_faq_output_proof.py`

## Mechanism

The smoke already invokes `scripts/build_extracted_ticket_faq_markdown.py` and
reads the resulting compact diagnostic JSON. That CLI result intentionally omits
full item bodies, so this slice also writes a small
full FAQ result JSON artifact from the same source fixture using the library
`TicketFAQMarkdownResult.as_dict()` shape that the FAQ-output adapter consumes.
The proof then runs the generated full FAQ output through the downstream bridge:

```python
source_rows = faq_output_to_source_rows(result_payload)
package = build_support_ticket_input_package(result_payload, outputs=("blog_post",))
```

The proof records adapted FAQ source-row count, whether any adapted row carries
`resolution_text`, and whether the support-ticket package reports
`support_ticket_resolution_evidence_present` with a nonzero count. The failure
predicate fails closed if the generated output cannot be consumed through the
adapter or if resolution-backed answers stop reaching the package contract.

## Intentional

- This is validation, not a new product path. It does not add UI/API controls
  for selecting saved FAQ reports as input to other generators.
- The fixture stays small because the goal is contract composition, not scale.
  Existing FAQ scale and route stress docs cover large-row behavior separately.
- The proof uses the existing support-ticket package contract instead of adding
  a FAQ-specific resolution evidence check.

## Deferred

- UI/API selection for feeding saved FAQ reports into blog or landing generation
  remains deferred until that workflow is picked as a vertical slice.
- Hosted SaaS FAQ route proof remains blocked on deployed API URL, bearer token,
  and account id.
- Parked hardening: none.

## Verification

Ran locally:

- `python -m pytest tests/test_smoke_content_ops_faq_output_proof.py -q` - 4
  passed.
- `python scripts/smoke_content_ops_faq_output_proof.py --artifact-dir
  "$tmpdir"` - passed; summary reported 3 generated FAQ items, 3 adapted
  `faq_output` source rows, and 1 support-ticket resolution evidence example.
- `python -m py_compile scripts/smoke_content_ops_faq_output_proof.py
  tests/test_smoke_content_ops_faq_output_proof.py` - passed.
- `python scripts/audit_plan_doc.py plans/PR-FAQ-Output-Proof-Ingestion-Bridge.md`
  - passed.
- `python scripts/audit_plan_code_consistency.py
  plans/PR-FAQ-Output-Proof-Ingestion-Bridge.md` - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file
  /home/juan-canfield/Desktop/atlas-pr-bodies/faq-output-proof-ingestion-bridge.md`
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Smoke proof bridge | 116 |
| Focused tests | 33 |
| Plan doc | 96 |
| **Total** | **244** |
