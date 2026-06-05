## Why this slice exists

The FAQ deflection draft-quality arc now has the safety pieces in place:
question/resolution scoping, cleaned step scaffolding, and the fail-closed
`resolution_evidence_scoped` gate. The remaining product polish gap is the
answer summary for proven drafts: it still reads like internal provenance
plumbing ("Verified resolution evidence ... supports the draft answer") instead
of help-center copy a buyer could publish after review.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection-backend

Slice phase: Product polish

1. For `resolution_evidence` items, render the answer summary as direct
   customer-facing help-center prose derived from `resolution_text`.
2. Preserve fail-closed doctrine: unproven items keep the existing review-needed
   answer, and proven copy still comes only from uploaded resolution evidence.
3. Normalize proven resolution steps into complete sentences without changing
   grouping, ranking, paywall, snapshot, or evidence-scope behavior.
4. Render the paid deflection report's proven-answer section from the polished
   item answer instead of generic evidence-proven scaffolding.
5. Update focused tests and frontend example JSON that lock the old scaffolding
   out and prove the prose remains evidence-derived.

### Files touched

- `extracted_content_pipeline/ticket_faq_markdown.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_extracted_ticket_faq_macro_writeback.py`
- `plans/PR-Deflection-Help-Center-Prose.md`

## Mechanism

The item builder already passes `resolution_texts` into step generation. This
slice threads those same texts into the answer-summary builder. When an item is
`resolution_evidence`, the summary is built from the first one or two cleaned
resolution excerpts:

```python
To resolve this, <resolution-derived instruction>. Then <next instruction>.
```

The helper only reformats the uploaded resolution wording into complete
sentences. It does not introduce product facts, fallback recommendations, or
generic synthetic instructions. Drafts without resolution evidence keep the
existing "No verified resolution evidence..." review warning.

Resolution-backed steps are also sentence-normalized so the paid report reads
like a help-center draft instead of extracted fragments.

The deflection report renderer now uses the item `answer` for proven answers.
That keeps the paid report and structured `faq_result.items[]` aligned while
leaving review-needed sections unchanged.

## Intentional

- This is deliberately not an LLM rewrite. It keeps copy deterministic and
  traceable to `resolution_text` while improving the buyer-facing shape.
- This does not change evidence scoping, output checks, report snapshots,
  paywall behavior, ranking, grouping, or support-contact escalation.
- The review-needed fallback keeps its explicit warning language because there
  is still no resolution evidence for those answers.
- Fallback provenance text remains only as a non-scaffolding last resort when a
  malformed proven item has no usable resolution excerpt; normal proven items
  render from `resolution_text`.

## Deferred

- Future slice: richer multi-sentence article composition if/when we introduce
  a deterministic renderer or evaluated LLM rewrite over scoped evidence.
- Considered hardening: the SaaS demo preflight dotenv entry in `HARDENING.md`
  is unrelated to answer prose and remains parked.
- Parked hardening: none.

## Verification

- `pytest tests/test_extracted_ticket_faq_markdown.py -q` -- 154 passed.
- `pytest tests/test_extracted_ticket_faq_macro_writeback.py tests/test_extracted_ticket_faq_markdown.py -q` -- 161 passed.
- `pytest tests/test_extracted_ticket_faq_markdown.py tests/test_content_ops_faq_report_contract_docs.py tests/test_content_ops_deflection_report.py tests/test_smoke_content_ops_faq_output_proof.py -q` -- 184 passed.
- `pytest tests/test_content_ops_faq_saas_demo_corpus.py tests/test_check_content_ops_faq_search_route_contract.py tests/test_smoke_content_ops_faq_scale_run.py tests/test_smoke_content_ops_faq_lifecycle.py tests/test_smoke_content_ops_cfpb_faq_markdown.py tests/test_extracted_content_ops_execution_smoke.py tests/test_atlas_content_ops_input_provider.py tests/test_extracted_ticket_faq_markdown.py tests/test_content_ops_faq_report_contract_docs.py tests/test_content_ops_deflection_report.py tests/test_smoke_content_ops_faq_output_proof.py -q` -- 379 passed, 1 warning.
- Representative 420-row SaaS sample check -- 8 items, 7 proven,
  `resolution_evidence_scoped=True`, no unscoped proven answers, no old
  `Verified resolution evidence` scaffold in proven answers.
- `scripts/validate_extracted_content_pipeline.sh` run with bash -- passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -- passed.
- `scripts/check_ascii_python.sh` run with bash -- passed.
- Full extracted mirror with live deflection env vars blanked before
  `scripts/run_extracted_pipeline_checks.sh` -- 2912 passed, 10 skipped, 1
  known local failure in
  `tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py::test_script_preflight_uses_atlas_db_settings_fallback`
  from the parked SaaS demo preflight dotenv hardening entry.
- `scripts/local_pr_review.sh` with the current PR body -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| FAQ generator/report renderer | ~75 |
| Tests and frontend example | ~60 |
| Plan doc | ~75 |
| **Total** | **~210** |

Under the 400 LOC soft cap.
