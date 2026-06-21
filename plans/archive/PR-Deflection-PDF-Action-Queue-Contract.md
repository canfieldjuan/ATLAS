# PR-Deflection-PDF-Action-Queue-Contract

## Why this slice exists

#1612 shifted the paid deflection deliverable toward an actionable work queue:
the buyer result page skims the top three fixes while the PDF/email lane needs
a denser, curated view that is still bounded. The model already gives
`priority_fix_queue` a result-page limit, PDF limit, and backlog limit, but the
adjacent action sections still truncate to the result-page top three inside the
producer. That forces later PDF rendering to either stay too thin or invent its
own downstream selection rules.

Root cause: the model contract only encoded multi-surface limits for
`priority_fix_queue`; `top_unresolved_repeats`, `drafted_resolutions`, and
`already_covered_still_recurring` were still result-page-shaped producer
outputs despite being declared for both `web` and `pdf`.

This change fixes the root for this contract layer by making the producer carry
PDF-sized, bounded paid action sections with explicit web/PDF limits. It does
not render the PDF; the renderer slice consumes this contract next.

Diff-size note: this slice exceeds the 400 LOC soft cap because the review fix
is inseparable from the contract change: historical stored `deflection.v1`
reports need the same limit defaults before S4B can safely render from this
contract. Splitting that compatibility guard would leave one pushed commit with
two valid stored shapes under the same schema version.

## Scope (this PR)

Ownership lane: issue-1612/deflection-full-report-delivery-actionability
Slice phase: Vertical slice

1. Add explicit result-page/PDF action-section limits to the paid
   `deflection.v1` model contract for unresolved repeats, drafted resolutions,
   and already-covered-still-recurring callouts.
2. Keep those sections bounded to the PDF limit in the model while preserving
   result-page top-three metadata for downstream renderers.
3. Prove Snapshot remains fail-closed: these paid action sections still have no
   `snapshot_safe_fields`, so new paid payload is absent from free Snapshot.
4. Normalize historical stored `deflection.v1` action sections so missing
   limit fields are backfilled before renderer-facing routes consume them.
5. Max files: 5.

### Review Contract

Acceptance criteria:
- `priority_fix_queue` remains unchanged: result page limit 3, PDF limit 10,
  backlog limit 25.
- `top_unresolved_repeats`, `drafted_resolutions`, and
  `already_covered_still_recurring` expose `result_page_limit` and `pdf_limit`
  in `required_data` and `data`.
- Those three sections keep up to `_ACTION_PDF_LIMIT` items in the paid model,
  not only `_ACTION_RESULT_PAGE_LIMIT`.
- Single-ticket unresolved rows remain excluded from unresolved repeats.
- No action-section field is marked `snapshot_safe_fields`; free Snapshot does
  not include the new paid section payload.
- Historical stored models missing these action-section limit fields are
  backfilled at the storage projection boundary without mutating the raw
  artifact.

Affected surfaces:
- ATLAS deflection report-model producer only.
- ATLAS stored report-model projection for paid/unlocked historical reports.
- Tests that lock the report model contract.
- The committed frontend contract example generated from the producer shape.

Risk areas:
- Accidentally widening the free Snapshot projection.
- Breaking downstream consumers that rely on `top_item_count`.
- Making renderer-specific choices in the producer.

- Reviewer rules triggered: R1, R2, R3, R10, R13, R14.

### Files touched

- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `extracted_content_pipeline/deflection_report_access.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-PDF-Action-Queue-Contract.md`
- `tests/test_content_ops_deflection_report.py`

## Mechanism

The existing action constants stay canonical:

```python
_ACTION_RESULT_PAGE_LIMIT = 3
_ACTION_PDF_LIMIT = 10
_ACTION_BACKLOG_LIMIT = 25
```

This slice applies the same explicit limit pattern already used by
`priority_fix_queue` to the three secondary action sections. The producer keeps
up to `max(_ACTION_RESULT_PAGE_LIMIT, _ACTION_PDF_LIMIT)` rows for those paid
sections and includes both limit values in section data. The section registry's
`required_data` is updated so consumers can rely on those fields instead of
hardcoding limits.

Snapshot remains allowlist-only by construction. These paid action sections
continue to define no `snapshot_safe_fields`, so `_snapshot_report_model_projection`
drops them from the free projection.

## Intentional

- No PDF renderer changes in this PR. S4A defines the bounded contract; S4B
  should render from it.
- `top_item_count` remains the count of included model rows for compatibility.
  Renderers use `result_page_limit` or `pdf_limit` to decide how many to show.
- The evidence export/backlog table remains capped separately at the existing
  default 25.
- The slice does not add S6 delta/writeback identity fields.

## Deferred

- S4B: render the PDF action sections from these explicit limits.
- S4C: add cross-surface PDF/result-page assertions that prove renderer output
  agrees with the paid model while Snapshot stays closed.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_content_ops_deflection_report.py -q` -- 127 passed.
- `python -m pytest tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_deflection_example_matches_producer_shape -q` -- 1 passed.
- `python -m pytest tests/test_content_ops_faq_report_contract_docs.py -q` -- 5 passed.
- `python -m pytest tests/test_content_ops_deflection_report.py::test_stored_deflection_report_model_backfills_legacy_action_limits tests/test_content_ops_deflection_report.py::test_stored_deflection_report_model_tolerates_legacy_and_schema_drift tests/test_content_ops_deflection_report.py::test_in_memory_deflection_report_store_round_trips_report_model -q` -- 3 passed.
- `python -m py_compile extracted_content_pipeline/faq_deflection_report.py extracted_content_pipeline/deflection_report_access.py tests/test_content_ops_deflection_report.py` -- passed.
- `bash scripts/validate_extracted_content_pipeline.sh` -- passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -- passed.
- `bash scripts/check_ascii_python.sh` -- passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/PR-Deflection-PDF-Action-Queue-Contract.md` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 16 |
| `extracted_content_pipeline/deflection_report_access.py` | 47 |
| `extracted_content_pipeline/faq_deflection_report.py` | 30 |
| `plans/PR-Deflection-PDF-Action-Queue-Contract.md` | 143 |
| `tests/test_content_ops_deflection_report.py` | 219 |
| **Total** | **455** |
