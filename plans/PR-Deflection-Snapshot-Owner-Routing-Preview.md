# PR-Deflection-Snapshot-Owner-Routing-Preview

## Why this slice exists

Issue #1874 tracks the product gap found while reviewing the public deflection
Snapshot landing demo: the paid report now routes repeated ticket clusters to a
likely owner/action lane, but the free Snapshot still exposes only question
volume and cost. That means the landing Snapshot can show repeated work, but
not the buyer-facing promise that the report turns repeats into routeable
owner/action work.

Root cause: the Snapshot projection allowlist was intentionally created before
the product-gap owner/action fields existed, so `top_questions` and
`top_blind_spots` project only rank/question/count style fields even when the
paid report model already carries safe owner/action/cost labels. This fixes the
root at the producer contract instead of joining demo-only fields in the
frontend.

Diff size note: this crosses the 400 LOC target because the contract change
must refresh generated JSON/TS artifacts and the exact-shape contract tests in
the same PR; the behavior change itself is limited to the producer projection.

## Scope (this PR)

Ownership lane: deflection/snapshot-owner-routing-preview
Slice phase: Product polish

1. Add a narrow free-Snapshot routing preview to `top_questions` and
   `top_blind_spots`: `owner_lane`, `action_label`, and
   `estimated_support_cost`.
2. Keep raw paid/export-only action metadata out of Snapshot payloads.
3. Regenerate the ATLAS-owned example artifacts and update contract docs/tests.
4. Leave the portfolio render for the follow-up consumer slice.

### Review Contract

Acceptance criteria:
- Snapshot rows expose only the safe routing teaser fields named above.
- `top_blind_spots` derive routing preview from `top_unresolved_repeats.items`.
- `top_questions` derive routing preview from the matching paid action/backlog
  row when present, with a conservative safe fallback when no match exists.
- Snapshot examples and docs are generated from the producer, not hand-authored.
- Paid/export-only fields remain absent from the free Snapshot JSON.

Affected surfaces:
- Free deflection Snapshot payloads.
- Generated frontend contract/example docs consumed by atlas-portfolio.
- Existing paid report model shape is not changed.

Risk areas:
- Privacy boundary: do not leak routing signals, source IDs, evidence, Jira
  templates, raw metadata, product-gap summary prose, or full recommendation
  prose to the free Snapshot.
- Contract drift: Snapshot docs/examples must match producer-generated fields.
- Legacy artifact fallback: non-report-model snapshot projection must keep a
  safe fallback rather than requiring paid report sections.

Reviewer rules triggered: R1, R2, R3, R8, R10, R13, R14.

### Files touched

- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_deflection_snapshot_example.json`
- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Snapshot-Owner-Routing-Preview.md`
- `portfolio-ui/api/content-ops/deflection/report-model-contract.js`
- `portfolio-ui/api/content-ops/deflection/snapshot-contract.js`
- `portfolio-ui/src/types/deflectionReportModel.ts`
- `portfolio-ui/src/types/deflectionSnapshot.ts`
- `scripts/generate_deflection_frontend_contract_types.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`
- `tests/test_generate_deflection_frontend_contract_types.py`

## Mechanism

Extend the Snapshot field allowlists for visible repeat rows with three safe
display fields:

- `owner_lane`: normalized display label.
- `action_label`: short finite label derived from status/fix type, not the full
  paid `recommended_action` prose.
- `estimated_support_cost`: numeric row cost already used in paid action rows.

For report-model projection, build a rank/question keyed lookup from the paid
action/backlog sections. `top_blind_spots` already come from action rows, so
they can project the routing preview directly. `top_questions` merge the
matching routing preview into ranked-question rows; if no action row matches,
they use a conservative fallback (`Support Ops`, `Review repeat`, computed
support cost) so legacy or sparse report models remain safe.

For legacy artifact fallback, compute the same safe routing preview from the
artifact item status/cost plus a conservative owner fallback. The Snapshot
contract docs and generated JSON examples are then refreshed from the producer
path so atlas-portfolio can regenerate from the ATLAS-owned artifact.

## Intentional

- This PR does not expose `recommended_action`; the Snapshot gets a short
  `action_label` so the free surface stays teaser-safe.
- This PR does not expose `owner_category`, `routing_signals`,
  `product_gap_summary`, `customer_vocabulary`, `jira_template`,
  `top_evidence`, `source_ids`, or raw CSV metadata.
- This PR does not update atlas-portfolio rendering; the consumer slice should
  land after the generated contract/example exists on ATLAS main.

## Deferred

- atlas-portfolio consumer slice: regenerate the frontend contract/example and
  render owner/action chips inside the existing Snapshot row cards on
  `/systems/support-ticket-deflection/snapshot`.

Parked hardening: none.

## Verification

- `pytest tests/test_content_ops_deflection_report.py tests/test_content_ops_faq_report_contract_docs.py tests/test_content_ops_faq_deflection_snapshot_example_generator.py tests/test_generate_deflection_frontend_contract_types.py -q` -- 213 passed.
- `python scripts/generate_deflection_snapshot_example.py --check && python scripts/generate_deflection_frontend_contract_types.py --check` -- passed.
- `scripts/validate_extracted_content_pipeline.sh` -- passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -- passed.
- `scripts/check_ascii_python.sh` -- passed.
- Pending before push: `bash scripts/local_pr_review.sh --current-pr-body-file <body-file>`.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 9 |
| `docs/frontend/content_ops_faq_deflection_snapshot_example.json` | 24 |
| `docs/frontend/content_ops_faq_report_contract.md` | 22 |
| `extracted_content_pipeline/faq_deflection_report.py` | 135 |
| `plans/PR-Deflection-Snapshot-Owner-Routing-Preview.md` | 143 |
| `portfolio-ui/api/content-ops/deflection/report-model-contract.js` | 4 |
| `portfolio-ui/api/content-ops/deflection/snapshot-contract.js` | 4 |
| `portfolio-ui/src/types/deflectionReportModel.ts` | 4 |
| `portfolio-ui/src/types/deflectionSnapshot.ts` | 10 |
| `scripts/generate_deflection_frontend_contract_types.py` | 1 |
| `tests/test_content_ops_deflection_report.py` | 90 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 17 |
| `tests/test_generate_deflection_frontend_contract_types.py` | 6 |
| **Total** | **469** |
