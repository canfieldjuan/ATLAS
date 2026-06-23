# PR-Deflection-Report-Projection-Contract

## Why this slice exists

#1805 is the remaining report-contract half after the snapshot contract
derivation finished. The snapshot arc now has a backend-owned projection
contract, runtime drift proof, generated frontend types, and proxy/portfolio
consumers. The paid `DeflectionStructuredReport` does not: frontend/report
consumers still infer section `data` fields from examples, docs, or local
validators, which recreates the hand-sync pattern that caused the
`top_blind_spots` phantom.

Root cause: `deflection_report_model_contract_shape()` exposes paid section
metadata (`id`, surfaces, limits, `required_data`, and snapshot-safe fields) but
does not publish the paid section data projection or the hosted-consumer-safe
allowlist. This slice fixes the root for the backend contract-definition layer
by adding an ATLAS-owned `report_projection` sibling to `snapshot_projection`.
It does not enforce runtime parity or generate TypeScript yet; those need the
published contract this slice creates.

This slice is over the usual 400 LOC soft budget because the contract is only
useful if it names every paid section, every nested action/question row field,
and the separate hosted-consumer-safe allowlist in the same source shape.
Splitting the field lists from the contract helper would leave the next runtime
enforcement slice comparing against an incomplete source of truth.

## Scope (this PR)

Ownership lane: deflection/report-contract-1805
Slice phase: Production hardening

1. Add a backend-owned `report_projection` contract to
   `deflection_report_model_contract_shape()` for the paid structured report.
2. Describe each paid section's top-level `data.projected_fields`,
   optional projected fields where the producer has branch-dependent output,
   conditional section presence, nested collection item fields for
   `rows`/`items`/`phrases`, and the hosted-consumer-safe allowlist.
3. Pin the action-section contract to both full paid action row fields
   (`repeat_key`, `cluster_id`, scoring, evidence metadata) and the smaller
   hosted result-page allowlist so later consumers do not confuse backend
   availability with safe page payload construction.
4. Add focused contract tests proving the projection is registry-derived where
   possible, covers all sections, includes identity/delta fields in the paid
   projection, and keeps raw evidence fields out of the hosted-consumer-safe
   allowlist.
5. Update the frontend handoff doc to name `report_projection` as the source
   for the future report-model type generator.

### Review Contract

- Acceptance criteria:
  - [ ] `deflection_report_model_contract_shape()` includes a
        `report_projection` object with the current report schema version,
        model fields, section fields, and one projection entry per paid section.
  - [ ] Every `report_projection.sections[]` entry is tied to
        `DEFLECTION_REPORT_SECTION_REGISTRY` metadata and the section order
        matches the registry/report priority order.
  - [ ] Action sections declare full paid item projected fields including
        `repeat_key`, `cluster_id`, `identity_basis`, `identity_confidence`,
        `representative_phrasing`, and `top_evidence`.
  - [ ] Action sections separately declare hosted-consumer-safe item fields
        that exclude raw evidence/phrasing and identity/delta fields until a
        consumer explicitly opts into an export/delta surface.
  - [ ] Branch-dependent `support_tax` annualized cost fields are modeled as
        optional projected fields rather than required fields.
  - [ ] Conditionally-emitted sections (`source_file`, `outcome_diagnostics`)
        carry presence metadata so runtime enforcement/codegen does not treat
        them as always required.
  - [ ] Nested arrays inside action items, including `top_evidence`, are marked
        as nested collections rather than nested objects.
  - [ ] `source_file.source_label` is not hosted-consumer-safe.
  - [ ] Runtime report behavior is unchanged.
- Affected surfaces:
  - `extracted_content_pipeline/faq_deflection_report.py`
  - `docs/frontend/content_ops_faq_report_contract.md`
  - deflection report contract tests
- Risk areas: accidentally widening the hosted result-page payload contract,
  creating a second section registry, or generating report-model types from a
  contract that still omits real runtime fields.
- Reviewer rules triggered: R1, R2, R10, R14; boundary-probe required because
  this publishes allowlist metadata for a privacy/safety boundary.

### Files touched

- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Report-Projection-Contract.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`

## Mechanism

The existing section registry remains the source for paid section identity,
titles, priorities, surfaces, default limits, `required_data`, and
`snapshot_safe_fields`. This slice adds a report projection helper that walks
that registry and attaches section-specific data field metadata.

For simple sections, `projected_fields` names the section `data` keys emitted
by the producer. Every section also carries `presence` metadata so generators
can distinguish always-emitted sections from conditional sections. For
collection sections, the entry also carries a `collection` object naming the
collection field, item type, nested object fields, nested collection fields, and
the hosted-consumer-safe nested fields. Action-oriented sections share one full
paid item field list and a separate `hosted_consumer_safe_fields` list. That
separation is deliberate: the backend paid model can carry export/delta/evidence
material, while hosted result pages must construct a smaller payload rather than
validating and passing through raw fields.

The next slice can compare `build_deflection_report_artifact(...).report_model`
runtime output against this contract. The following generator slice can emit
the frontend report-model artifact from the same shape.

## Intentional

- No runtime enforcement in this slice. This PR creates the backend-owned
  contract so the next slice can enforce it against
  `build_deflection_report_artifact()` without mixing contract definition and
  runtime drift logic.
- No frontend TypeScript generation or `atlas-portfolio` change yet. Starting
  there would recreate the #370 wait-on-ATLAS-artifact problem.
- Hosted-consumer-safe fields intentionally exclude action identity/delta
  fields for now even though the paid backend projection includes them. Monthly
  deltas can consume the full paid model or a dedicated delta artifact later;
  the hosted page should stay allowlist-constructed.
- `source_file.source_label` remains paid-only because the current CLI can pass
  a local/customer path label. A future display-label normalization slice can
  opt it into hosted payloads deliberately.

## Deferred

- ATLAS runtime enforcement: compare representative
  `build_deflection_report_artifact()` report-model output to
  `report_projection`.
- ATLAS generated frontend report-model artifact under `portfolio-ui/src/types`
  once the runtime enforcement is in place.
- Cross-repo/proxy consumption: update `atlas-portfolio` and any proxy contract
  generator to consume the ATLAS-owned paid report-model artifact.

Parked hardening: none.

## Verification

- python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_projection_contract_is_registry_derived tests/test_content_ops_deflection_report.py::test_deflection_report_projection_separates_paid_and_hosted_action_fields tests/test_content_ops_deflection_report.py::test_deflection_report_projection_marks_raw_question_evidence_export_only tests/test_content_ops_deflection_report.py::test_deflection_snapshot_projection_contract_is_registry_derived tests/test_content_ops_deflection_report.py::test_deflection_snapshot_projected_fields_match_runtime_output tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_report_contract_links_example -q -- 6 passed.
- python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_projection_contract_is_registry_derived tests/test_content_ops_deflection_report.py::test_deflection_report_projection_separates_paid_and_hosted_action_fields tests/test_content_ops_deflection_report.py::test_deflection_report_projection_marks_raw_question_evidence_export_only tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_report_contract_links_example -q -- 4 passed after Codex P2 fixes.
- python -m pytest tests/test_content_ops_deflection_report.py tests/test_content_ops_faq_report_contract_docs.py -q -- 160 passed.
- python -m py_compile extracted_content_pipeline/faq_deflection_report.py tests/test_content_ops_deflection_report.py tests/test_content_ops_faq_report_contract_docs.py -- passed.
- git diff --check -- passed.
- bash scripts/validate_extracted_content_pipeline.sh -- passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt -- Atlas runtime import findings: 0.
- bash scripts/check_ascii_python.sh -- passed.
- Pending before push: bash scripts/local_pr_review.sh via scripts/push_pr.sh.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_report_contract.md` | 77 |
| `extracted_content_pipeline/faq_deflection_report.py` | 456 |
| `plans/PR-Deflection-Report-Projection-Contract.md` | 163 |
| `tests/test_content_ops_deflection_report.py` | 162 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 7 |
| **Total** | **865** |
