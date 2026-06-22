# PR-Deflection-Snapshot-Projection-Contract

## Why this slice exists

#1799/#1800 fixed the immediate phantom `top_blind_spots` field by emitting a
real snapshot-safe projection from `top_unresolved_repeats`. The PR body
explicitly deferred the broader structural hardening: the snapshot shape is
still described across hand-synced backend model code, contract prose, example
JSON, and frontend parser types. Tests catch some drift after the fact, but
there is no backend-owned contract that tells consumers which free snapshot
field is derived from which paid section and which allowlisted fields are safe.

Root cause: the free Snapshot projection has runtime allowlist construction, but
the projection contract itself is implicit and scattered. This slice fixes the
root for the backend contract layer by adding a machine-readable snapshot
projection contract derived from the report section registry and pinned in
tests/docs. It does not codegen frontend types, change the snapshot payload, or
alter which paid fields are exposed.

## Scope (this PR)

Ownership lane: deflection/full-report-actionability
Slice phase: Production hardening

1. Add a backend-owned `snapshot_projection` contract to
   deflection_report_model_contract_shape, covering the five free snapshot
   top-level fields and their source sections/field allowlists.
2. Pin `top_blind_spots` to `top_unresolved_repeats.items.{rank,question,ticket_count}`
   in the contract so future section changes cannot silently drift from the
   free snapshot.
3. Add focused tests proving the contract is derived from
   `DEFLECTION_REPORT_SECTION_REGISTRY`, excludes paid action/evidence fields,
   and matches the documented free snapshot shape.
4. Update the frontend contract doc to name the machine-readable contract and
   keep codegen/TS derivation deferred.

### Review Contract

- Acceptance criteria:
  - [ ] deflection_report_model_contract_shape includes a
        `snapshot_projection` object with `schema_version`, top-level snapshot
        field order, and per-field source metadata.
  - [ ] Snapshot section fields are copied from the section registry's
        `snapshot_safe_fields`, not retyped literal allowlists.
  - [ ] `top_blind_spots` is bound to `top_unresolved_repeats`, `items`, and
        only `rank`/`question`/`ticket_count`.
  - [ ] Paid action/evidence fields such as `source_ids`, `top_evidence`,
        `representative_phrasing`, `priority_score`, and identity/delta fields
        do not appear in the snapshot projection contract; `answer`/`steps`
        appear only in the separately gated teaser full-answer contract.
  - [ ] Runtime snapshot payload behavior is unchanged.
- Affected surfaces:
  - `extracted_content_pipeline/faq_deflection_report.py`
  - `docs/frontend/content_ops_faq_report_contract.md`
  - report contract tests
- Risk areas: widening the free Snapshot boundary, changing the process
  freshness contract accidentally, or creating a second hand-maintained
  allowlist that drifts from the registry.
- Reviewer rules triggered: R1, R2, R10, R14; boundary-probe required because
  this exposes privacy-boundary contract metadata.

### Files touched

- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Snapshot-Projection-Contract.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`

## Mechanism

deflection_report_model_contract_shape already exposes the paid report
section registry. This slice adds a sibling `snapshot_projection` block whose
section-backed entries read `snapshot_safe_fields` directly from
`DEFLECTION_REPORT_SECTION_REGISTRY`. Non-section-backed snapshot fields
(`locked_questions` and `teaser`) declare their special projection policy, so
the whole free Snapshot contract is discoverable from one backend-owned shape.

The runtime build_deflection_snapshot payload is left alone. Tests compare
the new contract to the registry and the documented `DeflectionSnapshot` shape
so the contract fails before a free/paid projection drift ships.

## Intentional

- No frontend type generation or `atlas-portfolio` change in this slice. The
  new contract is the backend source needed before a later codegen slice can be
  useful.
- No `schema_version` bump. This is additive metadata on the contract endpoint;
  the persisted `deflection.v1` report model and snapshot payload are unchanged.
- No change to which fields project into the free Snapshot.

## Deferred

- Generate frontend/portfolio types from the backend contract instead of
  maintaining parallel TypeScript shapes.
- Derive committed example JSON fixtures from the backend contract/generator
  instead of hand-editing examples.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_model_contract_shape_requires_version_bump tests/test_content_ops_deflection_report.py::test_deflection_snapshot_projection_contract_is_registry_derived tests/test_content_ops_faq_report_contract_docs.py -q -- 7 passed.
- Command: python -m pytest tests/test_content_ops_deflection_report.py tests/test_deflection_snapshot_report_drift.py tests/test_content_ops_faq_report_contract_docs.py -q -- 166 passed.
- Command: python -m py_compile extracted_content_pipeline/faq_deflection_report.py tests/test_content_ops_deflection_report.py tests/test_content_ops_faq_report_contract_docs.py -- passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh -- passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt -- Atlas runtime import findings: 0.
- Command: bash scripts/check_ascii_python.sh -- passed.
- Command: git diff --check -- passed.
- Command: python scripts/sync_pr_plan.py plans/PR-Deflection-Snapshot-Projection-Contract.md --check -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_report_contract.md` | 29 |
| `extracted_content_pipeline/faq_deflection_report.py` | 124 |
| `plans/PR-Deflection-Snapshot-Projection-Contract.md` | 122 |
| `tests/test_content_ops_deflection_report.py` | 77 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 3 |
| **Total** | **355** |
