# PR-Resolution-Audit-Artifact-Title

## Why this slice exists

Issue #1882 is a customer-facing naming cleanup for the deflection deliverable.
The paid artifact and PDF still ship the old `Support Ticket Deflection Report`
label even though the public product framing has moved to `Resolution Audit`.
The free teaser should likewise read `Resolution Snapshot`.

Root cause: the public artifact names are duplicated as defaults and test
fixtures across the report producer, PDF renderer, CLI, generated examples, and
hosted snapshot consumers. The first push changed the producer contract but did
not propagate the new Snapshot `title` field through the in-repo proxy/parser
allowlists, so the contract advertised a field that consumers stripped.

This PR fixes the source of the customer-facing defaults and regenerates the
canonical examples so downstream consumers see the same title. It intentionally
leaves internal identifiers alone: `deflection_report`, `deflection.v1`, field
names, file paths, and the bare system/category name remain unchanged.

Diff-size note: the PR is slightly over the 400 LOC soft cap after review fixes
because the title rename changes committed generated examples and the
resolution-live-proof fixture, and the stale CI assertions need to be updated in
the same slice to keep the producer -> contract -> consumer chain green.

## Scope (this PR)

Ownership lane: deflection/report-artifact-naming
Slice phase: Product polish

1. Change the default paid report artifact title/H1 to `Resolution Audit`.
2. Change the default free snapshot title to `Resolution Snapshot`.
3. Change the PDF page header and fallback title to `Resolution Audit`.
4. Propagate the generated Snapshot `title` field through the hosted proxy and
   React result-page parser, with an older-snapshot fallback.
5. Regenerate the canonical report/snapshot examples and update title-string
   tests that assert the old public artifact name.
6. Keep internal identifiers, schema names, contract keys, and bare system name
   unchanged.

### Review Contract

- Acceptance criteria:
  - [ ] `build_deflection_report_artifact(...)`, `render_deflection_report(...)`,
        and `build_deflection_report_model(...)` default to a paid report title
        of `Resolution Audit`.
  - [ ] `build_deflection_snapshot(...)` defaults to `Resolution Snapshot` when
        projecting a report without an explicit snapshot title.
  - [ ] The PDF header and PDF markdown fallback title read `Resolution Audit`.
  - [ ] The CLI default title uses the same `Resolution Audit` constant as the
        producer.
  - [ ] The Snapshot projection contract has a self-describing `title`
        descriptor, and hosted proxy/result-page consumers preserve it.
  - [ ] Older stored snapshots without `title` still render with the
        `Resolution Snapshot` fallback.
  - [ ] Generated report/snapshot example JSON is regenerated and `--check`
        passes.
  - [ ] No internal identifiers are renamed; only customer-facing title strings
        change.
- Affected surfaces: deflection report producer, persisted-model fallback,
  generation plan defaults, CLI defaults, PDF renderer, generated frontend
  examples/types, contract docs, in-repo hosted proxy/result parser, PDF/export
  validators, and targeted smoke tests.
- Risk areas: generated artifact drift, PDF/report mismatch, accidental
  internal schema/key rename.
- Reviewer rules triggered: R1, R2, R5, R9, R10, R11, R12, R14.
- boundary-probe: grep verifies no remaining `Support Ticket Deflection Report`
  in customer-facing source/test/example surfaces changed by this slice, while
  internal `deflection_report` identifiers remain untouched.

### Files touched

- `atlas_brain/deflection_pdf_renderer.py`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json`
- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_deflection_snapshot_example.json`
- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/deflection_report_access.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `extracted_content_pipeline/generation_plan.py`
- `plans/PR-Resolution-Audit-Artifact-Title.md`
- `portfolio-ui/api/content-ops/deflection/atlas-report.js`
- `portfolio-ui/api/content-ops/deflection/snapshot-contract.js`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`
- `portfolio-ui/scripts/faq-deflection-full-report-qa-hosted-smoke.test.mjs`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`
- `portfolio-ui/src/pages/FaqDeflectionResult.tsx`
- `portfolio-ui/src/types/deflectionSnapshot.ts`
- `scripts/build_content_ops_deflection_report.py`
- `scripts/generate_deflection_frontend_contract_types.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_deflection_resolution_live_proof.py`
- `tests/test_content_ops_faq_report_contract_docs.py`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_generate_deflection_frontend_contract_types.py`
- `tests/test_smoke_content_ops_deflection_pdf_export_validators.py`

## Mechanism

The report producer gets explicit public title constants for the two customer
surfaces: paid report = `Resolution Audit`, free snapshot = `Resolution
Snapshot`. Existing optional `title`/`report_title` inputs still override the
default, preserving callers that intentionally provide a custom label.

The PDF renderer uses the same paid title wording for its page header and
fallback generated Markdown title. The renderer still respects a model-provided
title when the artifact has one.

The snapshot/report example generator is run after the producer change so the
canonical JSON examples model production behavior: snapshot title is projected
from the generated report through the real snapshot builder, not hand-edited.
Tests that assert the old public title are updated to the new public title.

The hosted proxy and React fallback parser now preserve the generated Snapshot
`title` field. Missing titles from older stored snapshots fall back to
`Resolution Snapshot`; malformed present titles fail closed.

## Intentional

- Internal identifiers are not renamed. This is a display-name polish slice,
  not a contract/schema migration.
- `Support Ticket Deflection` as a system/category phrase remains allowed where
  it is not the artifact title `Support Ticket Deflection Report`.
- The TypeScript contract generator is run and checked because `title` is a new
  top-level Snapshot field. The generated TS/JS changes are contract-derived,
  not hand-authored consumer maps.
- The in-repo portfolio-ui consumer files are touched here because this ATLAS
  repo vendors the hosted proxy/result-page checks that consume the generated
  Snapshot contract.

## Deferred

- Cross-repo atlas-portfolio copy updates remain in the linked portfolio epic
  (`canfieldjuan/atlas-portfolio#420`) and are not touched from this ATLAS
  source-side slice.
- Broader landing page copy is deferred; this PR only fixes the authoritative
  generated artifact and PDF title surfaces.

Parked hardening: none.

## Verification

- Command: py_compile over the changed Python producer, PDF renderer, generator,
  and focused test files - passed.
- Command: `python scripts/generate_deflection_snapshot_example.py --check` - passed.
- Command: `python scripts/generate_deflection_frontend_contract_types.py --check` - passed.
- Command: `python -m pytest tests/test_generate_deflection_frontend_contract_types.py tests/test_smoke_content_ops_deflection_pdf_export_validators.py tests/test_extracted_content_control_surface_api.py -q` - passed, 201 passed, 1 skipped.
- Command: `npm --prefix portfolio-ui run test:deflection-full-report-qa-hosted-smoke` - passed.
- Command: `python -m pytest tests/test_deflection_snapshot_report_drift.py tests/test_content_ops_deflection_report.py::test_deflection_report_cli_ignores_legacy_max_items_cap tests/test_extracted_content_ops_live_execute_harness.py::test_deflection_report_execute_uncaps_paid_artifact_and_keeps_snapshot_top_n -q` - passed, 15 passed.
- Command: product surface manifest check - passed.
- Command: old customer-facing title grep across changed product surfaces - passed,
  no matches.
- Command: `python -m pytest tests/test_content_ops_deflection_report.py -q` -
  passed, 174 passed.
- Command: `npm --prefix portfolio-ui run test:deflection-atlas-proxy` - passed.
- Command: `npm --prefix portfolio-ui run test:deflection-result` - passed.
- Command: `npm --prefix portfolio-ui run build` - passed.
- Command: `python -m pytest tests/test_content_ops_faq_report_contract_docs.py tests/test_generate_deflection_frontend_contract_types.py tests/test_content_ops_deflection_resolution_live_proof.py -q` - passed, 31 passed.
- Command: `python -m pytest tests/test_generate_deflection_frontend_contract_types.py tests/test_deflection_snapshot_report_drift.py tests/test_run_deflection_full_report_qa_live_runner.py tests/test_smoke_content_ops_deflection_pdf_export_validators.py tests/test_content_ops_deflection_report.py::test_deflection_report_cli_ignores_legacy_max_items_cap -q` - passed, 72 passed.
- Pending before push: local PR review through `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/deflection_pdf_renderer.py` | 9 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md` | 2 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json` | 3 |
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 4 |
| `docs/frontend/content_ops_faq_deflection_snapshot_example.json` | 1 |
| `docs/frontend/content_ops_faq_report_contract.md` | 17 |
| `extracted_content_pipeline/README.md` | 2 |
| `extracted_content_pipeline/deflection_report_access.py` | 7 |
| `extracted_content_pipeline/faq_deflection_report.py` | 23 |
| `extracted_content_pipeline/generation_plan.py` | 3 |
| `plans/PR-Resolution-Audit-Artifact-Title.md` | 194 |
| `portfolio-ui/api/content-ops/deflection/atlas-report.js` | 7 |
| `portfolio-ui/api/content-ops/deflection/snapshot-contract.js` | 2 |
| `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs` | 15 |
| `portfolio-ui/scripts/faq-deflection-full-report-qa-hosted-smoke.test.mjs` | 2 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 4 |
| `portfolio-ui/src/pages/FaqDeflectionResult.tsx` | 45 |
| `portfolio-ui/src/types/deflectionSnapshot.ts` | 7 |
| `scripts/build_content_ops_deflection_report.py` | 3 |
| `scripts/generate_deflection_frontend_contract_types.py` | 8 |
| `tests/test_content_ops_deflection_report.py` | 19 |
| `tests/test_content_ops_deflection_resolution_live_proof.py` | 9 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 2 |
| `tests/test_extracted_content_control_surface_api.py` | 2 |
| `tests/test_generate_deflection_frontend_contract_types.py` | 27 |
| `tests/test_smoke_content_ops_deflection_pdf_export_validators.py` | 6 |
| **Total** | **423** |
