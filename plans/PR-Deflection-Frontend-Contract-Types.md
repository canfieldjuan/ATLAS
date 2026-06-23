# PR-Deflection-Frontend-Contract-Types

## Why this slice exists

Issue #1805 is the final snapshot-shape finish-line item after #1803/#1804/#1807:
the backend now exposes and runtime-enforces `snapshot_projection`, but the
in-repo frontend still hand-authors its snapshot TypeScript type in
`portfolio-ui/src/pages/FaqDeflectionResult.tsx`. That is the same class of
hand-sync drift that let `top_blind_spots` exist on one side before the backend
emitted it.

Root cause: the frontend snapshot contract is copied by hand instead of derived
from the backend report-model contract helper. This PR fixes the in-repo root
for `portfolio-ui` by generating the committed TS contract artifact from the
backend contract and adding a CI drift check. The separate `atlas-portfolio`
repo is the remaining cross-repo root and is deferred explicitly below because
this session is scoped to ATLAS.

This slice is over the 400 LOC soft cap because the generator, generated
artifact, negative tests, workflow enrollment, parser type adoption, and plan
are indivisible for a real drift gate. Splitting off the generated artifact or
CI enrollment would recreate the false-green hand-sync gap this slice exists to
close.

Review fix: the backend intentionally omits `source_date_start`,
`source_date_end`, and `source_window_days` when an export lacks a complete
source-date window. The root was the projection contract declaring those fields
as unconditionally required. This PR now marks them optional in the backend
contract, generates optional TS fields, normalizes absent optional values to
`null` in the React parser, and tests both complete-window and no-window
runtime snapshots.

## Scope (this PR)

Ownership lane: deflection/full-report-actionability
Slice phase: Production hardening

1. Generate `portfolio-ui` deflection snapshot TypeScript types from the backend
   `snapshot_projection` contract.
2. Replace the inline result-page snapshot type with the generated
   `DeflectionResultPageSnapshot` subset and preserve all backend-projected
   result-page fields in the parser.
3. Enroll a Python generator `--check` and focused generator tests in CI so
   committed frontend types fail when stale.

### Review Contract

- Acceptance criteria:
  - [ ] `portfolio-ui/src/types/deflectionSnapshot.ts` is generated from the
        backend report-model contract's `snapshot_projection`, not
        hand-authored.
  - [ ] The generator fails closed when the backend projects a new field without
        a frontend TS type mapping.
  - [ ] `FaqDeflectionResult.tsx` imports the generated result-page snapshot
        type and no longer defines its own inline `DeflectionSnapshot`.
  - [ ] The parser preserves all generated result-page fields currently emitted
        by the backend contract (`ticket_count`, non-repeat/window summary
        fields) without changing rendered copy/layout.
  - [ ] Valid backend snapshots without a complete source-date window are still
        accepted; only malformed present date-window values fail closed.
  - [ ] CI runs both the generator tests and the `--check` drift gate.
- Affected surfaces: backend contract generator script, in-repo `portfolio-ui`
  result page TypeScript type, portfolio result-page static test, deflection
  report CI workflow.
- Risk areas: generated artifact staleness, false-green codegen checks,
  unintended client-render behavior changes, cross-repo scope creep.
- Reviewer rules triggered: R1 requirements, R2 test evidence, R6 generated
  artifacts/contracts, R9 CI enrollment, R10 evaluator/checker contract
  behavior, R12 UI/route contract safety, R13 class-fix proof, R14 codebase
  verification.
- boundary-probe: required. This is a contract/checker PR; verify `--check`
  fails on stale output and the generator fails on an unmapped backend field.

### Files touched

- `.github/workflows/atlas_content_ops_deflection_report_checks.yml`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Frontend-Contract-Types.md`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`
- `portfolio-ui/src/pages/FaqDeflectionResult.tsx`
- `portfolio-ui/src/types/deflectionSnapshot.ts`
- `scripts/generate_deflection_frontend_contract_types.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_generate_deflection_frontend_contract_types.py`

## Mechanism

`scripts/generate_deflection_frontend_contract_types.py` imports the backend
report-model contract helper, reads the backend-owned `snapshot_projection`,
and renders a committed TS artifact at
`portfolio-ui/src/types/deflectionSnapshot.ts`.

The generator derives the field lists and object members from the backend
projection. Value types come from an explicit fail-closed map, so a new backend
field without a corresponding frontend type causes the generator/check to fail
instead of widening to `any`/`unknown`.

The backend projection contract also marks conditional source-date-window
summary fields with `optional_projected_fields`. The generator emits those as
optional TS fields and rejects optional fields that are not also projected.

`FaqDeflectionResult.tsx` imports `DeflectionResultPageSnapshot` from the
generated artifact. That type is a generated subset of the full snapshot:
`summary`, `top_questions`, and `top_blind_spots`, which matches the legacy
React page's rendered surface. The parser now constructs the full subset shape
instead of dropping backend-projected-but-unrendered fields. Optional
source-date-window fields normalize to `null` when absent, while malformed
present values still fail closed.

CI runs focused generator tests and `python
scripts/generate_deflection_frontend_contract_types.py --check` in the Atlas
Content Ops deflection report workflow.

## Intentional

- In-repo `portfolio-ui` only. `atlas-portfolio` is a separate repo and remains
  a fast-follow/handoff; this PR records the scope instead of silently claiming
  both repos are covered.
- The generated full `DeflectionSnapshot` includes `locked_questions` and
  `teaser`, but the current React result page consumes only the generated
  result-page subset to avoid widening the page's accepted/rendered surface in a
  type-generation slice.
- The plain-JS proxy projection in `portfolio-ui/api/content-ops/deflection`
  is not refactored to consume generated metadata in this PR. That is a
  separate runtime-projection hardening slice; this PR is the TS contract
  drift gate #1805 asked for.

## Deferred

- `atlas-portfolio` cross-repo codegen/check: issue #1805 names
  `web/src/lib/deflection-snapshot.ts` and
  `web/src/lib/deflection-report-contract.ts` as separate-repo consumers. That
  should be handled by a dedicated atlas-portfolio PR or explicit sync path.
- Shared generated runtime projection for `portfolio-ui/api/.../atlas-report.js`
  remains a follow-up; this PR does not change the hosted proxy behavior.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_generate_deflection_frontend_contract_types.py tests/test_content_ops_deflection_report.py::test_deflection_snapshot_projection_contract_is_registry_derived tests/test_content_ops_deflection_report.py::test_deflection_snapshot_projected_fields_match_runtime_output tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_report_contract_links_example -q`
  -- 8 passed.
- `python scripts/generate_deflection_frontend_contract_types.py --check` --
  passed.
- `npm --prefix portfolio-ui run test:deflection-result` -- passed.
- `npm --prefix portfolio-ui run build` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_deflection_report_checks.yml` | 10 |
| `extracted_content_pipeline/faq_deflection_report.py` | 9 |
| `plans/PR-Deflection-Frontend-Contract-Types.md` | 161 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 61 |
| `portfolio-ui/src/pages/FaqDeflectionResult.tsx` | 78 |
| `portfolio-ui/src/types/deflectionSnapshot.ts` | 95 |
| `scripts/generate_deflection_frontend_contract_types.py` | 260 |
| `tests/test_content_ops_deflection_report.py` | 28 |
| `tests/test_generate_deflection_frontend_contract_types.py` | 74 |
| **Total** | **776** |
